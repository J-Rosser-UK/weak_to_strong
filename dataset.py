import sys

sys.path.append("prm800k")
sys.path.append("prm800k/prm800k")
sys.path.append("prm800k/prm800k/grading")

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Dataset
from typing import Any, Literal, Union
from textwrap import dedent
from inspect_ai.model import get_model

import os
import json
import random
import time
import hashlib
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, Dataset
from inspect_ai.model import (
    GenerateConfig,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai._eval.eval import eval
from inspect_ai.scorer import (
    Score,
    scorer,
    accuracy,
)

from inspect_ai.dataset._dataset import MemoryDataset
from inspect_ai.dataset._util import data_to_samples, record_to_sample_fn
from inspect_ai.dataset import FieldSpec, RecordToSample
from prm800k.prm800k.grading.grader import grade_answer
from typing import Any, Union, Literal
from inspect_ai._eval.eval import eval
from inspect_ai.solver import solver, Solver


from metrics import ci_lower, ci_upper, median
from inspect_ai.scorer import scorer, Score, accuracy
import re


class MATH:
    """
    Loads the MATH dataset from local JSONL splits in `prm800k/prm800k/math_splits/`.
    For each record with fields [problem, answer, solution, subject, level, unique_id],
    it produces a prompt asking for the final answer in <answer></answer> tags and sets
    the correct target as the official gold answer.
    """

    def __init__(
        self,
        args=None,
        split: Union[
            Literal["train_half_1"], Literal["train_half_2"], Literal["test"]
        ] = "train_half_1",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:
        """
        :param args: Additional arguments (often used for random_seed, or other config).
        :param split: Which split to load. Allowed values: "train" or "test".
        :param shuffle: Whether to shuffle the dataset.
        :param limit: How many samples to limit in this split.
        """

        models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
        self.weak_student = "claude-3-haiku-20240307"
        self.strong_teacher = "claude-3-5-sonnet-20240620"

        self.split = split
        self.args = args

        self.num_few_shot = 5

        # load local lines (train.jsonl or test.jsonl), filter, shuffle, etc.
        train_first_half, train_second_half, test = self.build_dataset(
            path="./prm800k/prm800k/math_splits",
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=(self.args.random_seed if self.args else 42),
        )

        self.split_mapping = {
            "train_half_1": train_first_half,
            "train_half_2": train_second_half,
            "test": test,
        }

        self.raw_dataset = self.split_mapping[split]

        self.dataset = MemoryDataset(
            samples=data_to_samples(
                self.raw_dataset,
                record_to_sample_fn(self._record_to_sample),
                auto_id=False,
            ),
            # name=Path(path).stem,
        )

        weak_labelled_dataset = {}  # input, answer

        # select 5 random samples from the training set
        random_samples = self.select_random_gold_samples(
            train_first_half, self.num_few_shot
        )

        print("RANDOM SAMPLES", random_samples)

    def select_random_gold_samples(self, dataset, num_samples):
        random_samples = random.sample(dataset, num_samples)
        return random_samples

    def select_random_weak_samples(self, dataset, num_samples):
        random_samples = random.sample(dataset, num_samples)
        return random_samples

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a MATH record to a Sample. The problem is turned into a prompt
        asking for the final answer in <answer> tags, and the official 'answer'
        field is used as the correct target.
        """

        print(record)

        prompt = dedent(
            f"""{record["problem"]}

            Please enclose your final answer in <answer></answer> tags.
            """
        ).strip()

        # In MATH, the official 'answer' is the correct solution in string form,
        # e.g. "343" or "1/2", etc.

        gold_answer = record["answer"]

        # Additional info you want to store in metadata
        metadata = {
            "unique_id": record["unique_id"],
            "solution": record["solution"],
            "subject": record["subject"],
            "level": record["level"],
        }

        return Sample(
            input=prompt,
            target=gold_answer,
            metadata=metadata,
        )

    @solver
    def few_shot_solver(self) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            strong_model = get_model(f"anthropic/{self.strong_teacher}")
            weak_model = get_model(f"anthropic/{self.weak_student}")
            state.messages = []

            while not state.completed:

                for sample in self.select_random_gold_samples(
                    self.raw_dataset, self.num_few_shot
                ):
                    print(sample)
                    user_message = ChatMessageUser(content=sample["problem"])
                    state.messages.append(user_message)

                    assistant_message = ChatMessageAssistant(
                        content=sample["solution"]
                        + "\n <answer>"
                        + sample["answer"]
                        + "</answer>"
                    )
                    state.messages.append(assistant_message)

                message = ChatMessageUser(content=state.input)

                state.messages.append(message)

                output = await strong_model.generate(state.messages, state.tools)
                state.output = output
                state.messages.append(output.message)

                state.completed = True

            # print("generate", state.output.completion)

            return state

        return solve

    @task
    def match_task(self):
        """
        Returns a standard `Task` that uses `match_solver()` (simple forward)
        and `multi_choice_match()` or a custom scorer. You can customize or
        rename as needed. For MATH, you likely want to parse the <answer>…</answer>
        text and check correctness. The simplest approach uses `llm_match()`
        style logic or a custom text-compare approach.
        """
        return Task(
            time_limit=(self.args.task_timeout if self.args else 60),
            name=self.__class__.__name__,
            dataset=self.dataset,
            solver=self.few_shot_solver(),  # from the parent class
            scorer=self.llm_math_scorer(),  # implement a MATH-specific scorer
            config=GenerateConfig(temperature=0.5),
        )

    def build_dataset(
        self,
        path: str,
        sample_fields: FieldSpec | RecordToSample | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Similar to `filtered_hf_dataset`, but loading from a local JSONL file
        in `path/{split}.jsonl`. Then applies the same logic: filter, shuffle,
        limit, store a unique_id, etc.
        """

        train_file_path = os.path.join(path, f"train.jsonl")

        # Read lines from that file
        with open(train_file_path, "r") as f:
            raw_train_data = [json.loads(line.strip()) for line in f]

        # Assign a 'unique_id' to each record if not present
        for record in raw_train_data:
            if not record.get("unique_id"):
                # For MATH, unique_id is often already in JSON, but as a fallback:
                record_content = json.dumps(record, sort_keys=True).encode("utf-8")
                unique_id = hashlib.sha256(record_content).hexdigest()
                record["unique_id"] = unique_id

        test_file_path = os.path.join(path, f"test.jsonl")

        # Read lines from that file
        with open(test_file_path, "r") as f:
            raw_test_data = [json.loads(line.strip()) for line in f]

        # Assign a 'unique_id' to each record if not present
        for record in raw_test_data:
            if not record.get("unique_id"):
                # For MATH, unique_id is often already in JSON, but as a fallback:
                record_content = json.dumps(record, sort_keys=True).encode("utf-8")
                unique_id = hashlib.sha256(record_content).hexdigest()
                record["unique_id"] = unique_id

        # split the training data into two halves
        raw_train_data_first_half = raw_train_data[: len(raw_train_data) // 2]
        raw_train_data_second_half = raw_train_data[len(raw_train_data) // 2 :]

        # if shuffle:
        #     random.seed(seed)
        #     random.shuffle(raw_train_data)

        # Convert dict records to `Sample`s
        return raw_train_data_first_half, raw_train_data_second_half, raw_test_data

    @staticmethod
    @scorer(metrics=[accuracy(), ci_lower(), ci_upper(), median()])
    def llm_math_scorer():
        """
        A custom MATH scorer. It does the following:
         - Extracts the <answer>...</answer> substring from model output
         - Calls `grade_answer(model_answer, gold_answer)`
         - Returns 1 if correct, 0 otherwise
        """

        def parse_response_for_answer(response: str) -> str:
            # parse out <answer> ... </answer>
            match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return ""

        async def score(state, target):
            # The gold solution is in `target.text`
            gold_answer = target.text
            # The model output is in `state.output.completion`
            completion_text = state.output.completion

            # Attempt to parse <answer>some_value</answer>
            predicted_answer = parse_response_for_answer(completion_text)
            if not predicted_answer:
                # If nothing is found, consider it incorrect
                return Score(
                    name="math_scorer",
                    value=0.0,
                    answer=completion_text,
                    explanation="No <answer> tags found",
                )

            # Now use the official grader
            is_correct = grade_answer(predicted_answer, gold_answer)
            val = 1.0 if is_correct else 0.0

            return Score(
                name="math_scorer",
                value=val,
                answer=completion_text,
                explanation=f"Used `grade_answer`; correct={is_correct}",
            )

        return score


if __name__ == "__main__":

    tasks = []

    math_task = MATH(args=None, split="train_half_1", shuffle=False)
    tasks.append(math_task.match_task())

    models = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    models = [f"anthropic/{model}" for model in models]
    log_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))
    results = eval(
        tasks,
        model=models,
        limit=5,
        log_dir=f"./logs/{log_timestamp_str}/logs",  # specify where logs are stored
        log_format="json",  # choose log format ("eval" or "json")
        score=True,  # ensure scoring is enable
        max_tasks=500,
    )
