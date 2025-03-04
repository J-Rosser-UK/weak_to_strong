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

import os
import json
import random
import time
import hashlib
from pathlib import Path

from inspect_ai.dataset._dataset import MemoryDataset
from inspect_ai.dataset._util import data_to_samples, record_to_sample_fn
from inspect_ai.dataset import FieldSpec, RecordToSample
from prm800k.prm800k.grading.grader import grade_answer
from typing import Any, Union, Literal
from inspect_ai._eval.eval import eval

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
        split: Union[Literal["train"], Literal["test"]] = "train",
        shuffle: bool = True,
        limit: int = 1000,
    ) -> Dataset:
        """
        :param args: Additional arguments (often used for random_seed, or other config).
        :param split: Which split to load. Allowed values: "train" or "test".
        :param shuffle: Whether to shuffle the dataset.
        :param limit: How many samples to limit in this split.
        """
        self.split = split
        self.args = args

        # A mapping from user-chosen "train"/"test" to your local JSONL files
        split_mapping = {
            "train": "train",
            "test": "test",
        }

        # load local lines (train.jsonl or test.jsonl), filter, shuffle, etc.
        self.dataset = self.filtered_local_dataset(
            path="./prm800k/prm800k/math_splits",
            split=split,
            split_mapping=split_mapping,
            sample_fields=self._record_to_sample,
            shuffle=shuffle,
            seed=(self.args.random_seed if self.args else 42),
            limit=limit,
        )

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a MATH record to a Sample. The problem is turned into a prompt
        asking for the final answer in <answer> tags, and the official 'answer'
        field is used as the correct target.
        """

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
            # solver=self.match_solver(),  # from the parent class
            scorer=self.llm_math_scorer(),  # implement a MATH-specific scorer
            config=GenerateConfig(temperature=0.5),
        )

    def benchmark_filter(self, example: dict) -> bool:
        """
        If needed, add any filtering logic to skip certain records.
        Return True to keep, False to filter out.
        """
        # Example: skip empty problems
        if not example.get("problem", "").strip():
            return False
        return True

    def filtered_local_dataset(
        self,
        path: str,
        split: str,
        split_mapping: dict,
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

        data_to_sample = record_to_sample_fn(sample_fields)

        # Map user-provided "train"/"test" to the local file name
        actual_split_file = split_mapping[split]
        file_path = os.path.join(path, f"{actual_split_file}.jsonl")

        print("file", file_path)

        # Read lines from that file
        with open(file_path, "r") as f:
            raw_data = [json.loads(line.strip()) for line in f]

        # Filter the data
        raw_data = [d for d in raw_data if self.benchmark_filter(d)]

        if shuffle:
            random.seed(seed)
            random.shuffle(raw_data)

        # Assign a 'unique_id' to each record if not present
        for record in raw_data:
            if not record.get("unique_id"):
                # For MATH, unique_id is often already in JSON, but as a fallback:
                record_content = json.dumps(record, sort_keys=True).encode("utf-8")
                unique_id = hashlib.sha256(record_content).hexdigest()
                record["unique_id"] = unique_id

        print(f"Final {split} dataset length: {len(raw_data)}")

        # Convert dict records to `Sample`s
        return MemoryDataset(
            samples=data_to_samples(raw_data, data_to_sample, auto_id=False),
            name=Path(path).stem,
            location=str(file_path),
        )

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

    math_task = MATH(args=None, split="train", shuffle=False, limit=10)
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
