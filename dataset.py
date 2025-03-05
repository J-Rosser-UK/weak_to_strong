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
from rich import print
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

from inspect_ai.scorer import scorer, Score, accuracy
import re

from dotenv import load_dotenv

load_dotenv(override=True)


class MATH:
    """
    Loads the MATH dataset from local JSONL splits in `prm800k/prm800k/math_splits/`.
    For each record with fields [problem, answer, solution, subject, level, unique_id],
    it produces a prompt asking for the final answer in <answer></answer> tags and sets
    the correct target as the official gold answer.
    """

    def __init__(self, limit: int = 5, num_few_shot: int = 5) -> Dataset:
        """
        :param args: Additional arguments (often used for random_seed, or other config).
        :param split: Which split to load. Allowed values: "train" or "test".
        :param shuffle: Whether to shuffle the dataset.
        :param limit: How many samples to limit in this split.
        """
        self.log_timestamp_str = str(time.strftime("%Y%m%d-%H%M%S"))
        self.log_dir = f"./logs/{self.log_timestamp_str}/logs"
        self.limit = limit

        self.num_few_shot = num_few_shot
        self.random_seed = 42
        self.task_timeout = 60

        # load local lines (train.jsonl or test.jsonl), filter, shuffle, etc.
        self.train_half_1_raw, self.train_half_2_raw, self.test_raw = (
            self.build_dataset(
                path="./prm800k/prm800k/math_splits",
            )
        )

        self.few_shot_dataset_map = {
            "train_half_1": self.train_half_1_raw,
            "train_half_2": self.train_half_2_raw,
            "test": self.test_raw,
        }

        self.train_half_1_dataset = MemoryDataset(
            samples=data_to_samples(
                self.train_half_1_raw,
                record_to_sample_fn(self._record_to_sample),
                auto_id=False,
            ),
            name="train_half_1",
        )

        self.train_half_2_dataset = MemoryDataset(
            samples=data_to_samples(
                self.train_half_2_raw,
                record_to_sample_fn(self._record_to_sample),
                auto_id=False,
            ),
            name="train_half_2",
        )

        self.ld = {"model": {"dataset_name": []}}

    def run(self, weak_teacher, strong_student):

        self.weak_teacher = weak_teacher
        self.strong_student = strong_student

        tasks = [
            Task(
                time_limit=self.task_timeout,
                name=self.__class__.__name__,
                dataset=self.train_half_1_dataset,
                solver=self.few_shot_gold_solver(self.weak_teacher, "train_half_1"),
                scorer=self.grader_scorer(),
                config=GenerateConfig(temperature=0.5),
            ),
            Task(
                time_limit=self.task_timeout,
                name=self.__class__.__name__,
                dataset=self.train_half_2_dataset,
                solver=self.few_shot_gold_solver(self.weak_teacher, "train_half_2"),
                scorer=self.grader_scorer(),
                config=GenerateConfig(temperature=0.5),
            ),
        ]

        results = eval(
            tasks,
            model=[self.weak_teacher, self.strong_student],
            limit=self.limit,
            log_dir=self.log_dir,  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
            max_tasks=500,
        )

        # 'results' is a list of EvalLog objects (usually one per task)
        # Each EvalLog contains metrics for the entire task/dataset.
        gold_model_metrics = self._get_accuracy_from_results(results)

        # print(self.ld)

        tasks = [
            Task(
                time_limit=self.task_timeout,
                name=self.__class__.__name__,
                dataset=self.train_half_2_dataset,
                solver=self.few_shot_weak_solver(
                    self.strong_student, self.ld[self.weak_teacher]["train_half_2"]
                ),  # from the parent class
                scorer=self.grader_scorer(),  # implement a MATH-specific scorer
                config=GenerateConfig(temperature=0.5),
            )
        ]

        results = eval(
            tasks,
            model=self.strong_student,
            limit=self.limit,
            log_dir=self.log_dir,  # specify where logs are stored
            log_format="json",  # choose log format ("eval" or "json")
            score=True,  # ensure scoring is enable
            max_tasks=500,
        )

        w2s_model_metrics = self._get_accuracy_from_results(results)

        # calculate the performance gap between the weak teacher and the strong student
        weak_floor = gold_model_metrics[self.weak_teacher]["train_half_2"]["accuracy"]
        strong_ceiling = gold_model_metrics[self.strong_student]["train_half_2"][
            "accuracy"
        ]
        w2s_level = w2s_model_metrics[self.strong_student]["train_half_2"]["accuracy"]

        performance_gap_recovered = (w2s_level - weak_floor) / (
            strong_ceiling - weak_floor + 1e-9
        )

        return performance_gap_recovered, weak_floor, strong_ceiling, w2s_level

    def _get_accuracy_from_results(self, results):
        model_metrics = {}  # dictionary to hold info for each model

        for res in results:

            # 1) Get the model name and task name
            model_name = str(getattr(res.eval, "model", ""))
            dataset_name = str(getattr(res.eval.dataset, "name", ""))

            # 2) Initialize defaults (or None) for each metric
            accuracy = None

            # 3) Check if results and scores exist
            if res.results and res.results.scores:
                for score in res.results.scores:
                    if score.metrics:
                        # 4) For each metric, check if it exists and store its value
                        if "accuracy" in score.metrics:
                            accuracy = score.metrics["accuracy"].value

            # 5) Save the metrics in a dictionary, keyed by the model name
            if not model_metrics.get(model_name):
                model_metrics[model_name] = {dataset_name: {}}

            if not model_metrics[model_name].get(dataset_name):
                model_metrics[model_name][dataset_name] = {}

            model_metrics[model_name][dataset_name] = {
                "accuracy": accuracy,
            }

        return model_metrics

    def _select_random_samples(self, dataset, num_samples):
        if len(dataset) < num_samples:
            return dataset
        random_samples = random.sample(dataset, num_samples)
        return random_samples

    def _remove_boxed_tags(self, text):
        # This regex captures \boxed{...} ensuring it gets the full content inside
        return re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text)

    def _record_to_sample(self, record: dict[str, Any]) -> Sample:
        """
        Convert a MATH record to a Sample. The problem is turned into a prompt
        asking for the final answer in <answer> tags, and the official 'answer'
        field is used as the correct target.
        """

        prompt = dedent(
            f"""<user_problem>{record["problem"]}</user_problem>

            Please proceed with analyzing and solving the given problem and enclose your final answer in <answer></answer> tags.
            """
        ).strip()

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
    def few_shot_gold_solver(self, model_name, few_shot_dataset_name) -> Solver:
        model = get_model(model_name)
        few_shot_dataset = self.few_shot_dataset_map[few_shot_dataset_name]

        async def solve(state: TaskState, generate: Generate) -> TaskState:

            state.messages = []

            state.messages.append(ChatMessageSystem(content=self.system_prompt))

            while not state.completed:

                for sample in self._select_random_samples(
                    few_shot_dataset, self.num_few_shot
                ):

                    user_message = ChatMessageUser(
                        content=dedent(
                            f"""<user_problem>{sample["problem"]}</user_problem>

                        Please proceed with analyzing and solving the given problem and enclose your final answer in <answer></answer> tags."""
                        ).strip()
                    )
                    state.messages.append(user_message)

                    assistant_message = ChatMessageAssistant(
                        content=self._remove_boxed_tags(sample["solution"])
                        + "\n <answer>"
                        + sample["answer"]
                        + "</answer>"
                    )
                    state.messages.append(assistant_message)

                message = ChatMessageUser(content=state.input)

                state.messages.append(message)

                output = await model.generate(state.messages, state.tools)
                state.output = output
                state.messages.append(output.message)

                if not self.ld.get(model_name):
                    self.ld[model_name] = {few_shot_dataset_name: []}
                elif not self.ld[model_name].get(few_shot_dataset_name):
                    self.ld[model_name][few_shot_dataset_name] = []

                self.ld[model_name][few_shot_dataset_name].append(
                    {
                        "problem": state.input,
                        "output": output.completion,
                        "target": state.target.text,
                    }
                )

                state.completed = True

            return state

        return solve

    @solver
    def few_shot_weak_solver(self, model_name, labelled_dataset) -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:

            model = get_model(model_name)
            state.messages = []

            state.messages.append(ChatMessageSystem(content=self.system_prompt))

            while not state.completed:

                for sample in self._select_random_samples(
                    labelled_dataset, self.num_few_shot
                ):

                    user_message = ChatMessageUser(content=sample["problem"])
                    state.messages.append(user_message)

                    assistant_message = ChatMessageAssistant(content=sample["output"])
                    state.messages.append(assistant_message)

                message = ChatMessageUser(content=state.input)

                state.messages.append(message)

                output = await model.generate(state.messages, state.tools)
                state.output = output
                state.messages.append(output.message)

                state.completed = True

            return state

        return solve

    def build_dataset(
        self,
        path: str,
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
    @scorer(metrics=[accuracy()])
    def grader_scorer():
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

    @property
    def system_prompt(self):
        return """You are an AI assistant designed to help users solve a wide variety of problems. Your task is to analyze the given problem, think through the solution process, and provide a clear, accurate answer.

Here is the problem submitted by the user will be placed in <user_problem></user_problem> tags.

Please follow these steps to address the problem:

1. Carefully read and analyze the problem.
2. In <problem_analysis> tags, break down your thought process and approach to solving the problem. This step is crucial for complex problems that require multiple steps or considerations.
3. Solve the problem based on your analysis.
4. Format your final answer and enclose it within <answer> tags.

Important: Always ensure that your final answer is enclosed within the <answer> tags, regardless of the complexity or length of your response.

Example of the expected output structure:

<problem_analysis>
1. Problem Summary: [Restate the problem in your own words]
2. Key Information and Variables: [List the important data points and variables]
3. Solution Steps:
   - [Step 1]
   - [Step 2]
   - [Step 3]
4. Potential Challenges or Alternative Approaches: [Discuss any difficulties or other methods to consider]
</problem_analysis>

<answer>
[Your final, concise answer to the problem goes here. This should be the direct response to the user's question or the solution to their problem.]
</answer>

Please proceed with analyzing and solving the given problem when given."""


if __name__ == "__main__":

    models = [
        "claude-3-5-sonnet-latest",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    models = [f"anthropic/{model}" for model in models]

    weak_teacher = models[3]
    strong_student = models[0]

    math_task = MATH(limit=50, num_few_shot=5)
    performance_gap_recovered, weak_floor, strong_ceiling, w2s_level = math_task.run(
        weak_teacher, strong_student
    )

    print("Performance gap recovered:", performance_gap_recovered)
