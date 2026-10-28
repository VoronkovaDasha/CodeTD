# This file is adapted from bigcode-evaluation-harness
# (https://github.com/bigcode-project/bigcode-evaluation-harness),
# licensed under the Apache License, Version 2.0. A copy of the license is
# in LICENSES/Apache-2.0.txt (see also http://www.apache.org/licenses/LICENSE-2.0).
#
# Modifications from the original:
#   - Task classes rewritten as standalone classes sharing a small base class
#     (CodeTask); prompts are cached per example.
#   - HumanEval: optional trailing newline after the prompt (`new_line`);
#     `process_results` can also return per-sample results (`return_errs`).
#   - MBPP: the docstring prompt is replaced by the few-shot "[BEGIN] ... [DONE]"
#     format (Austin et al., 2021) with one or two shots from the prompt split
#     (task ids 2 and 3); task 493 shows only its last test.
#   - MBPP post-processing: outputs are cut at the first "[DONE]" instead of at
#     the stop words; `get_valid_output` added (keeps "[DONE]", used for gen_len).

from datasets import load_dataset
from transformers import StoppingCriteria
from .code_eval import compute_code_eval


class CodeTask:
    """Shared logic for the code-generation tasks.
    """
    dataset_path = None

    def __init__(self, new_line):
        self.new_line = new_line
        self.dataset = load_dataset(path=self.dataset_path, split="test")
        # Default settings for compute_code_eval.
        self.k = [1, 10, 100]
        self.num_workers = 4
        self.timeout = 3.0
        self._prompt_cache = {}

    def prompt_for_idx(self, idx):
        """Prompt of dataset[idx], built on first use and then reused.
        """
        if idx not in self._prompt_cache:
            self._prompt_cache[idx] = self.get_prompt(self.dataset[idx])
        return self._prompt_cache[idx]

    def get_dataset(self):
        return self.dataset

    def get_prompt(self, doc):
        raise NotImplementedError

    def get_reference(self, doc):
        raise NotImplementedError

    def postprocess_generation(self, generation, idx):
        raise NotImplementedError

    def process_results(self, generations, references, return_errs=False):
        results, results_ext = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        if return_errs:
            return results, results_ext
        return results


class HumanEvalTask(CodeTask):
    dataset_path = "openai/openai_humaneval"

    def __init__(self, new_line):
        super().__init__(new_line)
        self.stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]
        self.num_workers = 16

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def get_prompt(self, example):
        prompt = example["prompt"].strip()
        if self.new_line:
            prompt += "\n"
        return prompt

    def get_reference(self, example):
        test_func = example["test"]
        entry_point = f"check({example['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx):
        """Prompt plus generated code, cut at the first stop word."""
        prompt = self.prompt_for_idx(idx)
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)


MBPP_SHOTS = [
    {
        "task_id": 2,
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
        "code": "def similar_elements(test_tup1, test_tup2):\r\n"
                "  res = tuple(set(test_tup1) & set(test_tup2))\r\n"
                "  return (res) ",
    },
    {
        "task_id": 3,
        "text": "Write a python function to identify non-prime numbers.",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "code": "import math\r\n"
                "def is_not_prime(n):\r\n"
                "    result = False\r\n"
                "    for i in range(2,int(math.sqrt(n)) + 1):\r\n"
                "        if n % i == 0:\r\n"
                "            result = True\r\n"
                "    return result",
    },
]


class MBPPTask(CodeTask):
    dataset_path = "google-research-datasets/mbpp"

    def __init__(self, new_line, two_shot=False):
        # MBPP (test split, 500 problems) follows the few-shot [BEGIN]/[DONE]
        # format. We use one shot by default (task 2) or two shots with --two_shot (tasks 2 and 3),
        # both from the MBPP prompt split.
        super().__init__(new_line)
        self.stop_words = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]
        self.two_shot = two_shot

    @staticmethod
    def _task_header(text, tests):
        """Task description, its tests and the opening [BEGIN] delimiter."""
        return (
            f"You are an expert Python programmer, and here is your task: {text} "
            f"Your code should pass these tests:\n\n{tests}\n[BEGIN]"
        )

    def _format_shot(self, shot):
        tests = "\n".join(shot["test_list"])
        code = shot["code"].replace("\r", "")
        return self._task_header(shot["text"], tests) + f"\n{code}\n[DONE]\n"

    def get_few_shot(self, n_shots):
        return "\n".join(self._format_shot(s) for s in MBPP_SHOTS[:n_shots])

    def get_prompt(self, doc):
        tests = "\n".join(doc["test_list"])
        # Problem with task_id 493 has very long input.
        # We keep only the last test to process the problem.
        if doc["task_id"] == 493:
            tests = "\n".join(doc["test_list"][-1:])
        prompt = self._task_header(doc["text"], tests)
        if self.new_line:
            prompt += "\n"
        few_shot = self.get_few_shot(2 if self.two_shot else 1)
        return few_shot + prompt

    def get_reference(self, doc):
        return "\n".join(doc["test_list"])

    def get_valid_output(self, generation, idx):
        """Prompt plus generated code up to and including the first "[DONE]".
        """
        prompt = self.prompt_for_idx(idx)
        generation = generation[len(prompt) :]
        token_idx = generation.find("[DONE]")
        if token_idx != -1:
            # Keep the "[DONE]" delimiter for MBPP.
            generation = generation[: token_idx + len("[DONE]")]
        return prompt + generation

    def postprocess_generation(self, generation, idx):
        """Generated code used for evaluation: cut before the first "[DONE]".
        """
        prompt = self.prompt_for_idx(idx)
        generation = generation[len(prompt) :]
        generation = generation.split("[DONE]")[0].strip()
        return generation


class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                stop_string in decoded_generation for stop_string in self.eof_strings
            )
        self.check_fn = check_fn

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        return all(self.check_fn(decoded_generation) for decoded_generation in decoded_generations)

