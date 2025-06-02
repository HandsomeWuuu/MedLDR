"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
import numpy as np

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()

def parse_gold_icd(sol):
    return sol.split(';')

def extract_answer_easy(content):
    # 1. 先查看是不是标准格式
    # 2. 如果是标准格式，直接提取答案
    # 3. 如果不是标准格式，去掉 think 部分，然后提取答案
    # 4. 如果还是没有，返回空列表

    format_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    format_match = re.match(format_pattern, content, re.DOTALL | re.MULTILINE)
    if format_match is not None:
        # 处理有 <answer> 的情况 -- 必须要标标准准的，不然会生成大量的重复内容
        answer_pattern = r"<answer>(.*?)</answer>$"
        answer_match = re.search(answer_pattern, content, re.DOTALL | re.MULTILINE)

        answer_content = answer_match.group(1) if answer_match else ""
        if len(re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL | re.MULTILINE)) > 1:
            answer_content = ""

    else:
        # 处理没有 <answer> 但是有正确回答的情况 -- 大多数是这样的
        think_pattern = r"^<think>(.*?)</think>"
        # 拿掉 think_pattern 的部分剩下的就是 answer 的
        answer_content = re.sub(think_pattern, '', content, flags=re.DOTALL)

    try:
        json_data = json.loads(answer_content)
        predictions = json_data.get('diagnoses', [])

        if not isinstance(predictions,list):
            predictions = []
    except:
        predictions = []
    
    return predictions


def extract_answer(content):
    think_pattern = r"<think>(.*?)</think>"
    
    # 拿掉 think_pattern 的部分剩下的就是 answer 的
    answer_content = re.sub(think_pattern, '', content, flags=re.DOTALL)

    # 加载 json : 
    try:
        json_data = json.loads(answer_content)  # Changed from matches to answer_content
        predictions = json_data.get('diagnoses', [])
        # print('predictions',predictions)
    except:
        predictions = []
    
    return predictions

def verify_result(content, answer_parsed, gold_parsed):

    reward_list = []

    for gold in gold_parsed:
        if gold in answer_parsed:
            reward_list.append(1.0)
        # elif gold in content:
        #     reward_list.append(0.5)
        else:
            reward_list.append(0.0)
    
    return float(np.mean(reward_list))

def verify_result_acc(answer_parsed, gold_parsed):

    # reward_list = []
    # ACC easy: 用两个 list 的交集 除以 gold 的长度
    # ACC hard: 用两个 list 的交集长度 除以 两个 list 的并集长度 --  用这个
    # 再确定下 answer_parsed 的格式
    
    # 规避一些混过来的 格式不对的 case
    try:
        # 输出不能有重复的
        if len(answer_parsed) != len(set(answer_parsed)):
            return 0.0

        reward = len(set(answer_parsed) & set(gold_parsed)) / len(set(answer_parsed) | set(gold_parsed))
    except:
        reward = 0.0
    
    return reward

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):

        gold_parsed = parse_gold_icd(sol)
        # print('gold_parsed:',gold_parsed)
        answer_prediction = extract_answer(content)
        # answer_prediction = extract_answer_easy(content)
        
        # if len(answer_prediction) == 0:
        #     rewards.append(0.0)
        # else:
        reward = verify_result_acc(answer_prediction, gold_parsed)
        rewards.append(reward)

    print("acc rewards:", rewards)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]

    matches = []
    for content in completion_contents:
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        # 按照 提取 diagonses 的严格格式来
        if match is not None:
            answer_pattern = r"<answer>(.*?)</answer>$"
            answer_match = re.search(answer_pattern, content, re.DOTALL | re.MULTILINE)

            answer_content = answer_match.group(1) if answer_match else ""
            # 如果是多个答案的情况，就不要了
            if len(re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL | re.MULTILINE)) > 1:
                match = None 
            
            # 尝试提一下答案，有问题就不要了
            try:
                json_data = json.loads(answer_content)
                predictions = json_data.get('diagnoses', [])

                if not isinstance(predictions,list):
                    match = None       
            except:
                match = None
        
        matches.append(match is not None)

    return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solutions: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    rewards = []
    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    try:
        """Returns a reward function that evaluates code snippets in a sandbox."""
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
        verification_info = kwargs["verification_info"]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
            )
            for code, info in zip(code_snippets, verification_info)
        ]
        with Sandbox(timeout=30, request_timeout=3) as sbx:
            for script in scripts:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    output = float(execution.text)
                except (TypeError, ValueError):
                    output = 0.0
                rewards.append(output)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)
    return rewards
