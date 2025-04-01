import math
import re
import os

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import random


# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is the same as the ground truth."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
    
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         match = re.search(r'<answer>(.*?)</answer>', content)
#         if match:
#             answer_content = match.group(1).strip()  # 提取并去除前后空格
#             if answer_content == sol:
#                 reward = 1.0
#                 os.makedirs("completion_samples", exist_ok=True)
#                 log_file = os.path.join("completion_samples", "success_completion_samples.txt")
#                 with open(log_file, "a") as f:
#                     f.write(f"\n\n==============\n")
#                     f.write(content)
#                     f.write(sol)
#         rewards.append(reward)
#     return rewards

# def validate_response_structure(processed_str: str) -> bool:
#     """Performs comprehensive validation of response structure.
    
#     Args:
#         processed_str: Processed response string from the model
        
#     Returns:
#         Boolean indicating whether all formatting requirements are met
#     """
#     #print("\n[Structure Validation]")
#     validation_passed = True

#     # Check required tags
#     tags = {
#         'think_start': ('<think>', 1),
#         'think_end': ('</think>', 1),
#         'answer_start': ('<answer>', 1),
#         'answer_end': ('</answer>', 1)
#     }

#     positions = {}
#     for tag_name, (tag_str, expected_count) in tags.items():
#         count = processed_str.count(tag_str)
#         positions[tag_name] = pos = processed_str.find(tag_str)
        
#         #print(f"  {tag_str}: count={count}, position={pos}")
        
#         if count != expected_count:
#             #print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
#             validation_passed = False

#     # Verify tag order
#     if (positions['think_start'] > positions['think_end'] or
#         positions['think_end'] > positions['answer_start'] or
#         positions['answer_start'] > positions['answer_end']):
#         #print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
#         validation_passed = False
#     else:
#         pass
#         #print("  Tag sequence validation passed")

#     return validation_passed

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""

#     # pattern = r"^<think>.*?</think><answer>.*?</answer>$"
#     # completion_contents = [completion[0]["content"] for completion in completions]
#     # matches = [re.match(pattern, content) for content in completion_contents]

#     # https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py#L170C5-L171C76
#     rewards = [] 
#     completion_contents = [completion[0]["content"] for completion in completions]
#     for content in completion_contents:
#         reward = -1.0
#         format_correct = validate_response_structure(content)
#         if format_correct:
#             reward = 1.0
#             os.makedirs("completion_samples", exist_ok=True)
#             log_file = os.path.join("completion_samples", "completion_samples.txt")
#             with open(log_file, "a") as f:
#                 f.write(f"\n\n==============\n")
#                 f.write(content)
                
#         rewards.append(reward)
#         # print("="*50)
#         # print(content)
#         # print("&"*10)
#         # print(reward)
#         # #print(answer_content, gold, reward)
#         # print("%"*50)
#     return rewards

def accuracy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        match = re.search(r'<answer>(.*?)</answer>', content)
        reward = 0.0
        if match:
            answer_content = match.group(1).strip()  # 提取并去除前后空格
            if sol == "yes" or sol == "no":
                answer = answer_content.split(" ")
                for i in answer:
                    if i.lower() == sol:
                        reward = 1.0
                        break
            else:
                if len(answer_content.split(" ")) > 5:
                    reward = 0.0
                else:
                    answer_content = answer_content.replace("million", "").replace("billion", "").replace(",", "")
                    try:
                        
                        if answer_content == sol:
                            reward = 1.0
                        elif "%" in answer_content:
                            answer = float(answer_content.replace("%", ""))
                            answer = answer / 100
                            if abs(float(sol) - answer) < 0.001:
                                reward = 1.0
                        elif "$" in answer_content:
                            answer = answer_content.replace("$", "")
                            if abs(float(answer) - float(sol)) < 0.001:
                                reward = 1.0
                        elif abs(float(answer_content) - float(sol)) < 0.001:
                            reward = 1.0
                    except:
                        reward = 0.0
            
            if reward == 0:
                if random.random() < 0.10:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "completion_samples_error.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(content)
                        f.write(f"gt:{sol}")
            else:
                if random.random() < 0.10:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "completion_samples_correct.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(content)
                        f.write(f"gt:{sol}")
                
            
        rewards.append(reward)
    return rewards
    

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward(completions, **kwargs):
    """Reward function that checks for clear step-by-step reasoning.
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