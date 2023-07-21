import json
import logging
import pathlib
from typing import List
import numpy as np

import yaml
from lang import Interpreter, list_manip_dsl_gen

import trlx
from trlx.data.configs import TRLConfig

logger = logging.getLogger(__name__)

all_func_names = list(list_manip_dsl_gen.keys())

class DSLDataset:
    def __init__(self):
        with open("dataset/train.json", "r") as f:
            self.train_data = json.load(f)
        with open("dataset/test.json", "r") as f:
            self.test_data = json.load(f)
        logger.info("Sucessfully loaded the dataset")

    def load_datapoints(self, split="train"):
        if split == "train":
            for datapoint in self.train_data:
                if "ERROR" not in datapoint["input"]:
                    # yield datapoint["input"]
                    # yield datapoint
                    # rename input, output to prompt, original_output
                    yield {"prompt": datapoint["input"], "original_output": datapoint["output"]}
        elif split == "test":
            for datapoint in self.test_data:
                # yield datapoint["input"]
                # yield datapoint
                # rename input, output to prompt, original_output
                yield {"prompt": datapoint["input"], "original_output": datapoint["output"]}

interpreter = Interpreter()

EOS_TOKEN = "<|endoftext|>"

def reward_fn(samples, **kwargs):
    reward_list = []
    for sample in samples:
        code = sample.split("Function:")[1].strip()

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            reward_list.append(-1)
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                reward_list.append(1)
            else:
                # if the output is incorrect, we give it a negative reward.
                reward_list.append(-0.5)

    return reward_list


def dense_reward_fn1(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:

    reward_list: List[float] = []
    for sample in samples:
        code = sample.split("Function:")[1].strip()

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            reward_list.append(-1)
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                reward_list.append(1)
            else:
                # if the output is incorrect, we give it a negative reward.
                reward_list.append(-0.5)

    tok_scores: List[List[float]] = []
    for sample, prompt, response, text_score in zip(samples, prompts, outputs, reward_list):
        toks = tokenizer(response).input_ids
        tok_score = [0.0] * len(toks)
        tok_score[-1] = text_score
        tok_scores.append(tok_score)
    return tok_scores


def count_used_func_names(code: str) -> dict:
    # count how many times each function name is used in the code
    # return a dict with function names as keys and counts as values
    used_func_names = {}
    for func_name in all_func_names:
        used_func_names[func_name] = code.count(func_name + "(")
    return used_func_names

def total_func_count(used_func_names: dict):
    return sum(used_func_names.values())

def diff_func_counts(used_a: dict, used_b: dict):
    # return the difference in counts of function names used
    diff = 0
    for func_name in all_func_names:
        diff += abs(used_a[func_name] - used_b[func_name])
    return diff

def missing_func_count(used_a: dict, used_b: dict):
    # return the difference in counts of function names used
    diff = 0
    for func_name in all_func_names:
        diff += max(0, used_a[func_name] - used_b[func_name])
    return diff

def reward_func_usage(expected: str, generated: str) -> float:
    # maximum value is 1
    # minimum value is 0 when count of different function names used is >= # of function names in expected
    # return 1 - (count of different function names used / # of function names in expected)
    expected_used = count_used_func_names(expected)
    generated_used = count_used_func_names(generated)
    diff = diff_func_counts(expected_used, generated_used) / 2
    max_diff = total_func_count(expected_used)
    return 1 - (diff / max_diff)

def reward_output_length(expected_length, observed_length, max_possible_length=10):
    length_diff = abs(expected_length - observed_length)
    steepness = 2
    reward = np.exp(-steepness * length_diff / max_possible_length)
    # norm_reward = min(1.0, max(-1, reward))
    # final_reward = norm_reward * 2 - 1
    scaled_reward = reward * 2 - 1
    # final_reward = max(-1, min(1, scaled_reward))
    final_reward = clamp_reward(scaled_reward)
    return final_reward

# def reward_func_usage_nums(perc_used: float, perc_missing: float) -> float:
#     steepness = 1
#     used_reward = np.exp(steepness*(perc_used - 1))
#     missing_reward = np.exp(-steepness*(perc_missing))
#     reward = 2*(used_reward * missing_reward)-1
#     # return max(-1, min(1, reward))
#     return clamp_reward(reward)

def reward_func_usage_nums(perc_used: float, perc_missing: float) -> float:
    used_steepness = 1
    used_reward = np.exp(used_steepness*(perc_used - 1))
    missing_steepness = 1
    missing_reward = np.exp(-missing_steepness*(perc_missing))
    sub_reward = used_reward * missing_reward
    reward = 2*sub_reward-1
    return clamp_reward(reward)

def reward_func_usage_nums_text(expected: str, generated: str) -> float:
    # maximum value is 1
    # minimum value is 0 when count of different function names used is >= # of function names in expected
    # return 1 - (count of different function names used / # of function names in expected)
    expected_used = count_used_func_names(expected)
    generated_used = count_used_func_names(generated)
    diff = missing_func_count(expected_used, generated_used)
    max_diff = total_func_count(expected_used)
    perc_missing = diff / max_diff
    perc_used = 1 - (diff / max_diff)
    # print("perc_used:", perc_used)
    # print("perc_missing:", perc_missing)
    # return reward_func_usage_nums(perc_used, perc_missing)
    return reward_func_usage_nums(perc_used, 0.1 if perc_missing > 0 else 0)

def pairwise_diff(expected_arr, observed_arr):
    diff = 0
    for i in range(len(expected_arr)):
        # if expected_arr[i] not in observed_arr:
        if i >= len(observed_arr) or expected_arr[i] != observed_arr[i]:
            diff += 1
    return diff

def reward_pairwise_diff(expected_arr, observed_arr):
    diff = pairwise_diff(expected_arr, observed_arr)
    # return 1 - (diff / len(expected_arr))
    return 2*(1 - (diff / len(expected_arr)))-1

def clamp_reward(reward: float, min_reward: float = -1.0, max_reward: float = 1.0) -> float:
    return max(min_reward, min(max_reward, reward))

# def scale_reward(reward: float, min_before: float, max_before: float, min_after: float, max_after: float) -> float:
# numpy to scale from [-1, +1] to [-0.5, +0.7]

def dense_reward_fn2(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:

    reward_list: List[float] = []
    # for sample in samples:
    for sample_index, sample in enumerate(samples):
        code = sample.split("Function:")[1].strip()

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            reward_list.append(-1)
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                reward_list.append(1)
            else:
                # if the output is incorrect, we give it a negative reward.
                # reward_list.append(-0.5)
                expected_code = kwargs['original_output'][sample_index]
                generated_code = code
                partial_reward = reward_func_usage(expected_code, generated_code)
                # scale partial_reward from [-1, +1] to [-0.5, +0.7]
                partial_reward = max(-0.5, (partial_reward * 1.2) - 0.5)
                reward_list.append(partial_reward)

    tok_scores: List[List[float]] = []
    for sample, prompt, response, text_score in zip(samples, prompts, outputs, reward_list):
        toks = tokenizer(response).input_ids
        tok_score = [0.0] * len(toks)
        tok_score[-1] = text_score
        tok_scores.append(tok_score)
    return tok_scores


def calculate_weighted_reward(rewards, weights):
    # Ensure the lengths of rewards and weights are the same
    if len(rewards) != len(weights):
        raise ValueError("Length of rewards and weights must be equal")
    
    # Ensure the sum of weights equals 1
    if not 0.99 <= sum(weights) <= 1.01:
        raise ValueError("Sum of weights must be approximately 1")
    
    # Calculate the weighted reward
    weighted_reward = sum(reward * weight for reward, weight in zip(rewards, weights))
    
    return weighted_reward

def reward_matching_brackets_text(code: str) -> float:
    opens = code.count("(")
    closes = code.count(")")
    diff = abs(opens - closes)
    total_brackets = opens + closes
    # perc_diff = diff / max_possible_length
    perc_diff = diff / total_brackets
    perc_match = 1 - perc_diff
    return np.interp(perc_match,
        (0, 1), 
        (-1, 1))

def dense_reward_fn3(samples: List[str], prompts: List[str], outputs: List[str], tokenizer, **kwargs) -> List[List[float]]:

    reward_list: List[float] = []
    # for sample in samples:
    for sample_index, sample in enumerate(samples):
        code = sample.split("Function:")[1].strip()

        if code.endswith(EOS_TOKEN):
            code = code[:-len(EOS_TOKEN)]

        output = eval(sample.split("Output:")[1].strip().split("Function:")[0].strip())
        interpreted_output = interpreter(code)
        if interpreted_output == "ERROR":
            # If the code is unparsable, we give it a negative reward.
            reward_list.append(-1)
        else:
            # if the code is parseable
            if output == interpreted_output:
                # if the output is correct, we give it a positive reward.
                reward_list.append(1)
            else:
                # if the output is incorrect, we give it a negative reward.
                # reward_list.append(-0.5)
                expected_code = kwargs['original_output'][sample_index]
                generated_code = code

                # reward for output length
                expected_length = len(output)
                observed_length = len(interpreted_output)
                # max_possible_length = 200 # completion tokens
                max_possible_length = max(expected_length, observed_length) # items in output array

                length_reward = reward_output_length(expected_length, observed_length, max_possible_length=max_possible_length)
                funcs_reward = reward_func_usage_nums_text(expected_code, generated_code)
                pairwise_reward = reward_pairwise_diff(output, interpreted_output)
                matching_brackets_reward = reward_matching_brackets_text(generated_code)

                rewards = [
                    funcs_reward,
                    length_reward,
                    pairwise_reward,
                    matching_brackets_reward,
                ]
                # weights = [0.34, 0.33, 0.33]
                # weights = [0.45, 0.2, 0.35]
                weights = [0.4, 0.1, 0.3, 0.2]
                weighted_reward = calculate_weighted_reward(rewards, weights)

                # partial_reward = max(-0.5, (partial_reward * 1.2) - 0.3)
                # partial_reward = max(-0.5, (weighted_reward * 1.2) - 0.3)
                # partial_reward = max(-0.1, (weighted_reward * 1.2) - 0.3)

                # scale weighted_reward from [-1, +1] to [-0.1, +0.9] without clipping
                # partial_reward = (weighted_reward * 1.0) - 0.1
                partial_reward = np.interp(weighted_reward, 
                            (-1, 1), 
                            (-0.1, 0.9))

                reward_list.append(partial_reward)

    tok_scores: List[List[float]] = []
    for sample, prompt, response, text_score in zip(samples, prompts, outputs, reward_list):
        toks = tokenizer(response).input_ids
        tok_score = [0.0] * len(toks)
        tok_score[-1] = text_score
        tok_scores.append(tok_score)
    return tok_scores

config_path = pathlib.Path(__file__).parent.joinpath("configs/trlx_ppo_config.yml")
with config_path.open() as f:
    default_config = yaml.safe_load(f)


def main(hparams={}):
    use_dense = True
    # if use_dense then append "_dense" to default_config.train.project_name in a new dict
    final_config = default_config
    if use_dense:
        final_config = default_config.copy()
        final_config["train"]["project_name"] = final_config["train"]["project_name"] + "_dense"
    config = TRLConfig.update(final_config, hparams)

    # Dataset
    dataset = DSLDataset()
    train_prompts = list(dataset.load_datapoints(split="train"))[:1000]
    test_prompts = list(dataset.load_datapoints(split="test"))[:20]

    trainer = trlx.train(
        # reward_fn=reward_fn,
        # if use_dense then use dense_reward_fn
        reward_fn=dense_reward_fn3 if use_dense else reward_fn,
        prompts=train_prompts,
        eval_prompts=test_prompts,
        config=config,
    )
    trainer.save_pretrained("dataset/trained_model")


if __name__ == "__main__":
    # TEST REWARD FUNTION
    assert (reward_fn(["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -4]),1)"])) == [1]
    assert (reward_fn(["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -a]),1)"])) == [-1]
    assert (reward_fn(["Input: 1 Output: [-4,-5,-2] Function: div_n(reverse([-2, -5, -3]),1)"])) == [-0.5]

    main()
