import io
import random
import copy
from collections import defaultdict
import logging
import datasets
import re
from ..components.utils import retrieve_answer_from_last_step, eval_output

logger = logging.getLogger(__name__)

def retrieve_answer_from_gsm8k(example: dict) -> str:
    return re.match(r'[\S\s]*#### (.*)$', example['answer'])[1]

def load_qa_dataset(dataset_name):
    """ All datasets are loaded from huggingface hub """
    if dataset_name == "gsm8k":
        full_dataset = list(datasets.load_dataset('gsm8k', 'main', split='test'))
        for example in full_dataset:
            example["answer"] = retrieve_answer_from_gsm8k(example)
    elif dataset_name == "math500":
        full_dataset = list(datasets.load_dataset("xinzhel/math500-float", split='test'))
    elif dataset_name == "spart_yn":
        full_dataset = list(datasets.load_dataset("xinzhel/spart_yn", split='test'))
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return full_dataset
          
def get_accuracy(full_dataset, results_from_file, extract_method="dfs", verbose=False):
    correct_count = 0
    incorrect_indices = []
    for example_idx in range(len(full_dataset)):
        if verbose:
            print("\nexample_idx: ", example_idx)
        # ground truth
        example = full_dataset[example_idx]
        answer = example['answer']
        
        # model output
        try:
            final_trace_plus_all = results_from_file.results[example_idx] 
        except IndexError:
            print(f"results_from_file.results[{example_idx}] -> IndexError")
            break
        
        # extract answer from model output
        if extract_method == "dfs":
            output = extract_answer_from_dfs_path(final_trace_plus_all[0], verbose=verbose)
        elif extract_method == "aggregation":
            output = extract_answer_from_aggregation(final_trace_plus_all[1:], verbose=verbose)
        else:
            raise ValueError(f"Unknown extract_method: {extract_method}")

        # caculate accuracy
        correct = eval_output(answer, output)
                
        if verbose:
            print("answer: ", answer, "; output: ", output, "; correct: ", correct)
        if not correct:
            incorrect_indices.append(example_idx)
            
        correct_count += correct
        example_idx += 1

    accuracy = correct_count / len(full_dataset)
    return {"accuracy": accuracy, "correct_count": correct_count, "total_examples": len(full_dataset), "incorrect_indices": incorrect_indices}


# ------------------- Answer Extraction ------------------
def extract_answer_from_aggregation( paths, use_reward=False, weight_policy: str = 'edge', verbose=False):
    assert weight_policy in ['edge']

    # extraction
    final_steps_for_answer_extracion = [path[-1].state[-1]  for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)] # path[-1].state is not None and len(path[-1].state) > 0: expanded node that has not been simulated
    if use_reward:
        rewards = [path[-1].reward for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)]
    else:
        rewards = [path[-1].fast_reward for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)]
    depths = [path[-1].depth for path in paths if  path and (path[-1].state is not None and len(path[-1].state) > 0)]
    assert len(final_steps_for_answer_extracion) == len(rewards) == len(depths)

    # aggregation
    answer_dict = defaultdict(lambda: 0)
    num_terminal = 0
    for final_step, reward, depth in zip(final_steps_for_answer_extracion, rewards, depths):
        if verbose:
            print("Final step:", final_step)
        answer = retrieve_answer_from_last_step(final_step) 
        if answer == "":
            continue
        else:
            num_terminal += 1
        if weight_policy == 'edge':
            answer_dict[answer] += reward
        elif weight_policy == 'edge_inverse_depth':
            answer_dict[answer] += reward / depth
    
    if num_terminal > 1:
        if verbose:
            print(f"Number of terminal nodes with answers: {num_terminal}")
            print("Answer dict:", answer_dict)

    if len(answer_dict) == 0:
        return ""
    return max(answer_dict, key=lambda answer: answer_dict[answer])

def extract_answer_from_dfs_path(final_trace, verbose=False):
    
    terminal_state = final_trace[-1].state if len(final_trace) > 0 else None
    final_step = terminal_state[-1]
    try:
        if verbose:
            print("Final step:", final_step)
        output = retrieve_answer_from_last_step(final_step)
    except IndexError:
        if verbose:
            print("IndexError in retrieve_answer_from_last_step. terminal_state: ", terminal_state)
        output = ""
    return output