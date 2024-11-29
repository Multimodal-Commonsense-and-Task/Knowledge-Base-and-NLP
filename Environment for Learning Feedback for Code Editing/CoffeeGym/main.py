import sys
import argparse
import asyncio
import json
import os
from tqdm import tqdm
from copy import deepcopy
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import OpenAIModel, VllmModel
from utils.utils import extract_python_code
from utils.path import save_json_file

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

GT_WRONG_FEEDBACK_PATH = "DATA_PATH_TO_WRONG_FEEDBACK"
GT_CORRECT_FEEDBACK_PATH = "DATA_PATH_TO_CORRECT_FEEDBACK"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/test_cases.json", help="The dataset name")
    parser.add_argument("--get_inference_file", type=str, default=None, help="The file to get the inference results")
    parser.add_argument("--do_generate", action="store_true", help="Whether to generate the initial code")
    parser.add_argument("--only_feedback", action="store_true", help="Whether to only generate feedback")
    parser.add_argument("--debug", action="store_true", help="Turn this on when you want to activate debug mode. It will run over only 10 instances.")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106", help="model name")
    parser.add_argument("--model_url", type=str, help="url to the vllm server")
    parser.add_argument("--model_port", type=int, default=8000, help="The port number of the model")
    parser.add_argument("--reward_model_name", type=str, default="MODEL_NAME", help="sub model name")
    parser.add_argument("--reward_model_url", type=str, default="http://localhost", help="url to the vllm server for reward model")
    parser.add_argument("--reward_model_port", type=int, default=8008, help="The port number of the sub model")
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing")
    ## generate args ##
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--stop_sequence", type=str, nargs='+', default=None)
    parser.add_argument("--sampling_num", type=int, default=1, help="The number of samples to generate per instance")
    parser.add_argument("--feedback_type", type=str, choices=['gt_wrong_feedback', "gt_correct_feedback", "generated"])
    parser.add_argument("--save_test_cases", action="store_true")
    parser.add_argument("--iter_data_size", type=int, default=300, help="Number of data instances to process per iteration")
    parser.add_argument("--iter_start_idx", type=int, default=0, help="The start idx of data instances to process")

    ## Additional ##
    args = parser.parse_args()
    print(args)
    return args



def run_code(code, input_data):
    try:
        # Use the absolute path of the currently running Python interpreter
        python_executable = sys.executable

        # Print diagnostic information about the environment
        # print(f"Using Python executable: {python_executable}")
        # # print(f"Current working directory: {os.getcwd()}")
        # print(f"Environment PATH: {os.getenv('PATH')}")
        # print(f"Environment PYTHONPATH: {os.getenv('PYTHONPATH')}")

        # # Log the input data
        # print(f"Input data: {input_data}")

        process = subprocess.Popen(
            [python_executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )


        input_data = input_data.encode()

        try:
            stdout_data, stderr_data = process.communicate(input=input_data, timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return "Timeout"

        if process.returncode != 0:
            # Print stderr data for debugging
            error_message = stderr_data.decode("utf-8")
            
            return f"Error: {error_message}"

        # Print stdout data for debugging
        # print(f"Subprocess output: {stdout_data.decode('utf-8').strip()}")
        return stdout_data.decode("utf-8").strip()
    except FileNotFoundError as fnf_error:
        # print(f"FileNotFoundError: {fnf_error}")
        return str(fnf_error)
    except Exception as e:
        # print(f"Unexpected error while executing code: {e}\nCode:\n{code}")
        return str(e)

def compare_output(predictions, references):
    """
    Compares the output from the code execution with the expected references.

    Args:
        predictions (list): List of outputs generated by the code.
        references (list): List of expected outputs.

    Returns:
        list: A list of boolean values indicating if the prediction matches the reference.
    """
    predictions = [p.replace("\r\n", "\n").strip("\n") if p is not None else None for p in predictions]
    references = [r.replace("\r\n", "\n").strip("\n") if r is not None else None for r in references]

    results = [p == r for p, r in zip(predictions, references)]
    return results

async def async_inference(model, prompt_template, item, args, idx, mode="feedback"):
    """
    mode: feedback and refine -> refine is for the reward model
    """
    if mode == "feedback":
        input_prompt = model.apply_function_to_prompt(prompt_template, mode, **item)
        response = await model.async_inference(input_prompt, temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p, frequency_penalty=args.frequency_penalty, stop=args.stop_sequence, n=args.sampling_num)
        return idx, {
            "feedback_input_prompt": input_prompt,
            "raw_response": response,
            "feedback": response
        }
    else:
        feedbacks = item['feedbacks'] if 'feedbacks' in item else [item['gold_feedback']]
        refined_codes = []

        async def infer(feedback):
            input_prompt = model.apply_function_to_prompt(prompt_template, mode, **{**item, "feedback": feedback})
            response = await model.async_inference(input_prompt, temperature=0.2, max_tokens=args.max_tokens, top_p=0.95, frequency_penalty=args.frequency_penalty, stop=args.stop_sequence, n=1)
            extracted_code = extract_python_code(response[0]) ## only one response
            return input_prompt, response, extracted_code

        results = await asyncio.gather(*[infer(feedback) for feedback in feedbacks])

        for input_prompt, response, extracted_code in results:
            refined_codes.append(extracted_code)

        return idx, {
            "input_prompt": results[0][0],  # All input_prompts are the same, so take the first one
            "raw_response": [result[1] for result in results],
            "refined_codes": refined_codes
        }


async def process_items_concurrently(main_model, prompt_template, items, args, mode="feedback"):
    tasks = [async_inference(main_model, prompt_template, item, args, idx, mode=mode) for idx, item in enumerate(items)]
    results = [None] * len(items)
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        idx, result = await future
        results[idx] = result
    
    return results

def run_code_for_feedback(refined_code, cur_testcase_inputs, cur_testcase_outputs, d):
    cur_outputs = [run_code(refined_code, input_data) for input_data in cur_testcase_inputs]
    cur_testcase_pass_results = compare_output(cur_outputs, cur_testcase_outputs)
    d.setdefault("testcase_outputs", []).append(cur_outputs)
    d.setdefault("testcase_pass_results", []).append(cur_testcase_pass_results)
    score = sum(cur_testcase_pass_results) / len(cur_testcase_pass_results) if cur_testcase_pass_results else 0
    return score

async def main(args):
    ## Load the data ##
    with open(args.dataset, "r") as f:
        data = json.load(f)
    
    if args.get_inference_file is not None:
        with open(args.get_inference_file, "r") as f:
            data = json.load(f)
            
    if args.debug:
        data = data[:10]
    
    ## Load main model ##
    if args.do_generate:
        if "gpt" in args.model_name:
            main_model = OpenAIModel(model_name=args.model_name)
        else:
            main_model = VllmModel(model_name=args.model_name, port=args.model_port, url=args.model_url)

    if not args.only_feedback:
        if "gpt" in args.reward_model_name:
            reward_model = OpenAIModel(model_name=args.reward_model_name)
        else:
            reward_model = VllmModel(model_name=args.reward_model_name, port=args.reward_model_port, url=args.reward_model_url)

    all_data = deepcopy(data)
    overall_avg_scores = []

    for start_idx in range(args.iter_start_idx, len(data), args.iter_data_size):
        end_idx = min(start_idx + args.iter_data_size, len(data))
        stream_data = all_data[start_idx:end_idx]

        if args.do_generate: # generate feedback using the feedback model (main_model)
            prompt_template = main_model.get_prompt_template(mode = "feedback") ## TODO: change ##
            feedback_results = await process_items_concurrently(main_model, prompt_template, stream_data, args, "feedback")
            for idx, item in enumerate(stream_data):
                data[start_idx + idx]['feedback_input_prompt'] = feedback_results[idx]['feedback_input_prompt']
                data[start_idx + idx]['feedbacks'] = feedback_results[idx]['feedback']
        elif args.get_inference_file is not None:
            pass
        else: # only evaluate the feedback annotated in our Coffee dataset
            for idx, item in enumerate(stream_data):
                data[start_idx + idx]['feedbacks'] = [item['gold_feedback']]
             
        if args.only_feedback:
            fragment_save_path = os.path.join(args.save_dir, "fragments")
            os.makedirs(fragment_save_path, exist_ok=True)
            with open(os.path.join(fragment_save_path, f"{args.feedback_type}_feedback_results_sampling{args.sampling_num}_part{start_idx}.json"), "w") as f:
                json.dump(data[start_idx:end_idx], f, indent=4)
            continue
                
        else:
            stream_data = deepcopy(data[start_idx:end_idx])
            refine_prompt_template = reward_model.get_prompt_template(mode="refine")
            code_results = await process_items_concurrently(reward_model, refine_prompt_template, stream_data, args, "refine")
            
            for idx, item in enumerate(stream_data):
                data[start_idx + idx]['refine_input_prompt'] = code_results[idx]['input_prompt']
                data[start_idx + idx]['raw_outputs'] = code_results[idx]['raw_response']
                data[start_idx + idx]['refined_codes'] = code_results[idx]['refined_codes']
                data[start_idx + idx]['refined_codes_formatted'] = [cs.split("\n") for cs in code_results[idx]['refined_codes']]
            
            avg_scores = []
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                for di, d in enumerate(tqdm(data[start_idx:end_idx])):
                    cur_problem_id = d["problem_id"]
                    cur_testcase = d['test_cases']
                    cur_testcase_inputs = [t[0] for t in cur_testcase]
                    cur_testcase_outputs = [t[1] for t in cur_testcase]
                    cur_scores = []
                    feedback_score_pairs = []
                    if not args.save_test_cases:
                        data[start_idx + di].pop("test_cases", None)

                    for feedback, refined_code in zip(d['feedbacks'], d['refined_codes']):
                        cur_outputs = list(executor.map(run_code, [refined_code] * len(cur_testcase_inputs), cur_testcase_inputs))
                        cur_testcase_pass_results = compare_output(cur_outputs, cur_testcase_outputs)
                        d.setdefault("testcase_outputs", []).append(cur_outputs)
                        d.setdefault("testcase_pass_results", []).append(cur_testcase_pass_results)
                        score = sum(cur_testcase_pass_results) / len(cur_testcase_pass_results) if cur_testcase_pass_results else 0
                        cur_scores.append(score)
                        feedback_score_pairs.append((feedback, score))
                    
                    avg_score = sum(cur_scores) / len(cur_scores) if cur_scores else 0
                    d['feedback_score_pairs'] = feedback_score_pairs
                    avg_scores.append(avg_score)
                    if not args.save_test_cases:
                        d.pop("testcase_outputs", None)
                        d.pop("testcase_pass_results", None)
                    print(f"Average score for problem {cur_problem_id}: {avg_score}")

            
            overall_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0
            overall_avg_scores.append(overall_avg_score)
            print(f"Overall average score for chunk {start_idx}-{end_idx}: {overall_avg_score}")
            if not args.save_test_cases:
                        d.pop("testcase_outputs", None)
                        d.pop("testcase_pass_results", None)
            fragment_save_path = os.path.join(args.save_dir, "fragments")
            os.makedirs(fragment_save_path, exist_ok=True)
            with open(os.path.join(fragment_save_path, f"{args.feedback_type}_scoring_results_sampling{args.sampling_num}_part{start_idx}.json"), "w") as f:
                json.dump(data[start_idx:end_idx], f, indent=4)

    overall_avg_score = sum(overall_avg_scores) / len(overall_avg_scores) if overall_avg_scores else 0
    print(f"Overall average score: {overall_avg_score}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.only_feedback:
        save_name = os.path.join(args.save_dir, f"{args.feedback_type}_feedback_results_sampling{args.sampling_num}.json")
    else: 
        save_name = os.path.join(args.save_dir, f"{args.feedback_type}_scoring_results_sampling{args.sampling_num}.json")
        
    with open(save_name, "w") as f:
        json.dump(data, f, indent=4)
    return overall_avg_score

def calculate_precision(TP, FP):
    precision = TP / (TP + FP)
    return precision

def calculate_recall(TP):
    recall = TP / (TP + 100 - TP)
    return recall

def calculate_f1_score(precision, recall):
    F1_score = 2 * (precision * recall) / (precision + recall)
    return F1_score


if __name__ == "__main__":
    args = parse_args()
    if not args.do_generate and args.get_inference_file is None:
        args.dataset = GT_WRONG_FEEDBACK_PATH
        print(f"running wrong feedback evaluation on {args.dataset}")
        args.feedback_type = "gt_wrong_feedback"
        false_positive = asyncio.run(main(args))

        
        args.dataset = GT_CORRECT_FEEDBACK_PATH
        print(f"running correct feedback evaluation on {args.dataset}")
        args.feedback_type = "gt_correct_feedback"
        true_positive = asyncio.run(main(args))

        precision = round(calculate_precision(true_positive, false_positive), 2)
        recall = round(calculate_recall(true_positive), 2)
        f1 = round(calculate_f1_score(precision, recall), 2)
        score_list = [str(score) for score in [true_positive, false_positive, precision, recall, f1]]
        with open(os.path.join(args.save_dir, "metrics.csv"), "w") as f:
            f.write(",".join(score_list))

    else:
        pass_ratio = asyncio.run(main(args))
