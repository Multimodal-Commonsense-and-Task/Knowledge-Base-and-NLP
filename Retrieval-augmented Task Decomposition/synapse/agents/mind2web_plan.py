import logging
from lxml import etree
import numpy as np
import json
import os
import ipdb
import copy
import random
from pathlib import Path
import time
import re

from .planner_agent import Agent

from io import StringIO
import pprint
pprinter = pprint.PrettyPrinter(indent=4)
#ppp = pprinter.pprint

# Function to pprint without a trailing newline using 'ppp' as the command
def ppp(obj):
    # Use StringIO to capture the output of pprint
    output = StringIO()
    pprinter.pprint(obj, stream=output)
    
    # Get the string from the StringIO object
    string = output.getvalue()
    
    # Remove the last character (newline) if it exists and print
    if string.endswith('\n'):
        string = string[:-1]
    print(string, end='')


from synapse.envs.mind2web.env_utils import (
    get_target_obs_and_act,
    get_target_obs,
    calculate_f1,
    parse_act_str,
    construct_act_str,
)
from synapse.utils.llm import generate_response, num_tokens_from_messages, MAX_TOKENS, extract_from_response, set_api
from synapse.memory.mind2web.build_memory import (
    load_memory,
    retrieve_exemplar_name,
    get_specifiers_from_sample,
    get_top_k_obs,
)

from synapse.memory.mind2web.annotate_memory import (
    insert_thoughts_into_trajectory,
    annotate_exemplar,
    load_annotated_memories,
    annotate_and_insert_thoughts_into_trajectory,
    Prompts
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()    
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))        
logger.addHandler(handler)
# from gpt import GPT

def eval_synapse_new(task_id, args, sample):
    set_api(args.api)
    # initialize metrics
    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    if args.no_trajectory:
        assert args.no_memory
        exemplars = [
            [
                {
                    "role": "user",
                    "content": "Observation:\n```\n<html> <div> <div> <a tock home page /> <button id=0 book a reservation. toggle open> <span> Book a reservation </span> </button> <button book a reservation. toggle open> </button> </div> <div> <select id=1 type> <option reservations true> Dine in </option> <option pickup> Pickup </option> <option delivery> Delivery </option> <option events> Events </option> <option wineries> Wineries </option> <option all> Everything </option> </select> <div id=2> <p> Celebrating and supporting leading women shaking up the industry. </p> <span> Explore now </span> </div> </div> </div> </html>\n```\nTask: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest\nPrevious actions:\nNone\nNext action:",
                },
                {"role": "assistant", "content": "`SELECT [1] [Pickup]`"},
            ],
            [
                {
                    "role": "user",
                    "content": "Observation:\n```\n<html> <div> <nav main menu> <ul> <li> <div button> Car Sales </div> <div id=0> <div> <div> <div> Buy A Car </div> <div> Plan Your Purchase </div> </div> <div> <h4> Its Tax Refund Time. Treat Yourself to an Upgrade. </h4> <p> With a variety of options, invest your refund in what you really want - a quality, used vehicle from Enterprise. </p> <a> View Inventory </a> </div> </div> </div> </li> <div id=1> Enterprise Fleet Management </div> </ul> </nav> <div region> <button id=2 selected pick-up date 03/19/2023> <span> <span> 19 </span> <div> <span> Mar </span> <span> 2023 </span> </div> </span> </button> </div> </div> </html>\n```\nTask: Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.\nPrevious actions:\n[searchbox]  Pick-up & Return Location (ZIP, City or Airport) (... -> TYPE: Brooklyn\n[option]  Brooklyn, NY, US Select -> CLICK\nNext action:",
                },
                {"role": "assistant", "content": "`CLICK [2]`"},
            ],
            [
                {
                    "role": "user",
                    "content": "Observation:\n```\n<html> <form search> <input id=6385 search q blazer search by keyword /> <button submit search> </button> <button button close> </button> </form> </html>\n```\nTask: Find a black blazer for men with L size and add to wishlist.\nPrevious actions:\n[svg]   -> CLICK\nNext action:",
                },
                {"role": "assistant", "content": "`TYPE [6385] [blazer]`"},
            ],
        ]
    else:
        memory = load_memory(args.memory_path)
        with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
            memory_mapping = json.load(f)
        if not args.no_memory:
            specifier = get_specifiers_from_sample(sample)
            retrieved_exemplar_names, scores = retrieve_exemplar_name(
                memory, specifier, args.retrieve_top_k
            )
            exemplars = [memory_mapping[name] for name in retrieved_exemplar_names]
        else:
            seed = 0
            random.seed(seed)
            exemplars = random.sample(memory_mapping, args.retrieve_top_k)

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]

    prev_actions = []
    prev_obs = []
    previous_k = 9999

    set_api(args.api)

    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]

        if args.no_trajectory:
            # Continue next loop if the ground truth element is not in the cleaned html
            if len(pos_candidates) == 0:
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                prev_actions.append(act_repr)
                conversation.append("The ground truth element is not in cleaned html")
                continue

            obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
            query = f"Observation:\n```\n{obs}\n```\nTask: {sample['confirmed_task']}\nPrevious actions:\n"
            if len(prev_actions) > 0:
                for a in prev_actions[-previous_k:]:
                    query += f"{a}\n"
            else:
                query += "None\n"
            query += "Next action:"
            query = [{"role": "user", "content": query}]
            prev_actions.append(act_repr)
        else:
            target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)
            # Continue next loop if the ground truth element is not in the cleaned html
            if len(pos_candidates) == 0:
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                prev_obs.append("Observation: `" + target_obs + "`")
                prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
                conversation.append("The ground truth element is not in cleaned html")
                continue

            # create the zipped list of previous observations and actions
            zipped_list = list(zip(prev_obs, prev_actions))
            # truncate the list to the last k elements
            if len(zipped_list) > previous_k:
                zipped_list = zipped_list[-previous_k:]

            query = []
            #for o, a in zip(prev_obs, prev_actions):
            for o, a in zipped_list:
                if len(query) == 0:
                    query.append(
                        {
                            "role": "user",
                            "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                            + o,
                        }
                    )
                else:
                    query.append({"role": "user", "content": o})
                query.append({"role": "assistant", "content": a})
            obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
            if len(query) == 0:
                query.append(
                    {
                        "role": "user",
                        "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                        + "Observation: `"
                        + obs
                        + "`",
                    }
                )
            else:
                query.append({"role": "user", "content": "Observation: `" + obs + "`"})
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
        
        total_num_tokens = num_tokens_from_messages(sys_message + query, args.model)
        if total_num_tokens > MAX_TOKENS[args.model]:
            logger.info(
                f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[args.model]}), skipping..."
            )
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            conversation.append(
                {
                    "input": sys_message + query,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            continue
        

        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = num_tokens_from_messages(
                sys_message + demo_message + e + query, args.model
            )
            if total_num_tokens > MAX_TOKENS[args.model]:
                logger.info(
                    f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                )
                break
            else:
                demo_message.extend(e)

        #ipdb.set_trace()
        message = sys_message + demo_message + query
        response, info = generate_response(
            messages=message,
            model=args.model,
            temperature=args.temperature,
            stop_tokens=["Task:", "obs:"],
        )
        
        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    if args.no_trajectory:
        log_dir = Path(f"{args.log_dir}/{args.model}/{args.benchmark}/no_mem_no_traj")
    else:
        log_dir = Path(
            f"{args.log_dir}/{args.model}/{args.benchmark}{'/no_mem' if args.no_memory else ''}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)


        

def eval_synapse_orig(task_id, args, sample):
    #ipdb.set_trace()

    memory = load_memory(args.memory_path)
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        memory_mapping = json.load(f)

    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    if not args.no_memory:
        specifier = get_specifiers_from_sample(sample)
        retrieved_exemplar_names, scores = retrieve_exemplar_name(
            memory, specifier, args.retrieve_top_k
        )
        exemplars = [memory_mapping[name] for name in retrieved_exemplar_names]
    else:
        seed = 0
        random.seed(seed)
        exemplars = random.sample(memory_mapping, args.retrieve_top_k)

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]
    prev_actions = []
    prev_obs = []
    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        # target_obs, target_act = get_target_obs_and_act(s)
        _, target_act = get_target_obs_and_act(s)
        target_obs, _ = get_top_k_obs(s, args.top_k_elements)

        # stop if the ground truth element is not in the top-k candidates
        pos_candidates = s["pos_candidates"]
        pos_candidates = [c for c in pos_candidates if c["rank"] < args.top_k_elements]
        pos_ids = [c["backend_node_id"] for c in pos_candidates]

        if len(pos_ids) == 0:
            #ipdb.set_trace()
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            continue

        # get obs by pruning the tree with top-k candidates
        neg_candidates = s["neg_candidates"]
        neg_candidates = [c for c in neg_candidates if c["rank"] < args.top_k_elements]
        neg_ids = [c["backend_node_id"] for c in neg_candidates]
        all_candidates = pos_ids + neg_ids
        obs = get_target_obs(etree.fromstring(s["cleaned_html"]), all_candidates)

        # Generate action with OpenAI api
        query = []
        for o, a in zip(prev_obs, prev_actions):
            if len(query) == 0:
                query.append(
                    {
                        "role": "user",
                        "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                        + o,
                    }
                )
            else:
                query.append({"role": "user", "content": o})
            query.append({"role": "assistant", "content": a})
        if len(query) == 0:
            query.append(
                {
                    "role": "user",
                    "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                    + "obs: `"
                    + obs
                    + "`",
                }
            )
        else:
            query.append({"role": "user", "content": "obs: `" + obs + "`"})
        prev_obs.append("obs: `" + target_obs + "`")
        prev_actions.append("act: `" + target_act + "` (" + act_repr + ")")

        #ipdb.set_trace()

        model = args.model
        total_num_tokens = num_tokens_from_messages(sys_message + query, model)
        if total_num_tokens > MAX_TOKENS[model]:
            model = "gpt-3.5-turbo-16k-0613"
            logger.info(f"Using {model} due to context limit")
            total_num_tokens = num_tokens_from_messages(sys_message + query, model)
            if total_num_tokens > MAX_TOKENS[model]:
                logger.info(
                    f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[model]}), skipping..."
                )
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                conversation.append(
                    {
                        "input": sys_message + query,
                        "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                    }
                )
                continue

        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = num_tokens_from_messages(
                sys_message + demo_message + e + query, model
            )
            if total_num_tokens > MAX_TOKENS[model]:
                if model == "gpt-3.5-turbo-16k-0613":
                    logger.info(
                        f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                    )
                    break
                else:
                    model = "gpt-3.5-turbo-16k-0613"
                    logger.info(f"Using {model} due to context limit")
                    total_num_tokens = num_tokens_from_messages(
                        sys_message + demo_message + e + query, model
                    )
                    if total_num_tokens > MAX_TOKENS[model]:
                        logger.info(
                            f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                        )
                        break
                    else:
                        demo_message.extend(e)
            else:
                demo_message.extend(e)

        #ipdb.set_trace()
        set_api('azure1', 'gpt-35-turbo-16k-mnskim')
        message = sys_message + demo_message + query
        response, info = generate_response(
            messages=message,
            model='gpt-35-turbo-16k-mnskim',
            temperature=args.temperature,
            stop_tokens=["Task:", "obs:"],
        )
        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)
        
        #ipdb.set_trace()

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    log_dir = Path(
        f"{args.log_dir}/{args.model}/{args.benchmark}{f'/no_mem' if args.no_memory else ''}{f'_no_traj' if args.no_trajectory else ''}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)

def eval_trigger(task_id, args, sample):
    
    memory = load_memory(args.memory_path)
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        memory_mapping = json.load(f)

    log_dir = Path(
        f"{args.log_dir}/{args.model}/{args.benchmark}{f'/no_mem' if args.no_memory else ''}{f'_no_traj' if args.no_trajectory else ''}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    #ipdb.set_trace()

    if not args.no_memory:
        specifier_dict = get_specifiers_from_sample(sample, as_dict=True)
        task = specifier_dict["task"]

        specifier = get_specifiers_from_sample(sample)
        retrieved_exemplar_names, scores = retrieve_exemplar_name(memory, specifier, args.retrieve_top_k)
        
        exemplars = [memory_mapping[name] for name in retrieved_exemplar_names]
        
        exemplars_tasks = []
        for exemplar in exemplars:
            item = exemplar[0]
            exemplar_task = item["content"].split("\n")[0]
            #ipdb.set_trace()
            exemplars_tasks.append(exemplar_task)
        
        output_dict = {
            "retrieved_exemplar_names": [str(item) for item in retrieved_exemplar_names], 
            "scores": [str(item) for item in scores], 
            "exemplars": exemplars,
            "task": task,
            "specifier": specifier,
            "specifier_dict": specifier_dict,
            "exemplars_tasks": exemplars_tasks,
            }
        with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
            json.dump(output_dict, f, indent=2)
        #ipdb.set_trace()

        #return retrieved_exemplar_names, scores, exemplars

def eval_traj_sample(task_id, args, sample):
    #ipdb.set_trace()

    #prompt_helper = Prompts()
    #annotate_llm = "gpt-4-0613"
    #set_api("openai")
    # gpt = GPT()
    # gpt.set_api('azure1', 'gpt-35-turbo-16k-mnskim')
    # gpt-3.5-turbo-16k-0613

    memory = load_memory(args.memory_path)
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        memory_mapping = json.load(f)


    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    #ipdb.set_trace()

    

    if not args.no_memory:
        specifier = get_specifiers_from_sample(sample)
        retrieved_exemplar_names, scores = retrieve_exemplar_name(memory, specifier, args.retrieve_top_k)
        ipdb.set_trace()
        exemplars = [memory_mapping[name] for name in retrieved_exemplar_names]


        

        #ipdb.set_trace()

    else:
        # seed = 0
        random.seed(args.seed)
        exemplars = random.sample(memory_mapping, args.retrieve_top_k)

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]
    
    # set exemplars as global exemplars
    global_exemplars = exemplars

    prev_actions = []
    prev_obs = []
    previous_k = args.see_previous_k

    add_subtask = False
    add_subtask = True

    set_api(args.api)
    completed_subtasks = set()
    obs = None
    for tt, (s, act_repr) in enumerate(zip(sample["actions"], sample["action_reprs"])):
        #ipdb.set_trace()

        # target_obs, target_act = get_target_obs_and_act(s)
        _, target_act = get_target_obs_and_act(s)
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]
        target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)

        if not args.no_memory:
            if tt == 0: # Initialize the planner agent
                planner_agent = Agent(args, sample, memory, memory_mapping)
                planner_agent.state = target_obs
                planner_agent.specifier = get_specifiers_from_sample(sample, as_dict=True)
                planner_agent.reset()
                
                remaining_subtasks_strs = planner_agent.planner.get_remaining_subtasks(completed_subtasks)
                remaining_subtasks = []
                for subtask_str in remaining_subtasks_strs:
                    subtask = planner_agent.find_closest_subtask_function(subtask_str)
                    remaining_subtasks.append(subtask)
                
                #ipdb.set_trace()

        # Continue next loop if the ground truth element is not in the cleaned html
        if len(pos_candidates) == 0:
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)

            ## Add the previous observation and action to the list
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

            # Verify subtask completion
            set_api(args.api)
            completed_subtask = planner_agent.verify_subtask_completion(prev_obs, prev_actions, remaining_subtasks)
            #ipdb.set_trace()
            if completed_subtask is not None:
                completed_subtasks.add(completed_subtask)
            # Update the set of remaining subtasks
            remaining_subtasks_strs = planner_agent.planner.get_remaining_subtasks(completed_subtasks)
            remaining_subtasks = []
            for subtask_str in remaining_subtasks_strs:
                subtask = planner_agent.find_closest_subtask_function(subtask_str)
                remaining_subtasks.append(subtask)

            conversation.append("The ground truth element is not in cleaned html")
            continue
        

        query = []
        # create the zipped list of previous observations and actions
        zipped_list = list(zip(prev_obs, prev_actions))
        # truncate the list to the last k elements
        if len(zipped_list) > previous_k:
            zipped_list = zipped_list[-previous_k:]
            
        #for o, a in zip(prev_obs, prev_actions):
        for o, a in zipped_list:
            if len(query) == 0:
                if add_subtask:
                    query.append(
                        {
                            "role": "user",
                            "content": f"Task: {sample['confirmed_task']}\nSubtasks that have not been completed yet are: {remaining_subtasks_strs}\nTrajectory:\n"
                            + o,
                        }
                    )
                else:
                    query.append(
                        {
                            "role": "user",
                            "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                            + o,
                        }
                    )
            else:
                query.append({"role": "user", "content": o})
            query.append({"role": "assistant", "content": a})

        obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
        
        #if tt > 0:
            #if obs is not None:
            #    obs_prev = obs
            #else:
            #    obs_prev = first_obs # NOTE check this is correct
            #target_obs_prev = target_obs
        
        if len(query) == 0:
            if add_subtask:
                query.append(
                    {
                        "role": "user",
                        "content": f"Task: {sample['confirmed_task']}\nSubtasks that have not been completed yet are: {remaining_subtasks_strs}\nTrajectory:\n"
                        + "Observation: `"
                        + obs
                        + "`",
                    }
                )
            else:
                query.append(
                    {
                        "role": "user",
                        "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                        + "Observation: `"
                        + obs
                        + "`",
                    }
                )
        else:
            query.append({"role": "user", "content": "Observation: `" + obs + "`"})

        # Set current trajectory of the planner agent
        planner_agent.trajectory = copy.deepcopy(query)
        planner_agent.current_obs = copy.deepcopy(obs)
        planner_agent.previous_actions = copy.deepcopy(prev_actions)

        ## Add the previous observation and action to the list
        prev_obs.append("Observation: `" + target_obs + "`")
        prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

        #ipdb.set_trace()  
        
        # Update plan
        set_api(args.api)
        if tt == 0:
            # Initial plan: use overall exemplars as reference to create the decomposed subtasks as the plan
            #updated_plan = planner_agent.plan_update(exemplars=global_exemplars, completed_subtasks=completed_subtasks)
            1

        if tt > 0:            
            # In subsequent steps, use the subtasks to update the plan
            #updated_plan = planner_agent.plan_update(exemplars=None, completed_subtasks=completed_subtasks)
            #curr_subtask

            # Get the subtask and the exemplars for the subtask
            #curr_subtask = planner_agent.find_closest_subtask_function(updated_plan)
            
            #exemplars = curr_subtask.exemplars

            adaptive_specifier = '\n'.join(specifier.split("\n")[:-1])
            adaptive_specifier += f"Task: {' '.join(remaining_subtasks_strs)}\n"
            adaptive_retrieved_exemplar_names, adaptive_scores = retrieve_exemplar_name(
                memory, adaptive_specifier, args.retrieve_top_k
            )
            # if use adaptive exemplars                
            exemplars = [memory_mapping[name] for name in adaptive_retrieved_exemplar_names]
            
            #ipdb.set_trace()

        #ipdb.set_trace()
        # add the plan to the query
        #query.append({"role": "user", "content": f"""Here is the estimated plan for the next action:\n{updated_plan}"""})
        #ipdb.set_trace()

        #ipdb.set_trace()
        #query.append({"role": "user", "content": f"""Here is the estimate of what subtasks have been completed:\n{completed_subtasks}"""})

        #ipdb.set_trace()

        total_num_tokens = num_tokens_from_messages(sys_message + query, args.model)
        if total_num_tokens > MAX_TOKENS[args.model]:
            logger.info(
                f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[args.model]}), skipping..."
            )
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            conversation.append(
                {
                    "input": sys_message + query,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )


            set_api(args.api)
            # Verify subtask completion
            completed_subtask = planner_agent.verify_subtask_completion(prev_obs, prev_actions, remaining_subtasks)
            #ipdb.set_trace()
            if completed_subtask is not None:
                completed_subtasks.add(completed_subtask)
            # Update the set of remaining subtasks
            remaining_subtasks_strs = planner_agent.planner.get_remaining_subtasks(completed_subtasks)
            remaining_subtasks = []
            for subtask_str in remaining_subtasks_strs:
                subtask = planner_agent.find_closest_subtask_function(subtask_str)
                remaining_subtasks.append(subtask)

            continue
        

        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = num_tokens_from_messages(
                sys_message + demo_message + e + query, args.model
            )
            if total_num_tokens > MAX_TOKENS[args.model]:
                logger.info(
                    f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                )
                break
            else:
                demo_message.extend(e)

        #ipdb.set_trace()
        set_api(args.api)
        message = sys_message + demo_message + query
        try:
            response, info = generate_response(
                messages=message,
                model=args.model,
                temperature=args.temperature,
                stop_tokens=["Task:", "obs:"],
            )
        except Exception as e:
            print(e)
            ipdb.set_trace()
        
        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)
        

        set_api(args.api)
        # Verify subtask completion
        completed_subtask = planner_agent.verify_subtask_completion(prev_obs, prev_actions, remaining_subtasks)
        #ipdb.set_trace()
        if completed_subtask is not None:
            completed_subtasks.add(completed_subtask)
        # Update the set of remaining subtasks
        remaining_subtasks_strs = planner_agent.planner.get_remaining_subtasks(completed_subtasks)
        remaining_subtasks = []
        for subtask_str in remaining_subtasks_strs:
            subtask = planner_agent.find_closest_subtask_function(subtask_str)
            remaining_subtasks.append(subtask)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    if args.no_trajectory:
        log_dir = Path(f"{args.log_dir}/{args.model}/{args.benchmark}/no_mem_no_traj")
    else:
        log_dir = Path(
            f"{args.log_dir}/{args.model}/{args.benchmark}{'/no_mem' if args.no_memory else ''}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)



def eval_sample(task_id, args, sample):
    assert args.no_memory

    element_acc = []
    action_f1 = []
    step_success = []
    success = []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]
    exemplars = [
        [
            {
                "role": "user",
                "content": "Observation:\n```\n<html> <div> <div> <a tock home page /> <button id=0 book a reservation. toggle open> <span> Book a reservation </span> </button> <button book a reservation. toggle open> </button> </div> <div> <select id=1 type> <option reservations true> Dine in </option> <option pickup> Pickup </option> <option delivery> Delivery </option> <option events> Events </option> <option wineries> Wineries </option> <option all> Everything </option> </select> <div id=2> <p> Celebrating and supporting leading women shaking up the industry. </p> <span> Explore now </span> </div> </div> </div> </html>\n```\nTask: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest\nPrevious actions:\nNone\nNext action:",
            },
            {"role": "assistant", "content": "`SELECT [1] [Pickup]`"},
        ],
        [
            {
                "role": "user",
                "content": "Observation:\n```\n<html> <div> <nav main menu> <ul> <li> <div button> Car Sales </div> <div id=0> <div> <div> <div> Buy A Car </div> <div> Plan Your Purchase </div> </div> <div> <h4> Its Tax Refund Time. Treat Yourself to an Upgrade. </h4> <p> With a variety of options, invest your refund in what you really want - a quality, used vehicle from Enterprise. </p> <a> View Inventory </a> </div> </div> </div> </li> <div id=1> Enterprise Fleet Management </div> </ul> </nav> <div region> <button id=2 selected pick-up date 03/19/2023> <span> <span> 19 </span> <div> <span> Mar </span> <span> 2023 </span> </div> </span> </button> </div> </div> </html>\n```\nTask: Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.\nPrevious actions:\n[searchbox]  Pick-up & Return Location (ZIP, City or Airport) (... -> TYPE: Brooklyn\n[option]  Brooklyn, NY, US Select -> CLICK\nNext action:",
            },
            {"role": "assistant", "content": "`CLICK [2]`"},
        ],
        [
            {
                "role": "user",
                "content": "Observation:\n```\n<html> <form search> <input id=6385 search q blazer search by keyword /> <button submit search> </button> <button button close> </button> </form> </html>\n```\nTask: Find a black blazer for men with L size and add to wishlist.\nPrevious actions:\n[svg]   -> CLICK\nNext action:",
            },
            {"role": "assistant", "content": "`TYPE [6385] [blazer]`"},
        ],
    ]

    prev_actions = []
    previous_k = 5
    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)

        # stop if the ground truth element is not in the top-k candidates
        pos_candidates = s["pos_candidates"]
        pos_candidates = [c for c in pos_candidates if c["rank"] < args.top_k_elements]
        pos_ids = [c["backend_node_id"] for c in pos_candidates]
        if len(pos_ids) == 0:
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            continue

        # get obs by pruning the tree with top-k candidates
        neg_candidates = s["neg_candidates"]
        neg_candidates = [c for c in neg_candidates if c["rank"] < args.top_k_elements]
        neg_ids = [c["backend_node_id"] for c in neg_candidates]
        all_candidates = pos_ids + neg_ids
        obs = get_target_obs(etree.fromstring(s["cleaned_html"]), all_candidates)

        # Generate action with OpenAI api
        query = f"Observation:\n```\n{obs}\n```\nTask: {sample['confirmed_task']}\nPrevious actions:\n"
        if len(prev_actions) > 0:
            for a in prev_actions[-previous_k:]:
                query += f"{a}\n"
        else:
            query += "None\n"
        query += "Next action:"
        query = [{"role": "user", "content": query}]
        prev_actions.append(act_repr)

        model = args.model
        total_num_tokens = num_tokens_from_messages(sys_message + query, model)
        if total_num_tokens > MAX_TOKENS[model]:
            model = "gpt-3.5-turbo-16k-0613"

            logger.info(f"Using {model} due to context limit")
            total_num_tokens = num_tokens_from_messages(sys_message + query, model)
            if total_num_tokens > MAX_TOKENS[model]:
                logger.info(f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[model]}), skipping...")
                element_acc.append(0)
                action_f1.append(0)
                step_success.append(0)
                conversation.append(
                    {
                        "input": sys_message + query,
                        "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                    }
                )
                continue

        demo_message = []
        for e_id, e in enumerate(exemplars):
            # if True:
            #    ipdb.set_trace()

            total_num_tokens = num_tokens_from_messages(sys_message + demo_message + e + query, model)
            if total_num_tokens > MAX_TOKENS[model]:
                if model == "gpt-3.5-turbo-16k-0613":
                    logger.info(f"Using {e_id} / {len(exemplars)} exemplars due to context limit")
                    break
                else:
                    model = "gpt-3.5-turbo-16k-0613"
                    logger.info(f"Using {model} due to context limit")
                    total_num_tokens = num_tokens_from_messages(sys_message + demo_message + e + query, model)
                    if total_num_tokens > MAX_TOKENS[model]:
                        logger.info(f"Using {e_id} / {len(exemplars)} exemplars due to context limit")
                        break
                    else:
                        demo_message.extend(e)
            else:
                demo_message.extend(e)

                
        set_api("azure1")
        message = sys_message + demo_message + query
        try: # Attempt to generate a response using the primary API
            response, info = generate_response(
                messages=message,
                model="gpt-35-turbo-16k-mnskim",
                temperature=args.temperature,
                stop_tokens=["Task:", "obs:"],
            )
        except Exception as err:
            logger.error(f"Primary API error: {err}")
            # try OpenAI API
            set_api("openai", "gpt-3.5-turbo-16k-0613")
            response, info = generate_response(
                messages=message,
                model="gpt-3.5-turbo-16k-0613",
                temperature=args.temperature,
                stop_tokens=["Task:", "obs:"],
            )

        # ipdb.set_trace()

        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    log_dir = Path(
        f"{args.log_dir}/{args.model}/{args.benchmark}{f'/no_mem' if args.no_memory else ''}{f'_no_traj' if args.no_trajectory else ''}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, f"{task_id}.json"), "w") as f:
        json.dump(conversation, f, indent=2)
