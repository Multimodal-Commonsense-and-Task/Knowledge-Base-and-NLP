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

from synapse.envs.mind2web.env_utils import (
    get_target_obs_and_act,
    get_target_obs,
    calculate_f1,
    parse_act_str,
    construct_act_str,
)
from synapse.utils.llm import (
    generate_response,
    num_tokens_from_messages,
    MAX_TOKENS,
    extract_from_response,
)
from synapse.memory.mind2web.build_memory import (
    load_memory,
    retrieve_exemplar_name,
    get_specifiers_from_sample,
    get_top_k_obs,
)

logger = logging.getLogger(__name__)



def eval_traj_sample(task_id, args, sample):
    memory = load_memory(args.memory_path)
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        memory_mapping = json.load(f)

    

    exemplars_annotations = {}

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

        #ipdb.set_trace()

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
        ipdb.set_trace()

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
        ipdb.set_trace()

        prev_obs.append("obs: `" + target_obs + "`")
        prev_actions.append("act: `" + target_act + "` (" + act_repr + ")")

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

        

        def annotate(demo_message):
            """
            Message to annotate an exemplar
            
            """

            #print(f"Generating lookahead")
            start_time = time.time()

            def find_ids(html):
                # Regular expression to find 'id=' followed by one or more digits
                pattern = re.compile(r'id=(\d+)')
                # Find all matches and return them
                return pattern.findall(html)
            
            ids = find_ids(query[-1]["content"])
            #ipdb.set_trace()

            actions_so_far = []
            for i in range(len(query)):
                if query[i]["role"] == "assistant":
                    actions_so_far.append(query[i]["content"])
            actions_str = "\n".join(actions_so_far)
                    
            sys_message = [
                                {
                                    "role": "system",
                                    "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id."
                                }
                            ]
            
            #ipdb.set_trace()
            lookahead_query = copy.deepcopy(query)
            lookahead_query = lookahead_query[-1:]
           
            # thoughts
            lookahead_query[-1]["content"] = f"""Given this trajectory, generate a "thought" annotation before each action, explaining the purpose and effect of each action with respect to the task."""  

            lookahead_query[-1]["content"] = f"""For each action in the given trajectory, provide a detailed 'thought' annotation that precedes the action. These annotations should explain the reasoning behind the action, considering the current situation and the overall goal of the task. Delve into the expected outcomes, potential challenges, and how the action contributes to progress in the task. Ensure that each thought annotation clearly links the preceding observations to the upcoming action, illustrating the logical flow and strategic planning involved in each step. Start each thought with Thought: and separate each thought with a new line."""

            lookahead_query[-1]["content"] = f"""For each action in the given trajectory, provide a concise 'thought' annotation that precedes the action. These annotations should explain the reasoning behind the action, considering the current situation and the overall goal of the task. Delve into the expected outcomes, potential challenges, and how the action contributes to progress in the task. Ensure that each thought annotation clearly links the preceding observations to the upcoming action, illustrating the logical flow and strategic planning involved in each step. Start each thought with Thought: and separate each thought with a new line."""

            lookahead_query[-1]["content"] = f"""For each action in the given trajectory, provide a concise 'thought' annotation that precedes the action. A thought should explain the reasoning behind the action, considering the current situation and the overall goal of the task. Ensure that each thought annotation clearly links the preceding observations to the upcoming action, illustrating the logical flow and strategic planning involved in each step. Start each thought with Thought: and separate each thought with a new line. End your generation with EOS"""              

            #lookahead_query[-1]["content"] = f"""Think step-by-step. First, predict the type of the next action. Then evaluate the elements with ids ({ids}), for expected outcome of the action on each element. Here are the actions taken so far:\n{actions_str}"""

            #ipdb.set_trace()

            lookahead_message = sys_message + demo_message + lookahead_query           
            lookahead_response, lookahead_info = generate_response(messages=lookahead_message, model='gpt-4-0613', temperature=args.temperature, stop_tokens=['EOS'])
            #ipdb.set_trace()

            #print(f"Lookahead response: {lookahead_response}")
            
            end_time = time.time()
            #print(f"Lookahead time: {end_time - start_time}")     
            #ipdb.set_trace()             
            return lookahead_response

        def insert_thoughts_into_trajectory(original_trajectory, thoughts_str):
            """
            Inserts thoughts into the original trajectory before each action.

            :param original_trajectory: A list of dictionaries representing the original trajectory.
            :param thoughts_str: A string containing all thoughts, separated by "Thought:".
            :return: A new trajectory with thoughts inserted before actions.
            """
            #ipdb.set_trace()

            # Split the thoughts into a list
            thoughts = thoughts_str.split("Thought: ")[1:]  # The first split is empty, so skip it

            # Iterate through the original trajectory and insert thoughts before actions
            new_trajectory = []
            for item in original_trajectory:
                if item['role'] == 'user' and item['content'].startswith('act:'):
                    # Insert thought before action
                    thought_content = "Thought: " + thoughts.pop(0)
                    new_trajectory.append({'role': 'user', 'content': thought_content})
                new_trajectory.append(item)

            return new_trajectory

        demo_message = []
        for e_id, e in enumerate(exemplars):
            exemplar_id = retrieved_exemplar_names[e_id]
            if exemplar_id not in exemplars_annotations:
                print(f"Annotate exemplar {e_id}")
                e_thoughts = annotate(e)
                ipdb.set_trace()
                e_thoughts_inserted = insert_thoughts_into_trajectory(e, e_thoughts)
                #ipdb.set_trace()
                e = e_thoughts_inserted

                for item in e:
                    print(item)

                # Save the exemplar with thoughts inserted to a dict so we don't annotate repeatedly
                exemplars_annotations[exemplar_id] = e_thoughts_inserted

                #ipdb.set_trace()
            else:
                print(f"Use annotated exemplar {e_id}")
                e = exemplars_annotations[exemplar_id]

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
                        #ipdb.set_trace()
                        demo_message.extend(e)
                        
                        #e_narration = narrate(e)                        
                        #demo_message.extend(e_narration)
            else:
                demo_message.extend(e)            
                #ipdb.set_trace()

                #e_narration = narrate(e)
                #demo_message.extend(e_narration)


        #ipdb.set_trace()

        message = sys_message + demo_message + query
        response, info = generate_response(
            messages=message,
            model=model,
            temperature=args.temperature,
            stop_tokens=["Task:", "obs:"],
        )
        # response, info = generate_response(messages=message,model=model,temperature=args.temperature,stop_tokens=["Task:", "obs:"],)
        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        
        #pred_act = extract_from_response(response, "`")        
        #pred_op, pred_id, pred_val = parse_act_str(pred_act)
        action_regex = r"\nAction: (.+)"
        action_part = re.search(action_regex, response)
        pred_act = action_part.group(1) if action_part else None
        pred_op, pred_id, pred_val = parse_act_str(response)

        target_op, _, target_val = parse_act_str(target_act)
        print(f"Target action: {target_act}\nPredicted action: {response}")
        #ipdb.set_trace()

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
        #conversation.append({"pred_act": pred_act, "target_act": target_act})
        conversation.append({"pred_act": response, "target_act": target_act})

        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    ipdb.set_trace()

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


            #if True:
            #    ipdb.set_trace()



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

        message = sys_message + demo_message + query
        response, info = generate_response(
            messages=message,
            model=model,
            temperature=args.temperature,
            stop_tokens=["Task:", "obs:"],
        )

        #ipdb.set_trace()

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