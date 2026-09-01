import argparse
import logging
import os
import ipdb

from synapse.agents.miniwob import Agent as MiniwobAgent

from synapse.agents.compwob import Agent as CompwobAgent

from synapse.agents.comp_planner import Agent as CompPlannerAgent
from synapse.agents.comp_planner_subtask import Agent as CompPlannerSubtaskAgent

from synapse.agents.comp_planner_subtask import MaxLengthException

from synapse.utils.llm import MaxRetriesException

#from selenium.common.exceptions import *
from selenium.common.exceptions import WebDriverException

"""
logger = logging.getLogger("synapse")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.propagate = False
"""

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api", type=str, default="openai", choices=["openai", "azure1", "azure2"])


    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no_filter", action="store_true", default=False)
    parser.add_argument("--no_memory", action="store_true", default=False)

    # added
    parser.add_argument("--log_dir", type=str, required=False)
    parser.add_argument("--compwob_prompting_strategy", type=str, default="first", choices=["first", "second", "combination"])
    
    parser.add_argument("--heuristic_termination", type=int, default=-1, help="Terminate if the same action is repeated n times in a row")

    parser.add_argument("--comp_planner", action="store_true", default=False, help="use compositional planning agent")
    parser.add_argument("--subtask", action="store_true", default=False, help="use subtask level action generation")

    parser.add_argument("--reduce_if_fail", action="store_true", default=False, help="reduce the context if the agent fails to generate an action")

    parser.add_argument("--planner_type", type=str, default="planning", choices=["heuristic", "planning"])

    parser.add_argument("--refine_verify", action="store_true", default=False, help="use refine and verify")



    return parser

def get_agent(args):
    if args.subtask:
        assert args.comp_planner

    if args.comp_planner:
        if args.subtask:
            agent = CompPlannerSubtaskAgent(args=args)
            logger.info(f"Using CompPlannerSubtask agent")
        else:
            agent = CompPlannerAgent(args=args)
            logger.info(f"Using CompPlanner agent")
    elif args.env_name.startswith("compositional"):
        agent = CompwobAgent(args=args)
        logger.info(f"Using CompWoB agent")
    else:
        agent = MiniwobAgent(args=args)
        logger.info(f"Using MiniWoB agent")
    return agent


def main():

    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "synapse/memory/miniwob")
    #args.log_dir = os.path.join(current_path, "results/miniwob")

    #ipdb.set_trace()

    # agent
    agent = get_agent(args)

    # check if env has already been done: look at dir and see if num_episodes files exist
    
    #ipdb.set_trace()

    """
    if args.env_name in ["book-flight", "terminal", "use-autocomplete"]:
        max_steps = 2
    elif args.env_name in ["login-user", "login-user-popup"]:
        max_steps = 3
    elif args.env_name in ["guess-number", "tic-tac-toe"]:
        max_steps = 10
    else:
        max_steps = 1
    """
    max_steps = 50

    _base_path = os.path.join(args.log_dir, f"{args.model}/{args.env_name}")
    if os.path.exists(_base_path):
        # get file list in the directory
        files = os.listdir(_base_path)   
        ffiles = [item for item in files if (item.endswith('success.json') or item.endswith('fail.json'))]
        if len(ffiles) == args.num_episodes:
            # already done
            logger.info(f"Task {args.env_name} already done. Skipping...")
            return
        else:
            logger.info(f"Task {args.env_name} not done. Continuing...")


    #ipdb.set_trace()
    if args.refine_verify:
        # Allow for 3 retries
        max_retries = 3
        curr_retry = 0
    else:
        max_retries = 0
        curr_retry = 0

    logger.info(f"Running task: {args.env_name}")
    logger.info(f"Max steps: {max_steps}")
    for i in range(args.num_episodes):
        logger.info(f"Task {args.env_name} - Episode {i}")
        #ipdb.set_trace()
        agent.reset(seed=args.seed + i)
        agent.successful_actions = []
        #ipdb.set_trace()

        if args.heuristic_termination > 0:
            # Will use heuristic termination - Need to keep track of prev n actions
            prev_actions = []

        for _ in range(max_steps):
            obs = agent.filter()
            try: 
                actions = agent.act(obs)
                prev_actions.append(actions)
            except (MaxRetriesException, MaxLengthException) as e:
                logger.info(f"An error occurred: {e}")    
                #ipdb.set_trace()
                actions = None
                # 
            #ipdb.set_trace()

            # if the agent fails to generate an action end the episode
            if actions is None:
                break
            
            #ipdb.set_trace()
            if args.heuristic_termination:
                # check if last n actions are the same
                if len(prev_actions) >= args.heuristic_termination:
                    if all([prev_actions[-1] == prev_actions[i] for i in range(-args.heuristic_termination, -1)]):
                        error_str = """Terminating because last {args.heuristic_termination} actions are the same:"""
                        for f in prev_actions[-args.heuristic_termination:]:
                            error_str += f"\n{f}"
                        logger.info(error_str)
                        break

            #ipdb.set_trace()
            #if _ == 0:
                # TMP DEBUG
                #wrong_a = 'agent.click_xpath("//button[text()=\'Book Flight\']")'
                #logger.info(f"Do wrong action: {wrong_a}")
                #exec(wrong_a)
                #ipdb.set_trace()

            #ipdb.set_trace()
            #ipdb.set_trace()
            try:
                #ipdb.set_trace()
                logger.info(f"Actions:\n{actions}")
                exec(actions)
            except Exception as e:
                print(f"An error occurred: {e}")

            # At this point, the action has been executed successfully
            #ipdb.set_trace()

            #except Exception as e:
            #    ipdb.set_trace()
            #    logger.info(f"Failed to execute action. Try again.")
                #ipdb.set_trace()

            #ipdb.set_trace()

            # if the env is done
            if agent.done:

                if agent.reward > 0:
                    # slightly hacky but good enough - if the reward is positive and the agent is done, it's a success
                    logger.info(f"Success")
                    break # end the episode
                
                if args.refine_verify:
                    # allow retrying n times based on the max_retries
                    if agent.reward < 0:
                        
                        if agent.env.instance.exception is None:
                            # This is an issue with the task (i.e. submit button as first subtask)
                            logger.info(f"This task is either 1) not doable, or 2) the env is done without outputting an exception. Ending and going to next task...")
                            # the second case happens if ex) first subtask is executable but not the correct one, then if the second subtask is executed without executing the correct actions for the first subtask, the env will be done (failed) without an exception
                            #ipdb.set_trace()
                            break 

                        # we use this to simplify the failure checking. if agent.done but agent. reward <= 0 it indicates a failure from chromedriver
                        logger.info(f"Action failure detected.")

                        # reset the reward to 0
                        agent.reward = 0
                            
                        # simple retry: keep the planning simple, try the same subtask n times. if successful, subtask is successful
                        logger.info(f"Retrying with refine_verify mode.")
                        agent.failed_actions = [actions]

                        for _retry_num in range(max_retries):
                            logger.info(f"Retry {curr_retry} of {max_retries}")

                            try: 
                                r_actions = agent.refine_verify()
                                #ipdb.set_trace()
                                

                                try:
                                    logger.info(f"Actions:\n{actions}")
                                    exec(r_actions)
                                except Exception as e:
                                    print(f"An error occurred: {e}")

                            except (MaxRetriesException, MaxLengthException) as e:
                                logger.info(f"An error occurred: {e}")    
                                #ipdb.set_trace()
                                r_actions = ''
                                #
                            
                            prev_actions.append(r_actions)

                            if agent.reward >= 0:
                                # success
                                logger.info(f"Success after {curr_retry} retries.")
                                agent.successful_actions.append(r_actions)
                                #ipdb.set_trace()
                                break

                            
                            agent.failed_actions.append(r_actions)

                            #ipdb.set_trace()

                            # increment the retry counter
                            curr_retry += 1  
                            
                            
                            #ipdb.set_trace()

                        # if the agent is still not successful after max_retries, end the episode
                        logger.info(f"Failed after {curr_retry} retries.")
                        break    
                        #ipdb.set_trace()

                # Otherwise, end the episode
                else:
                    break
        
            else:
                # At this point, the action has been executed successfully - add it to the list of successful actions
                agent.successful_actions.append(actions)

            if agent.reward > 0:
                # slightly hacky but good enough - if the reward is positive and the agent is done, it's a success
                # 
                logger.info(f"Success")
                break # end the episode            

            try:
                logger.info(f"Planning")
                #ipdb.set_trace()
                #agent.curr_subtask = agent.planner.update_plan()
                curr_subtask = agent.plan_update(prev_actions=agent.successful_actions)
                if curr_subtask == "DONE":
                    #ipdb.set_trace()
                    agent.done = True
                    break 
                else:
                    agent.curr_subtask = curr_subtask
                #ipdb.set_trace()
            except:
                #ipdb.set_trace()
                logger.info(f"Failed to plan. Try again.")

        #ipdb.set_trace()
        agent.log_results()
    agent.close()


if __name__ == "__main__":
    #logger = logging.getLogger("synapse")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    #handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


    main()
