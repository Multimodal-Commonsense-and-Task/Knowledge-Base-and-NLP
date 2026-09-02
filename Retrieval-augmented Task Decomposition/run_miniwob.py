import argparse
import logging
import os
import ipdb

from synapse.agents.miniwob import Agent as MiniwobAgent

from synapse.agents.compwob import Agent as CompwobAgent

from synapse.agents.comp_planner import Agent as CompPlannerAgent
from synapse.agents.comp_planner_subtask import Agent as CompPlannerSubtaskAgent

from synapse.utils.llm import MaxRetriesException

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

    logger.info(f"Running task: {args.env_name}")
    logger.info(f"Max steps: {max_steps}")
    for i in range(args.num_episodes):
        logger.info(f"Task {args.env_name} - Episode {i}")
        #ipdb.set_trace()
        agent.reset(seed=args.seed + i)
        #ipdb.set_trace()

        if args.heuristic_termination > 0:
            # keep track of prev n actions
            prev_actions = []

        for _ in range(max_steps):
            obs = agent.filter()
            try: 
                actions = agent.act(obs)
                prev_actions.append(actions)
            except MaxRetriesException, AttributeError as e:
                logger.info(f"An error occurred: {e}")    
                #ipdb.set_trace()
                actions = None
                # 
            #ipdb.set_trace()

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

            try:
                logger.info(f"Actions:\n{actions}")
                exec(actions)
            except:
                #ipdb.set_trace()
                logger.info(f"Failed to execute action. Try again.")
            if agent.done:
                break
        
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
