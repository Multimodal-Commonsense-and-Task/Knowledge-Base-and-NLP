import logging
from pathlib import Path
import os
import ipdb
import json
import copy
from selenium.webdriver.common.keys import Keys

import sys

# TMP NOTE compwob assumes oracle exemplar retriever

TMP=True # use compwob or not

if not TMP:
    # Synapse miniwob
    from synapse.envs.miniwob.environment import MiniWoBEnv
else:
    # Compwob added
    #ipdb.set_trace()
    sys.path.append('/Users/minsookim/Workspace/web/compwob')
    #sys.path.append('/Users/minsookim/Workspace/web/compwob/miniwob-plusplus/python')
    #sys.path.append('/Users/minsookim/miniconda3/envs/synapse_comp/lib/python3.10/site-packages/')
    #from miniwob_plusplus.environment import MiniWoBEnvironment as MiniWoBEnv
    #
    from miniwob.environment import MiniWoBEnvironment as MiniWoBEnv 
    #from synapse.envs.compwob.environment import MiniWoBEnvironment as MiniWoBEnv
    #from synapse.envs.compwob.environment import MiniWoBEnv
    #ipdb.set_trace()

# Use synapse's code
from synapse.envs.miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)

# compwob code
"""
from miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
"""

# synapse's code adds special key handling for osx (not criticial)
"""
from miniwob.action import MiniWoBType
from synapse.envs.miniwob.action import (
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
)
"""

from synapse.memory.miniwob.builde_memory import load_memory, retrieve_exemplar_name
from synapse.utils.llm import (
    generate_response,
    extract_from_response,
    num_tokens_from_messages,
    MAX_TOKENS,
    extract_from_response, set_api
)
from synapse.utils.compwob import get_subtasks, get_subtasks_from_env_name
import ipdb

#logger = logging.getLogger("synapse")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
#handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
logger.addHandler(handler)

#logger.setLevel(logging.INFO)
#handler = logging.StreamHandler()    
#handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))        
#logger.addHandler(handler)

ENV_TO_FILTER = [
    "book-flight",
    "click-collapsible-2",
    "click-menu",
    "click-pie",
    "click-shape",
    "click-tab-2",
    "click-tab-2-hard",
    "count-shape",
    "email-inbox",
    "email-inbox-forward-nl",
    "email-inbox-forward-nl-turk",
    "email-inbox-nl-turk",
    "find-word",
    "grid-coordinate",
    "login-user-popup",
    "social-media",
    "social-media-some",
    "terminal",
    "tic-tac-toe",
    "use-autocomplete",
]


class Agent:
    def __init__(self, args):
        self.args = args
        #self.env = MiniWoBEnv(subdomain=args.env_name, headless=args.headless) # orig
        #self.env = MiniWoBEnv(subdomain=args.env_name) # compwob
        self.env = MiniWoBEnv(subdomain=args.env_name, headless=args.headless) # compwob

        #ipdb.set_trace()

        # NOTE TODO this will set no_filter true for all compwob tasks
        #if self.args.env_name not in ENV_TO_FILTER:
        if True:
            self.args.no_filter = True
        if not args.no_memory:
            self.memory = load_memory(args.memory_path)
        self.prompts = None
        self.prompt_type = None
        self.state = None
        self.task = None
        self.done = False
        self.reward = 0
        self.log_path = None
        self.trajectory = None
        self.conversation = None
        self.token_stats = None
        self.demo_traj = []

        #ipdb.set_trace()
        #set_api('openai')
        set_api(args.api)
    

    def reset(self, seed: int) -> None:
        #ipdb.set_trace()
        if not TMP:
            self.state = self.env.reset(seed=seed)
        else:
            self.state = self.env.reset(seeds=[seed])
        self.task = self.env.get_task()

        if self.args.comp_planner:
            self.subtask_functions = []
            self.decomposed_subtasks = self.decompose_task(self.task)
            #ipdb.set_trace()
            for subtask in self.decomposed_subtasks:
                subtask_func = LLMFunction(subtask, self.state, self.memory, self.args.memory_path)
                self.subtask_functions.append(subtask_func)
            #ipdb.set_trace()    

        self.done = False
        self.reward = 0
        self.trajectory = []
        self.conversation = []
        self.token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if self.args.no_memory:
            if self.args.env_name == "click-tab-2-hard":
                exemplar_name = "click-tab-2"
            elif self.args.env_name in [
                "email-inbox",
                "email-inbox-forward-nl",
                "email-inbox-forward-nl-turk",
            ]:
                exemplar_name = "email-inbox-nl-turk"
            else:
                exemplar_name = self.args.env_name
        else:
            #ipdb.set_trace()
            query = "Task: " + self.task + "\nState:\n" + self.state

            if not self.args.comp_planner:

                if self.args.env_name.startswith("compositional."): # a CompWob task
                    subtasks = get_subtasks_from_env_name(self.args.env_name)
                    first_task = subtasks[0]
                    second_task = subtasks[1]

                    #ipdb.set_trace()
                    if self.args.compwob_prompting_strategy == 'first':
                        #top_k_exemplars_first = retrieve_exemplar_name(self.memory, first_task, 10)
                        #top_k_exemplars_second = retrieve_exemplar_name(self.memory, second_task, 10)
                        #ipdb.set_trace()
                    
                        exemplar_name = first_task
                    elif self.args.compwob_prompting_strategy == 'second':
                        exemplar_name = second_task
                    else:
                        ipdb.set_trace()
                    
                else:
                    exemplar_name = retrieve_exemplar_name(self.memory, query, 3)

            else: # Compositional Planner

                # NOTE tmp 
                exemplar_name = self.subtask_functions[0].exemplar_name

                # set overall prompt type
                self.set_prompt_type()
                
                #ipdb.set_trace()

        #ipdb.set_trace()
        logger.info(f"###### Task name: {self.task}")
        if self.args.env_name.startswith("compositional."): # a CompWob task
            logger.info(f"###### CompWob Prompting Strategy: {self.args.compwob_prompting_strategy}")
        logger.info(f"###### Exemplar name: {exemplar_name}")
        #ipdb.set_trace()


        # Initialize the base path with the log directory and model/env_name
        base_path = os.path.join(self.args.log_dir, f"{self.args.model}/{self.args.env_name}")

        # Initialize the filename starting with 'seed_' and the seed value
        filename = f"seed_{seed}"

        # Check if no_filter is true and env_name is in the specified filter list
        #if self.args.no_filter and self.args.env_name in ENV_TO_FILTER:
        if True:
            filename = f"no_filt_{filename}"  # Prefix filename with 'no_filt_'

        # Check if no_memory is true
        if self.args.no_memory:
            filename = f"no_mem_{filename}"  # Prefix filename with 'no_mem_'

        # Check if exemplar_name is different from env_name
        if exemplar_name != self.args.env_name:
            filename += f"_{exemplar_name}"  # Append '_exemplar_name' to the filename

        # Add the '.json' extension to the filename
        filename += ".json"

        # Combine the base path and filename to create the full path
        self.log_path = Path(os.path.join(base_path, filename))


        """
        self.log_path = Path(
            os.path.join(
                self.args.log_dir,
                f"{self.args.model}/{self.args.env_name}/{f'no_filt_' if self.args.no_filter and self.args.env_name in ENV_TO_FILTER else ''}{f'no_mem_' if self.args.no_memory else ''}seed_{seed}{'' if exemplar_name == self.args.env_name else f'_{exemplar_name}'}.json",
            )
        )
        """

        #ipdb.set_trace()
        #if self.log_path.parent.exists():
           # get file list in the directory
            #files = os.listdir(self.log_path.parent)   
            #ffiles = [item for item in files if (item.endswith('success.json') or item.endswith('fail.json'))]

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log path: {self.log_path}")        

        if not self.args.comp_planner:
            1
                
            #ipdb.set_trace()
            with open(os.path.join(self.args.memory_path, "exemplars.json"), "r") as rf:
            
                _prompts = json.load(rf)
                if not exemplar_name in _prompts:
                    # unseen task
                    # get a prompt through retrieval
                    #ipdb.set_trace()
                    logger.info(f"###### Unseen task: {exemplar_name}")
                    exemplar_name = retrieve_exemplar_name(self.memory, query, 3)
                    self.prompts = _prompts[exemplar_name]
                    logger.info(f"###### Using retrieved exemplar: {exemplar_name}")
                else:
                    self.prompts = _prompts[exemplar_name]
            
                #demo = self.prompts["demo"]

        #else:
            #demo1 = self.subtask_functions[0].prompts['demo']
            #demo2 = self.subtask_functions[1].prompts['demo']
            #demo = self.subtask_functions[0].prompts['demo'] # NOTE TODO just take first subtask for now
            #ipdb.set_trace()
            
            # TODO set self.prompt_type to first subtask's prompt type
            #1

        #ipdb.set_trace()

        #self.demo_trajs = [] # reset
        #for subtask_function in self.subtask_functions:
            #func_demo_traj = make_demo_traj(subtask_function)
            #ipdb.set_trace()

    

    def filter(self) -> str:
        """

        """
        
        # TMP NOTE TODO
        if not self.args.comp_planner:
            ipdb.set_trace()
        
        else:
            #ipdb.set_trace()
            prompt_type = self.prompt_type
            #demo = self.prompts["demo"]
            #prompts = self.prompts
            #prompts = None
            
            # Debugging/TMPsolution
            #prompt_type = self.subtask_functions[0].prompt_type # NOTE debug
            #demo = self.subtask_functions[0].prompts['demo'] # NOTE debug
            #prompts = self.subtask_functions[0].prompts # NOTE debug

        if prompt_type in ["state_act", "multi_state_act"]:
            obs = self.state
        

        return obs

    def decompose_task(self, task: str) -> str:
        """
        Decompose the task into subtask functions
        """
        logger.info(f"###### Decomposing task: {task}")

        sys_message = [
            {
                "role": "system",
                "content": "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```",
            }
        ]

        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task. 
             End your generation with EOS.
            """}
        ]
        query_message.append(
            {"role": "user", "content": "Here is the task:\n" + task}
        )
        message = sys_message + query_message

        try:
            response, info = generate_response(
                messages=message,
                model=self.args.model,
                temperature=self.args.temperature,
                stop_tokens=['EOS'],
            )
            parsed_subtasks = self.parse_subtask(response, remove_prefix=True)
            logger.info(f"###### Decomposed subtasks: {parsed_subtasks}")
        except Exception as e:
            logger.info(f"{e}\nFailed to decompose the task. Use the original task")
            parsed_subtasks = []

        #ipdb.set_trace()

        return parsed_subtasks

    def parse_subtask(self, subtask: str, remove_prefix=False) -> str:
        """
        parse llm response into subtask list
        """
        subtasks = subtask.split('\n')
        out = []
        for subtask in subtasks:
            if subtask != '':
                if remove_prefix:
                    subtask = subtask.replace('Subtask', '')
                out.append(subtask)
        return out

    def act(self, obs: str):
        sys_message = [
            {
                "role": "system",
                "content": "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```",
            }
        ]

        #ipdb.set_trace()
        # NOTE 
        if self.args.comp_planner:
            
            # Construct query message using subtask functions
            query_message = []
            for subtask_function in self.subtask_functions:

                #if 'multi' in subtask_function.prompt_type:
                    #ipdb.set_trace()

                query_message.append(
                    {"role": "user", "content": f"Subtask demonstration: {subtask_function.function_desc}"}
                )
                # NOTE 
                subtask_demo_traj = subtask_function.get_demo_traj(max_num_steps=-1)
                # TMP NOTE TODO check if it's not the same as other subtasks
                redundant = False
                tmp_check = subtask_demo_traj[0]['content']
                for item in query_message:
                    if tmp_check == item['content']:
                        redundant = True
                        break
                
                #ipdb.set_trace()
                if not redundant: # only add if not redundant    
                    query_message.extend(subtask_demo_traj)
                #else:
                    #ipdb.set_trace()
            #ipdb.set_trace()


        else:
            # Default prompting method
            query_message = copy.deepcopy(self.demo_traj)

        ipdb.set_trace()
        if self.prompt_type in ["multi_state_act"]:
            query_message.append(
                {"role": "user", "content": "Task: " + self.task + "\nTrajectory:"}
            )
            for t in self.trajectory:
                query_message.append(
                    {
                        "role": "user",
                        "content": "Observation:\n" + t["obs"] + "\nAction:",
                    }
                )
                query_message.append(
                    {"role": "assistant", "content": "```\n" + t["act"] + "\n```"}
                )

        query_message.append(
            {"role": "user", "content": "Observation:\n" + obs + "\nAction:"}
        )
        message = sys_message + query_message
        total_num_tokens = num_tokens_from_messages(message, self.args.model)
        if total_num_tokens > MAX_TOKENS[self.args.model]:
            self.conversation.append(
                {
                    "input": message,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            return None
        response, info = generate_response(
            messages=message,
            model=self.args.model,
            temperature=self.args.temperature,
            stop_tokens=["Observation:"],
        )
        self.conversation.append(
            {"input": message, "output": response, "token_stats": info}
        )
        for k, v in info.items():
            self.token_stats[k] += v
        actions = extract_from_response(response, "```")

        #ipdb.set_trace()
        self.trajectory.append(
            {
                "obs": obs,
                "act": actions,
            }
        )

        #ipdb.set_trace()

        return actions

    def step(self, action):
        #ipdb.set_trace()
        self.state, reward, self.done, _ = self.env.step(action)
        #ipdb.set_trace()
        if self.done:
            self.reward = reward

    def log_results(self):
        filename = os.path.splitext(os.path.basename(self.log_path))[0]
        with open(self.log_path, "w") as f:
            json.dump(self.conversation, f, indent=2)
        if self.reward > 0:
            new_file_path = self.log_path.with_name(f"{filename}_success.json")
        else:
            new_file_path = self.log_path.with_name(f"{filename}_fail.json")
        os.rename(self.log_path, new_file_path)

    # Action: type a string via the keyboard
    def type(self, characters: str) -> None:
        action = MiniWoBType(characters)
        self.step(action)

    def click_xpath(self, xpath: str):
        action = MiniWoBElementClickXpath(xpath)
        self.step(action)

    def press(self, key_type: str) -> None:
        if key_type == "enter":
            action = MiniWoBType("\n")
        elif key_type == "space":
            action = MiniWoBType(" ")
        elif key_type == "arrowleft":
            action = MiniWoBType(Keys.LEFT)
        elif key_type == "arrowright":
            action = MiniWoBType(Keys.RIGHT)
        elif key_type == "backspace":
            action = MiniWoBType(Keys.BACKSPACE)
        elif key_type == "arrowup":
            action = MiniWoBType(Keys.UP)
        elif key_type == "arrowdown":
            action = MiniWoBType(Keys.DOWN)
        elif key_type in ["command+a", "command+c", "command+v"]:
            action = MiniWoBType(key_type)
        else:
            raise ValueError("Invalid instruction")
        self.step(action)

    def click_option(self, xpath: str):
        action = MiniWoBElementClickOption(xpath)
        self.step(action)

    def movemouse(self, xpath: str):
        action = MiniWoBMoveXpath(xpath)
        self.step(action)

    def close(self):
        self.env.close()

    def set_prompt_type(self):
        """
        set overall prompt type based on demos
        """
        #ipdb.set_trace()
        found_trajectory = False
        for subtask_function in self.subtask_functions:
            demo = subtask_function.prompts["demo"]
            if "trajectory" in demo[0]:
                found_trajectory = True
                break
        
        found_obs = False
        for subtask_function in self.subtask_functions:
            demo = subtask_function.prompts["demo"]
            if "obs" in demo[0]:
                found_obs = True
                break
        
        #if self.args.no_filter: # NOTE state_act or multi_state_act
        if True:
            #if "trajectory" not in demo[0]:
            # if trajectory not in any subtask, then overall prompt type is state_act
            if not found_trajectory:
                self.prompt_type = "state_act"
                # TODO keep a list of env names belonging to subtask exemplars
                assert self.args.env_name != "click-pie"  # context limit
            else: # if trajectory in any subtask, then set overall prompt type to multi_state_act
                self.prompt_type = "multi_state_act"
                assert self.args.env_name != "book-flight"  # context limit
            

        logger.info(f"###### Set subtask prompt types - Found trajectory: {found_trajectory}, Found obs: {found_obs}")
        logger.info(f"###### Set overall prompt type: {self.prompt_type}")

class LLMFunction(object):
    def __init__(self, function_desc, state, memory, memory_path):
        
        self.function_desc = function_desc
        self.state = state
        self.memory = memory
        self.memory_path = memory_path
        self.prompts = None
        self.exemplar_name = None
        self.setup()
        

    def setup(self):
        """
        Retrieve the prompts for the function
        """
    
        #ipdb.set_trace()
        with open(os.path.join(self.memory_path, "exemplars.json"), "r") as rf:
            _prompts = json.load(rf)
            #logger.info(f"###### Function: {self.function_desc}")
            query = "Task: " + self.function_desc + "\nState:\n" + self.state
            exemplar_name = retrieve_exemplar_name(self.memory, query, 3)
            self.prompts = _prompts[exemplar_name]
            logger.info(f"""###### Define function\n\t[Function] {self.function_desc}\n\t[Exemplars] {exemplar_name}""")
            self.exemplar_name = exemplar_name

        demo = self.prompts["demo"]

        #if self.no_filter:
        if True:
            if "trajectory" not in demo[0]:
                # NOTE not a trajectory exemplar
                self.prompt_type = "state_act" 
                assert self.exemplar_name != "click-pie"  # context limit
            else:
                # NOTE trajectory exemplar
                self.prompt_type = "multi_state_act"
                assert self.exemplar_name != "book-flight"  # context limit
        
        self.demo_traj = make_demo_traj(self)

        #ipdb.set_trace()
        #self.prompts = _prompts[self.function_desc]
        #ipdb.set_trace()

    def get_demo_traj(self, max_num_steps=-1):
        """
        Get the demo trajectory for the function
        """
        #ipdb.set_trace()
        if max_num_steps == -1:
            return self.demo_traj
        else:
            return self.demo_traj[:max_num_steps]

def make_demo_traj(subtask_function):
    """
    add demo to demo traj
    """
    prompts = subtask_function.prompts
    demo = prompts["demo"]
    exemplar_name = subtask_function.exemplar_name
    prompt_type = subtask_function.prompt_type

    demo_traj = []
    

    if prompt_type == "state_act" and "ablation_act_prompt" in prompts:
        demo_traj.append(
            {"role": "user", "content": prompts["ablation_act_prompt"]}
        )
    for d in demo:
        if prompt_type == "state_act":
            if "state" in d:  # fewer states due to context limit
                demo_traj.append(
                    {
                        "role": "user",
                        "content": "Observation:\n" + d["state"] + "\nAction:",
                    }
                )
                demo_traj.append(
                    {"role": "assistant", "content": "```\n" + d["act"] + "\n```"}
                )
        elif prompt_type == "multi_state_act":
            if exemplar_name in [
                "login-user-popup",
                "terminal",
                "use-autocomplete",
            ]:  # context limit
                if len(demo_traj) > 0:
                    break
            if all(
                "state" in t for t in d["trajectory"]
            ):  # fewer states due to context limit
                demo_traj.append(
                    {
                        "role": "user",
                        "content": "Task: " + d["task"] + "\nTrajectory:",
                    }
                )
                for t in d["trajectory"]:
                    demo_traj.append(
                        {
                            "role": "user",
                            "content": "Observation:\n" + t["state"] + "\nAction:",
                        }
                    )
                    demo_traj.append(
                        {
                            "role": "assistant",
                            "content": "```\n" + t["act"] + "\n```",
                        }
                    )
        

    return demo_traj