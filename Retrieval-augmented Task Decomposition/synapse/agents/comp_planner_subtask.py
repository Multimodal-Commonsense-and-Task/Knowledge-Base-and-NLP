import logging
from pathlib import Path
import os
import ipdb
import json
import copy
import re
from selenium.webdriver.common.keys import Keys
import random

import sys

class MaxLengthException(Exception):
    # Raised when the length of the trajectory exceeds the max length
    pass

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

"""
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
"""


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
    
    def get_state(self):
        """ Returns the current state of the agent. """
        return {"env": self.env,
                "prompts": self.prompts, 
                "prompt_type": self.prompt_type,
                "state": self.state,
                "task": self.task,
                "done": self.done,
                "reward": self.reward,
                "log_path": self.log_path,
                "trajectory": self.trajectory,
                "conversation": self.conversation,
                "token_stats": self.token_stats,
                "demo_traj": self.demo_traj,
                }
    
    def plan_update(self, prev_actions=None):
        curr_subtask = self.planner.update_plan(self.trajectory, prev_actions)
        return curr_subtask

    def reset(self, seed: int) -> None:
        #ipdb.set_trace()
        if not TMP:
            self.state = self.env.reset(seed=seed)
        else:
            self.state = self.env.reset(seeds=[seed])
        self.task = self.env.get_task()

        assert self.args.comp_planner == True



        self.done = False
        self.reward = 0
        self.trajectory = []
        self.conversation = []
        self.token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if self.args.comp_planner:
            #ipdb.set_trace()
            self.subtask_functions = []

            REPHRASE = True
            _task = self.task
            if REPHRASE:
                rephrased_task = self.rephrase_task(self.task)
                logger.info(f"###### Rephrased {self.task} to {rephrased_task}")

                # order
                decomposed_subtasks = self.order_subtasks(rephrased_task)
                if len(decomposed_subtasks) > 20:
                    logger.info(f"###### Too many subtasks ({len(decomposed_subtasks)}). Manually truncate to 20")
                    decomposed_subtasks = decomposed_subtasks[:20]

                numbered_subtasks = []
                for i, subtask in enumerate(decomposed_subtasks):
                    #numbered_subtasks.append(f"{i+1}. {subtask}")
                    # fast hack
                    if len(subtask) < 5:
                        continue
                    numbered_subtasks.append(f"{subtask}")

                #ipdb.set_trace()
                #_task = rephrased_task
                #self.task = rephrased_task
                #ipdb.set_trace()
                self.decomposed_subtasks = numbered_subtasks
                
                #ipdb.set_trace()

            else:
                #self.decomposed_subtasks = self.decompose_task(self.task)
                decomposed_subtasks = self.decompose_task(_task)
                decomposed_subtasks = parse_subtask_strings(decomposed_subtasks)
                self.decomposed_subtasks = decomposed_subtasks
                ipdb.set_trace()

            #ipdb.set_trace()
            for subtask in self.decomposed_subtasks:
                # if subtask doesn't start with number, then it's not a subtask
                if not subtask[0].isdigit():
                    continue
                #ipdb.set_trace()
                subtask_func = LLMFunction(subtask, self.state, self.memory, self.args.memory_path)
                self.subtask_functions.append(subtask_func)
            

            self.planner = Planner(self.subtask_functions, self.args.planner_type, args=self.args)
            self.actor = Actor(self.args.model, self.args.temperature, self.get_state)
            self.curr_subtask = self.plan_update()
            #ipdb.set_trace()


        if True:
            #ipdb.set_trace()
            #query = "Task: " + self.task + "\nState:\n" + self.state

            
            if True:

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


        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log path: {self.log_path}")        

        
    

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

    def order_subtasks(self, subtasks):
        """
        order subtasks based on the order of execution
        """
        #ipdb.set_trace()
        logger.info(f"###### Ordering subtasks: {subtasks}")

        sys_txt = "You are a large language model trained to navigate the web."

        #sys_txt = "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```"

        sys_message = [
            {
                "role": "system",
                "content": sys_txt,
            }
        ]

        #subtasks = subtasks.split('\n')
        #random.shuffle(subtasks)
        #subtasks = '\n'.join(subtasks)

        query_message = []
        
        query_message.append(
            {"role": "user", "content": "Here are the subtasks, in no particular order:\n" + subtasks}
        )

        query_message.append(
            {"role": "user", "content": "Here is the precise task instruction:\n" + self.task}
        )

        query_message.append(
            {"role": "user", "content": """Now, generate the sequence of subtasks in the order that they should be executed to complete the task in the exact specification. Follow the format, 1. [Subtask]. 2. [Subtask]. 3. [Subtask]...
            End your generation with EOS."""}
        )
        
        #query_message.append(
        #    {"role": "user", "content": """What is the first subtask that should be executed?
        #    End your generation with EOS."""}
        #)

        message = sys_message + query_message

        try:
            response, info = generate_response(
                messages=message,
                model=self.args.model,
                temperature=self.args.temperature,
                stop_tokens=['EOS'],
            )
            #ipdb.set_trace()
            ordered_subtasks = self.parse_subtask(response, remove_prefix=True)
            logger.info(f"###### Ordered subtasks: {ordered_subtasks}")

        except Exception as e:
            logger.info(f"{e}\nFailed to order the subtasks. Use the original subtasks")
            ordered_subtasks = subtasks

        return ordered_subtasks


    def rephrase_task(self, task: str) -> str:
        """
        Rephrase the task in a more natural ordering
        """

        logger.info(f"###### Rephrasing task: {task}")

        sys_txt = "You are a large language model trained to navigate the web."
        
        sys_txt = "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```"
        
        sys_message = [
            {
                "role": "system",
                "content": sys_txt,
            }
        ]

        #Repeat the task without omitting any parts, but order the steps in the chronological order of execution. Do not number the tasks, and only output the text of the task. End your generation with EOS.

        query_message = []
        query_message.append(
            {"role": "user", "content": "Here is the task:\n" + task}
        )

        """
        query_message.append(
                {"role": "user",
                "content": "Rewrite and reorder the steps for clarity, if necessary. Do not make any assumptions about the operation of the website, or omit steps. End your generation with EOS."}
            )
        """

        # V3
        query_message.append(
                {"role": "user",
                "content": """Your goal is to decompose the task into a set of high-level subtasks. For now, do not consider their order, simply list them in the order that they are presented in the task. Start each subtask with [Subtask] and end each subtask with \n.
                End your generation with EOS."""}
            )
        
        
        
        message = sys_message + query_message

        try:
            response, info = generate_response(
                messages=message,
                model=self.args.model,
                temperature=self.args.temperature,
                stop_tokens=['EOS'],
            )
            #ipdb.set_trace()
            rephrased_task = response
            #rephrased_task = 1
            logger.info(f"###### Rephrased task: {rephrased_task}")
        except Exception as e:
            logger.info(f"{e}\nFailed to rephrase the task. Use the original task")
            
        return rephrased_task

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

        # V2
        _query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task. 
             End your generation with EOS.
            """}
        ]

        # V3
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             First, repeat the task and reorder the steps to make it more coherent.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task. 
             End your generation with EOS.
            """}
        ]

        # V4
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task.
             Reorder the steps in the proper order of execution. 
             End your generation with EOS.
            """}
        ]

        # V4
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task.
             If necessary, reorder the steps in the proper order of execution. 
             End your generation with EOS.
            """}
        ]

        # V5 45.6/33.6
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task.
             Reorder the subtask steps if necessary, in their order of execution.
             End your generation with EOS.
            """}
        ]

        # v6
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. 
             Order the steps in the order that they will be executed.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task. 
             End your generation with EOS.
            """}
        ]

        # V2
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task. 
             End your generation with EOS.
            """}
        ]



        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Decompose the task into a number of high-level subtasks, then order the subtasks in the order that they must be executed to complete the task in the exact specification.
            """}
        ]

        # V3
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             First, repeat the task and if necessary, reorder the steps to make the order more coherent.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task. 
             End your generation with EOS.
            """}
        ]


        # V6
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             Assume the instructions precisely indicate the order of execution, without any errors or omissions.
             Aim to minimize the number of output subtasks, ensuring that each represents a significant and distinct aspect of the overall task.
             Order the subtask steps in the order that they should be executed to complete the task in the exact specification.
             End your generation with EOS.
            """}
        ]

        # V3
        query_message = [
            {"role": "user", 
             "content": """You will be given a task. Your goal is to decompose the task into a sequence of high-level subtasks.
             First, repeat the task and reorder the steps.
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
        #ipdb.set_trace()
        response, info, message, total_num_tokens = self.actor.act(self.task, obs, self.trajectory, self.curr_subtask)

        len_fail = False
        if response is None:
            len_fail = True
            if self.args.reduce_if_fail:
                # try reducing trajectory length until traj length is 1
                while len(self.trajectory) > 1:
                    logger.info(f"###### Failed to generate a response. Trying to reduce the trajectory length from {len(self.trajectory)} to {len(self.trajectory)-1}")
                    # remove first item 
                    self.trajectory.pop(0)
                    #self.curr_subtask = self.update_plan()
                    response, info, message, total_num_tokens = self.actor.act(self.task, obs, self.trajectory, self.curr_subtask)
                    if response is not None:
                        len_fail = False
                        break

            
            if len_fail:
                #ipdb.set_trace()
                logger.info(f"###### Failed to generate a response due to length. Terminate the episode.")
                self.done = True

                raise MaxLengthException("Failed to generate a response due to length. Terminate the episode.")
            

                self.done = True
                return None
            

                self.conversation.append(
                    {
                        "input": message,
                        "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                    }
                )    
        
        if not len_fail:
            
            actions = extract_from_response(response, "```")


            self.conversation.append(
                {"input": message, "output": response, "token_stats": info}
            )

            for k, v in info.items():
                self.token_stats[k] += v

            
            #ipdb.set_trace()
            self.trajectory.append(
                {
                    "obs": obs,
                    "act": actions,
                }
            )

            #ipdb.set_trace()

        else:
            #self.done = True
            #actions = None
            # throw an exception
            raise MaxLengthException("Failed to generate a response. Terminate the episode.")
        
        return actions


    #def refine_verify(self, task, obs, trajectory, curr_subtask, message, generated_response):
    def refine_verify(self):
        """
        see if generated response will solve the current subtask correctly

        """
        #ipdb.set_trace()
        task = self.task 
        obs = self.trajectory[-1]['obs'] # NOTE last obs
        trajectory = self.trajectory
        curr_subtask = self.curr_subtask
        message = self.conversation[-1]['input'] # NOTE last message
        generated_response = self.conversation[-1]['output'] # NOTE last generated response


        
        response, info, message, total_num_tokens = self.actor.refine_verify(task, obs, trajectory, curr_subtask, message, generated_response, self.failed_actions)

        actions = extract_from_response(response, "```")
        
        
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
    

    """
    def click_option(self, xpath: str):
        try:
            action = MiniWoBElementClickOption(xpath)
            self.step(action)
        except Exception as e:
            # Handle the exception as needed
            ipdb.set_trace()
            print(f"An error occurred in MiniWoBElementClickOption: {e}")
            raise  # Re-raise the exception
    """
            
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
        max_num_steps: -1 means all steps
        otherwise it should be even 
        """
        #ipdb.set_trace()
        if max_num_steps == -1:
            return self.demo_traj
        else:    
            if 'multi' in self.prompt_type:
                # return n steps from the end
                return self.demo_traj[-max_num_steps:]
            else:
                # return n steps from the beginning
                return self.demo_traj[:max_num_steps]
            

class Planner(object):
    def __init__(self, subtask_functions, planner_type='heuristic', args=None):
        """
        High-level planner, perform the subtasks and verify the results, move on to the next subtask
        
        """
        self.subtask_functions = subtask_functions
        self.planner_type = planner_type
        self.args = args

        # FIFO queue
        self.subtask_queue = []
        self.setup()
    
    def print_subtask_queue(self):
        for subtask_function in self.subtask_queue:
            logger.info(f"###### Subtask: {subtask_function.function_desc}")
        
    def setup(self):
        """
        Initialize the subtask queue
        """
        #ipdb.set_trace()
        for subtask_function in self.subtask_functions:
            self.subtask_queue.append(subtask_function)

        self.subtask_pointer = -1

    def update_plan(self, trajectory=None, prev_actions=None):
        
        #ipdb.set_trace()
        if self.planner_type == 'heuristic':
            return self._update_plan_heuristic(trajectory)
        elif self.planner_type == 'planning':
            return self._update_plan(trajectory, prev_actions)

    def _update_plan_heuristic(self, trajectory):
        """
        1. check the overall subtask progress, update if a subtask is completed
        2. move the subtask pointer (if necessary via the subtask progress)
        3. call the next subtask function to generate the next actions
        """

        if trajectory is None:
            # First subtask
            self.subtask_pointer = 0
            ipdb.set_trace()
        else:
            ipdb.set_trace()


        if self.subtask_pointer >= len(self.subtask_queue):
            ipdb.set_trace()
            return None
        else:
            ipdb.set_trace()
            curr_subtask = self.subtask_queue[self.subtask_pointer]
            logger.info(f"###### Current subtask: {curr_subtask.function_desc}")
        
            #ipdb.set_trace()
            return curr_subtask

    def _update_plan(self,trajectory=None, prev_actions=None):
        """
        NOTE heuristic

        1. check the overall subtask progress, update if a subtask is completed
        2. move the subtask pointer (if necessary via the subtask progress)
        3. call the next subtask function to generate the next actions

        return the next subtask function
        """
        #ipdb.set_trace()
        if (trajectory == []) or (trajectory is None):
            # First subtask
            self.subtask_pointer = 0
            #ipdb.set_trace()
            try:
                subtask = self.subtask_queue[self.subtask_pointer]
            except IndexError as e:
                ipdb.set_trace()
                #logger.info(f"{e}\nFailed to get the first subtask. Use the current subtask.")
                #subtask = self.subtask_queue[self.subtask_pointer]

            return subtask 
        else:
            #ipdb.set_trace()
            updated_pointer = self.llm_planning(trajectory, prev_actions)

            #self.subtask_pointer += 1
            self.subtask_pointer = updated_pointer

            if self.subtask_pointer >= len(self.subtask_queue):
                #pass
                # throw an exception
                return "DONE"
            else:
                curr_subtask = self.subtask_queue[self.subtask_pointer]
                logger.info(f"###### Current subtask: {curr_subtask.function_desc}")
            
                #ipdb.set_trace()
                return curr_subtask
    
    def llm_planning(self, trajectory, prev_actions):
        """
        use llm to get the next subtask
        """
        #curr_subtask_id = self.subtask_pointer
        #curr_subtask = self.subtask_queue[curr_subtask_id]

        system_message = self.get_system_message()

        query_message = self.construct_query_message(trajectory, prev_actions)

        message = system_message + query_message

        response, info = generate_response(
            messages=message,
            model=self.args.model,
            temperature=self.args.temperature,
            stop_tokens=['EOS'],
        )

        #ipdb.set_trace()

        try:
            next_subtask_id = self.parse_response(response)
            logger.info(f"###### Next subtask ID: {next_subtask_id}")
            next_subtask_id = next_subtask_id - 1
            #ipdb.set_trace()
        except Exception as e:
            logger.info(f"{e}\nFailed to parse the response. Use the current subtask.")
            next_subtask_id = self.subtask_pointer

        #ipdb.set_trace()
        return next_subtask_id
    

    def parse_response(self, text):
        # Regular expression pattern to match "[ID: <number>]" and stop at first non-digit character
        pattern = r"\[ID: (\d+)"

        # Search for the pattern in the provided text
        match = re.search(pattern, text)

        # If a match is found, convert the matching group to an integer
        if match:
            return int(match.group(1))
        else:
            return None

    def construct_query_message(self, trajectory, prev_actions):
        
        curr_subtask_id = self.subtask_pointer
        curr_subtask = self.subtask_queue[curr_subtask_id]

        # Construct query message 
        query_message = []
        
        subtasks_str = ""
        for idx, subtask_function in enumerate(self.subtask_queue):
            subtasks_str += f"{subtask_function.function_desc}\n"

        completed_subtasks_str = ""
        for idx, subtask_function in enumerate(self.subtask_queue[:curr_subtask_id]):
            completed_subtasks_str += f"{subtask_function.function_desc}\n"
        # add current subtask
        completed_subtasks_str += f"{curr_subtask.function_desc}\n"

        prev_actions_str = ""
        
        #ipdb.set_trace()
        for idx, action in enumerate(prev_actions):
            prev_actions_str += f"{action}\n"

        msg1 = []
        #msg1.append(f"""Here is the list of subtasks: \n{subtasks_str}""")
        #msg1.append(f"""Here are the actions that have been executed so far: \n{prev_actions}.""")
        #msg1.append(f"""Based on these actions, write what subtasks have been completed so far, and what is the next subtask that has not been completed yet.""")
        #msg1.append(f"""Write the completed subtasks first. Then, write the next subtask that has not been completed yet in the following format: [ID: (id of next subtask)]. End your generation with EOS.""")
        #msg1 = "\n".join(msg1)
        
        msg1.append(f"""Here is the list of subtasks:\n{subtasks_str}""")
        msg1.append(f"""Here are the subtasks that have been completed so far:\n{completed_subtasks_str}""")
        msg1.append(f"""Write the next subtask that should be executed, as [ID: (id of next subtask)]. End your generation with EOS.""")
        msg1 = "\n".join(msg1)

        query_message.append(
            {"role": "user", 
             "content": msg1}
        )

        #ipdb.set_trace()

        return query_message

    def get_system_message(self):
        """
        Get the system message
        """
        
        sys_message = [
            {
                "role": "system",
                "content": "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```",
            }
        ]
        return sys_message


class Actor(object):
    def __init__(self, model, temperature, get_state_callback):
        self.model = model
        self.temperature = temperature
        self.get_state_callback = get_state_callback
    
    def get_system_message(self):
        """
        Get the system message
        """
        
        sys_message = [
            {
                "role": "system",
                "content": "You are a large language model trained to navigate the web. To accomplish the task, use methods in the following Agent class to generate actions until you need the new state to proceed.\n```\nclass Agent:\n    def __init__(self, args):\n        ...\n\n    # Action: type a string via the keyboard\n    def type(self, characters: str) -> None:\n        ...\n\n    # Action: click an HTML element with a valid xpath\n    def click_xpath(self, xpath: str):\n        ...\n\n    # Actions: press a key on the keyboard, including:\n    # enter, space, arrowleft, arrowright, backspace, arrowup, arrowdown, command+a, command+c, command+v\n    def press(self, key_type: str) -> None:\n        ...\n\n    # Action: click an option HTML element in a list with a valid xpath\n    def click_option(self, xpath: str):\n        ...\n\n    # Action: move mouse cursor on an HTML element with a valid xpath\n    def movemouse(self, xpath: str):\n        ...\n```",
            }
        ]
        return sys_message
    
    def construct_query_message(self, task, obs, trajectory, subtask_function):
        """
        Construct a query message based on the subtask functions
        """

        # Construct query message using subtask functions
        query_message = []

        query_message.append(
            {"role": "user", "content": f"Subtask demonstration: {subtask_function.function_desc}"})

        #query_message.append(
        #    {"role": "user", 
        #     "content": f"""Generate the action(s) for the following subtask: {subtask_function.function_desc}. Here are demonstrations for the subtask:"""})

        # NOTE 
        subtask_demo_traj = subtask_function.get_demo_traj(max_num_steps=4)
        # TMP NOTE TODO check if it's not the same as other subtasks
            
        query_message.extend(subtask_demo_traj)
        
        query_message.append({"role": "user", "content": f"""Now, here is the current task and state"""})

        if subtask_function.prompt_type in ["multi_state_act"]:
            query_message.append(
                {"role": "user", "content": "Task: " + subtask_function.function_desc + "\nTrajectory:"}
            )
            for t in trajectory:
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
            {"role": "user", "content": "Observation:\n" + obs}
        )
        query_message.append(
            {"role": "user", "content": f"Generate the action(s) for the following subtask: {subtask_function.function_desc}"}
        )


        message = self.get_system_message() + query_message

        #ipdb.set_trace()
        return message

    
    def act(self, task, obs, trajectory, subtask_function):
        """
        Execute a subtask according to planner instructions
        """

        if subtask_function is None:
            # TODO improve this
            ipdb.set_trace()
            raise MaxLengthException("Failed to generate a response. Terminate the episode.")

        try:
            message = self.construct_query_message(task, obs, trajectory, subtask_function)
        except Exception as e:
            ipdb.set_trace()

        total_num_tokens = num_tokens_from_messages(message, self.model)
        if total_num_tokens > MAX_TOKENS[self.model]:
            
            #ipdb.set_trace()
            """
            self.conversation.append(
                {
                    "input": message,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            """
            return None, None, message, total_num_tokens

        #ipdb.set_trace()

        response, info = generate_response(
            messages=message,
            model=self.model,
            temperature=self.temperature,
            stop_tokens=["Observation:"],
        )

        return response, info, message, total_num_tokens

    def get_current_error(self):
        """
        Try to get current error from agent state
        """
        agent_state = self.get_state_callback()
        traceback = agent_state['env'].instance.traceback
        exception_type = agent_state['env'].instance.exception

        error_message = ""
        if not traceback is None:
            error_message = traceback.split('raise')[-1].strip()

        #ipdb.set_trace()
        return error_message, exception_type
        
    def construct_query_message_identify_error(self, message, generated_response, previous_actions, infer_error=False):

        agent_state = self.get_state_callback()
        if infer_error:
            
            # Construct query message using subtask functions
            query_message = []
            query_message.extend(message)
            #generated_actions = extract_from_response(generated_response, "```")
            
            generated_actions = ""
            for idx, actions in enumerate(previous_actions):
                generated_actions += f"Trial {idx+1}:\n{actions}\n"


            #query_message.append(
            #    {"role": "assistant", 
            #     "content": "```\n" + generated_actions + "\n```"
            #     }
            #)

            verification_message = ""
            verification_message += f"""For the current subtask, the following action(s), from previous attempts, have not been successful."""
            verification_message += f"""Identify possible errors, keeping in mind the available methods of the Agent class. End your generation with EOS.""" 
            verification_message += f"""Here are the actions: {generated_actions}"""

            
            #ipdb.set_trace()

            query_message.append(
                {"role": "user", "content": verification_message}
            )

        else:
            # Error feedback available
            error_message, exception_type = self.get_current_error()

            # Construct query message using subtask functions
            query_message = []
            query_message.extend(message)

            generated_actions = ""
            for idx, actions in enumerate(previous_actions):
                generated_actions += f"Trial {idx+1}:\n{actions}\n"

            error_message = ""
            if not exception_type is None:
                error_message = f"""For the current subtask, the following action(s), from previous attempts, have not been successful."""
                error_message += f"""Identify possible errors, keeping in mind the available methods of the Agent class. End your generation with EOS.""" 
                error_message += f"""Here are the actions: {generated_actions}"""
                error_message += f"""Here is the error message: {error_message}"""
                error_message += f"""Given the error, generate the correct action(s). End your generation with EOS."""

            query_message.append(
                {"role": "user", "content": error_message}
            )        

            #ipdb.set_trace()

        #ipdb.set_trace()

        return query_message


    def construct_query_message_repair_error(self, message, error_response):
        #ipdb.set_trace()

        # Construct query message using subtask functions
        query_message = []
        query_message.extend(message)
        
        query_message.append(
            {"role": "assistant", 
             "content": "```\n" + error_response + "\n```"
             }
        )

        repair_message = ""
        repair_message += f"Given the error, generate the correct action(s). End your generation with EOS.\n"
        #ipdb.set_trace()

        query_message.append(
            {"role": "user", "content": repair_message}
        )

        #ipdb.set_trace()

        return query_message


    def _construct_query_message_refine_verify(self, task, obs, trajectory, subtask_function, message, generated_response):

        # Construct query message using subtask functions
        query_message = []

        query_message.append(
            {"role": "user", "content": f"Subtask demonstration: {subtask_function.function_desc}"})
        
        # NOTE
        subtask_demo_traj = subtask_function.get_demo_traj(max_num_steps=4)
        
        query_message.extend(subtask_demo_traj)

        query_message.append({"role": "user", "content": f"""Now, here is the current task and state"""})

        if subtask_function.prompt_type in ["multi_state_act"]:
            query_message.append(
                {"role": "user", "content": "Task: " + subtask_function.function_desc + "\nTrajectory:"}
            )
            for t in trajectory:
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
            {"role": "user", "content": "Observation:\n" + obs}
        )
        query_message.append(
            {"role": "user", "content": f"Verify if the following action(s) solve the subtask: {subtask_function.function_desc}"}
        )
        query_message.append(
            {"role": "assistant", "content": "```\n" + generated_response + "\n```"}
        )


        message = self.get_system_message() + query_message


        #ipdb.set_trace()

        return message

    def refine_verify(self, task, obs, trajectory, subtask_function, message, generated_response, previous_actions):
        """
        see if generated response will solve
        """
        previous_actions = previous_actions
        #ipdb.set_trace()
        error_message = self.construct_query_message_identify_error(message, generated_response, previous_actions)
        #ipdb.set_trace()
        logger.info(f"###### Error identification: {error_message}")

        total_num_tokens = num_tokens_from_messages(message, self.model)

        """
        NOTE TODO
        total_num_tokens = num_tokens_from_messages(message, self.model)
        if total_num_tokens > MAX_TOKENS[self.model]:
            return None, None, message, total_num_tokens
        """

        error_response, error_info = generate_response(
            messages=error_message,
            model=self.model,
            temperature=self.temperature,
            stop_tokens=["EOS"],
        )

        error_message.append(
            {"role": "assistant", "content": "```\n" + error_response + "\n```"}
        )
        #ipdb.set_trace()

        return error_response, error_info, message, total_num_tokens
        
        repair_message = self.construct_query_message_repair_error(error_message, generated_response)
        #ipdb.set_trace()

        total_num_tokens = num_tokens_from_messages(message, self.model)

        """
        NOTE TODO
        total_num_tokens = num_tokens_from_messages(message, self.model)
        if total_num_tokens > MAX_TOKENS[self.model]:
            return None, None, message, total_num_tokens
        """    

        repair_response, repair_info = generate_response(
            messages=message,
            model=self.model,
            temperature=self.temperature,
            stop_tokens=["EOS"],
        ) 
        logger.info(f"###### Repaired actions: {repair_message}")

        ipdb.set_trace()
        return repair_response, repair_info, message, total_num_tokens

# Utils
        
def parse_subtask_strings(strings):
    # Compile a regex pattern to match strings that start with a digit followed by a period
    # and at least one space, then followed by non-space characters (actual content).
    pattern = re.compile(r'^\d+\.\s+\S+')
    
    # Filter the list of strings, keeping only those that match the pattern.
    # This skips strings that are only a digit followed by a period and space without further content.
    filtered_strings = [s for s in strings if pattern.match(s)]
    
    return filtered_strings

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