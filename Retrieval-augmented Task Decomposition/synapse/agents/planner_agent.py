import ipdb
from synapse.utils.llm import (
    generate_response,
    extract_from_response,
    num_tokens_from_messages,
    MAX_TOKENS,
    extract_from_response, set_api
)

from synapse.memory.mind2web.build_memory import (
    load_memory,
    retrieve_exemplar_name,
    get_specifiers_from_sample,
    get_top_k_obs,
)

from strsimpy.levenshtein import Levenshtein

import re
import logging
import json
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
#handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
logger.addHandler(handler)


class Agent:
    def __init__(self, args, sample, memory=None, memory_mapping=None):
        self.args = args
        self.sample = sample
        self.memory = memory
        self.memory_mapping = memory_mapping

        self.task = self.sample["confirmed_task"]
        self.trajectory = None
        self.prev_actions = []
        self.current_obs = None

        #ipdb.set_trace()
        set_api(self.args.api)

    def get_state(self):
        """ Returns the current state of the agent. """
        return {
                "args": self.args,
                "sample": self.sample,
                "trajectory": self.trajectory,
                "prev_actions": self.prev_actions,
                "current_obs": self.current_obs,
                }
    

    def plan_update(self, exemplars=None, completed_subtasks=None):
        
        #ipdb.set_trace()
        curr_subtask = self.planner.update_plan(state=self.get_state(), task=self.task, exemplars=exemplars, completed_subtasks=completed_subtasks)
        return curr_subtask

    def find_first_subtask(self, string):
        subtask_pattern = r'\[Subtask\]\s*(.*?)(?=\s*\[Subtask\]|\s*$)'
        match = re.search(subtask_pattern, string)
        if match:
            return match.group(1).strip()
        return None


    def find_closest_subtask_function(self, subtask_str):
        """Finds and returns the subtask function object that has the closest description to the given subtask description."""
        
        closest_function = None
        closest_distance = float('inf')

        levenshtein = Levenshtein()

        for subtask_function in self.subtask_functions:
            distance = levenshtein.distance(subtask_str, subtask_function.function_desc)
            if distance < closest_distance:
                closest_distance = distance
                closest_function = subtask_function

        return closest_function

    #def verify_subtask_completion(self, obs_prev, obs, prev_obs, act, remaining_subtasks):
    def verify_subtask_completion(self, prev_obs_list, prev_acts_list, remaining_subtasks):
        """Verifies if any subtask has been completed by the most recent action
               prev_obs[0] , prev_actions[0]
               prev_obs[1] , prev_actions[1]

               obs is prev_obs[-1]
        """

        #obs_prev = obs_prev.strip()
        #ipdb.set_trace()

        #all_subtasks = self.planner.subtask_functions
        all_subtasks = remaining_subtasks

        sys_message = [
            {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
        ]

       

        #ipdb.set_trace()

        query_message = []

        N_PREV=5
        N_PREV = min(N_PREV, len(prev_obs_list))
        if N_PREV == 1:
            obs_prev = prev_obs_list[0]
            act = prev_acts_list[0]
            query_message.append(
            {"role": "user", "content": "Here is the most recent observation:\n" + obs_prev}
            )
            query_message.append(
                {"role": "user", "content": "Here is the action that was executed in the previous observation:\n" + act}
            )
        else:
            #ipdb.set_trace()
            query_message.append(
                    {"role": "user", "content": f"Here is the most recent {N_PREV} observations and executed actions:\n"}
                )
            for ii in range(N_PREV):
                query_message.append(
                    {"role": "user", "content": f"Observation {ii+1}:\n" + prev_obs_list[ii]}
                )
                query_message.append(
                    {"role": "user", "content": f"Action {ii+1}:\n" + prev_acts_list[ii]}
                )
            #ipdb.set_trace()
        
        #query_message.append(
        #    {"role": "user", "content": "Here is the observation after the action was executed:\n" + #obs}
        #)
        query_message.append(
            {"role": "user", "content": "Here is the task:\n" + self.task}
        )
        query_message.append(
            {"role": "user", "content": "Here are the subtasks that have not been completed yet:\n" + '\n'.join([subtask.function_desc for subtask in all_subtasks])}
        )
        query_message.append(
            {"role": "user", "content": "Assuming that the previous action was executed successfully, write the subtask that was completed by the action. If no subtask was completed, write NONE. End your generation with EOS."}
        )
        query_message.append(
            {"role": "user", "content": "Completed subtask:"}
        )

        message = sys_message + query_message

        response, info = generate_response(
            messages=message,
            model=self.args.model,
            temperature=self.args.temperature,
            stop_tokens=['EOS'],
        )

        closest_subtask_function = self.find_closest_subtask_function(response)

        #ipdb.set_trace()

        return closest_subtask_function


    def reset(self):
        #ipdb.set_trace()
        
        if True:
            #ipdb.set_trace()
            self.subtask_functions = []

            REPHRASE = True
            USE_MEM_FOR_PLAN = True
            _task = self.task
            task_exemplars = None
            if REPHRASE:
                if USE_MEM_FOR_PLAN:
                    task_specifier = get_specifiers_from_sample(self.sample)
                    retrieved_exemplar_names, scores = retrieve_exemplar_name(
                        self.memory, task_specifier, self.args.retrieve_top_k
                    )
                    task_exemplars = [self.memory_mapping[name] for name in retrieved_exemplar_names]

                rephrased_task = self.rephrase_task(self.task, task_exemplars=task_exemplars)
                logger.info(f"###### Rephrased {self.task} to {rephrased_task}")
                self.decomposed_subtasks = parse_subtasks(rephrased_task)
                
                #ipdb.set_trace()

                """
                # order
                decomposed_subtasks = self.order_subtasks(rephrased_task)
                if len(decomposed_subtasks) > 20:
                    logger.info(f"###### Too many subtasks ({len(decomposed_subtasks)}). Manually truncate to 20")
                    decomposed_subtasks = decomposed_subtasks[:20]

                #ipdb.set_trace()
        

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
                """
                #ipdb.set_trace()

            else:
                #self.decomposed_subtasks = self.decompose_task(self.task)
                decomposed_subtasks = self.decompose_task(_task)
                decomposed_subtasks = parse_subtasks(decomposed_subtasks)
                self.decomposed_subtasks = decomposed_subtasks
                #ipdb.set_trace()

            #ipdb.set_trace()
            for subtask in self.decomposed_subtasks:
                # if subtask doesn't start with number, then it's not a subtask
                #if not subtask[0].isdigit():
                #    continue
                
                #ipdb.set_trace()
                subtask_func = LLMFunction(subtask, self.state, self.memory, self.memory_mapping, specifier=self.specifier, retrieve_top_k=self.args.retrieve_top_k)
                self.subtask_functions.append(subtask_func)

            #ipdb.set_trace()            

            self.planner = Planner(self.subtask_functions, self.args.planner_type, args=self.args)
            self.actor = Actor(self.args.model, self.args.temperature, self.get_state())
            #self.curr_subtask = self.plan_update()
            #ipdb.set_trace()


        #self.planner = Planner(self.subtask_functions, self.args.planner_type, args=self.args)
        #self.actor = Actor(self.args.model, self.args.temperature, self.get_state)
        #self.curr_subtask = self.plan_update()
        #ipdb.set_trace()

    def rephrase_task(self, task: str, task_exemplars=None) -> str:
        """
        Rephrase the task in a more natural ordering
        """

        logger.info(f"###### Rephrasing task: {task}")

        
        sys_message = [
            {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
        ]

        #Repeat the task without omitting any parts, but order the steps in the chronological order of execution. Do not number the tasks, and only output the text of the task. End your generation with EOS.

        query_message = []
        
        #query_message.append(
        #    {"role": "user", "content": "Here is the task:\n" + task}
        #)
        # V3
        #query_message.append(
        #        {"role": "user",
        #        "content": """Your goal is to decompose the task into a set of high-level subtasks. For now, do not consider their order, simply list them in the order that they are presented in the task. Start each subtask with [Subtask] and end each subtask with \n.
        #        End your generation with EOS."""}
        #    )
        
        if not task_exemplars is None:
            plan_examples = []
            for exemplar in task_exemplars:
                plan_example = [exemplar[0]]
                for item in exemplar:
                    if item['content'].startswith('act'):
                        plan_example.append(
                            {"role": "assistant", "content": item['content']}
                        )
                plan_examples.append(plan_example)
        
            # retrieval augmented plan
            rap_message = []
            rap_message.append({"role": "user", "content": "Now, you will be presented with a few trajectories of similar tasks. You can refer to them to understand what structure similar websites have, and what action trajectories are likely to be executable and successful."})
            for plan_example in plan_examples:
                rap_message.extend(plan_example)

                #ipdb.set_trace()
            rap_message.append({"role": "user", "content": "Here is the task: " + task})
            #rap_message.append({"role": "user", "content": "The website is: " + self.specifier['website']})
            #rap_message.append({"role": "user", "content": "Here are the previous actions: "})
            #rap_message.append({"role": "user", "content": "Here is the current HTML: "})
            
            
            #rap_message.append({"role": "user", "content": "Your goal is to decompose the task into a set of high-level subtasks. For now, do not consider their order, simply list them in the order that they are presented in the task. Start each subtask with [Subtask] and end each subtask with \n.End your generation with EOS."})
            #ipdb.set_trace()

            rap_message.append({"role": "user", "content": "Your goal is to decompose the task into a set of high-level subtasks. For now, do not consider their order, simply list them in the order that they are presented in the task. Start each subtask with [Subtask] and end each subtask with \n. Decompose into at most three subtasks. End your generation with EOS."})
            

            #retrieved_examples_message = (
            #f"'''Here is the task: {task}"
            #+ f"\nThe website is: {website}"
            #+ f"\nHere are the previous actions: {previous_actions_str}"
            #+ f"\nHere is the current HTML: {curr_html}"
            #+ f"\nNow, you will be presented with a few trajectories of similar tasks. You can refer to them to understand what structure similar websites have, and what action trajectories are likely to be executable and successful. '''"
            #)

            #f"\nUsing this information, plan the sequence of remaining actions to complete the task. You can also choose to end the task, if you think the task is complete, by printing DONE."

        #ipdb.set_trace()
            
        message = sys_message + rap_message

        set_api(self.args.api)
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

        #ipdb.set_trace()

        return rephrased_task

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

        #ipdb.set_trace()
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

class LLMFunction(object):
    def __init__(self, function_desc, state, memory, memory_mapping, specifier=None, retrieve_top_k=3):
        
        self.function_desc = function_desc
        self.state = state
        self.memory = memory
        self.memory_mapping = memory_mapping
        self.prompts = None
        self.exemplar_name = None
        self.specifier = specifier
        self.retrieve_top_k = retrieve_top_k
        self.setup()

    def __str__(self):
        return self.function_desc

    def __repr__(self):
        return self.function_desc        

    def setup(self):
        """
        Retrieve the prompts for the function
        """
    
        #with open(os.path.join(self.memory_path, "exemplars.json"), "r") as rf:
        #retrieved_exemplar_names, scores = retrieve_exemplar_name(memory, specifier, args.retrieve_top_k)

        #_prompts = json.load(rf)
        #logger.info(f"###### Function: {self.function_desc}")
        #query = "Task: " + self.function_desc + "\nState:\n" + self.state
        #exemplar_name = retrieve_exemplar_name(self.memory, query, 3)
        #ipdb.set_trace()

        # build specifier
        specifier = f"Website: {self.specifier['website']}\nDomain: {self.specifier['domain']}\nSubdomain: {self.specifier['subdomain']}\n"
        specifier += f"Task: {self.function_desc}"
       
        retrieved_exemplar_names, scores = retrieve_exemplar_name(self.memory, specifier, self.retrieve_top_k)
        exemplars = [self.memory_mapping[name] for name in retrieved_exemplar_names]

        self.exemplars = exemplars
        logger.info(f"###### Function: {self.function_desc}")
        logger.info(f"###### Exemplars:")
        for ii, exemplar in enumerate(exemplars):
            logger.info(f"[Exemplar {ii} task] {exemplar[0]['content'].split('Trajectory')[0].strip()}")

        #ipdb.set_trace()

        #self.prompts = _prompts[exemplar_name]
        #logger.info(f"""###### Define function\n\t[Function] {self.function_desc}\n\t[Exemplars] {exemplar_name}""")
        #self.exemplar_name = exemplar_name

        
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

    def update_plan(self, state, task, exemplars=None, completed_subtasks=None):
        
        #ipdb.set_trace()
        if self.planner_type == 'heuristic':
            return self._update_plan_heuristic(trajectory)
        
        elif self.planner_type == 'planning':
            return self._update_plan(state, task, exemplars=exemplars, completed_subtasks=completed_subtasks)

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

    def _update_plan(self, state, task, exemplars=None, completed_subtasks=None):
        """
        NOTE heuristic

        1. check the overall subtask progress, update if a subtask is completed
        2. move the subtask pointer (if necessary via the subtask progress)
        3. call the next subtask function to generate the next actions

        return the next subtask function
        """
        #ipdb.set_trace()
        trajectory = state['trajectory']
        prev_actions = state['prev_actions']
        current_obs = state['current_obs']

        if len(trajectory) < -2:
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
            remaining_plan = self.llm_planning(trajectory, prev_actions, current_obs, task, exemplars=exemplars, completed_subtasks=completed_subtasks)
            return remaining_plan
            #self.subtask_pointer += 1
            """
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
            """
                

    def llm_planning(self, trajectory, prev_actions, current_obs, task, exemplars=None, completed_subtasks=None):
        """
        use llm to get the next subtask
        """
        #curr_subtask_id = self.subtask_pointer
        #curr_subtask = self.subtask_queue[curr_subtask_id]
        #ipdb.set_trace()
        system_message = self.get_system_message()

        query_message = self.construct_query_message(trajectory, prev_actions, current_obs, task, exemplars=exemplars, completed_subtasks=completed_subtasks)

        message = system_message + query_message
        
        response, info = generate_response(
            messages=message,
            model=self.args.model,
            temperature=self.args.temperature,
            stop_tokens=['EOS'],
        )

        #ipdb.set_trace()


        """
        try:
            next_subtask_id = self.parse_response(response)
            logger.info(f"###### Next subtask ID: {next_subtask_id}")
            next_subtask_id = next_subtask_id - 1
            #ipdb.set_trace()
        except Exception as e:
            logger.info(f"{e}\nFailed to parse the response. Use the current subtask.")
            next_subtask_id = self.subtask_pointer
        """

        #ipdb.set_trace()
        return response
    

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
    
    def get_remaining_subtasks(self, completed_subtasks):
        # get remaining subtasks
        remaining_subtasks = []
        for subtask_function in self.subtask_queue:
            subtask_completed = False
            for completed_subtask in completed_subtasks:
                if subtask_function.function_desc == completed_subtask.function_desc:
                    subtask_completed = True
                    break

            if not subtask_completed:
                remaining_subtasks.append(subtask_function.function_desc)
        return remaining_subtasks

    def construct_query_message(self, trajectory, prev_actions, current_obs, task, exemplars=None, completed_subtasks=None):
        
        #curr_subtask_id = self.subtask_pointer
        #curr_subtask = self.subtask_queue[curr_subtask_id]

        # Construct query message 
        query_message = []
        
        subtasks_str = ""
        for idx, subtask_function in enumerate(self.subtask_queue):
            subtasks_str += f"{subtask_function.function_desc}\n"

        exemplars_str = ""
        

        #USE_EXEMPLARS = True
        #USE_EXEMPLARS = False
        
        #ipdb.set_trace()
        if not exemplars is None:
            for idx, exemplar in enumerate(exemplars):
                _act_id = 0
                for jj, item in enumerate(exemplar):
                    if jj == 0:
                        exemplars_str += f"[Example task {idx}] {item['content'].split('Trajectory')[0].strip()}\n"
                    else:
                        if item['content'].startswith('act'):
                            exemplars_str += f"[Action {_act_id}] {item['content']}\n"
                            _act_id += 1

            query_message.append(
                {"role": "user",
                    "content": f"""You will be presented with a few trajectories of similar tasks. You can refer to them to understand what structure similar websites have, and what action trajectories are likely to be executable and successful."""
                }
            )
            query_message.append(
                {"role": "user",
                    "content": f"""Here are the exemplars:\n{exemplars_str}"""
                }
            )
        
        if not completed_subtasks is None:
            completed_subtasks_str = ""
            for subtask in completed_subtasks:
                completed_subtasks_str += f"{subtask}\n"

            remaining_subtasks = self.get_remaining_subtasks(completed_subtasks)
            
            remaining_subtasks_str = ""
            for subtask in remaining_subtasks:
                remaining_subtasks_str += f"{subtask}\n"

            

            #ipdb.set_trace()

        prev_actions_str = ""
        
        #ipdb.set_trace()
        for idx, action in enumerate(prev_actions):
            prev_actions_str += f"{action}\n"

        #msg1 = []
        #msg1.append(f"""Here is the list of subtasks: \n{subtasks_str}""")
        #msg1.append(f"""Here are the actions that have been executed so far: \n{prev_actions}.""")
        #msg1.append(f"""Based on these actions, write what subtasks have been completed so far, and what is the next subtask that has not been completed yet.""")
        #msg1.append(f"""Write the completed subtasks first. Then, write the next subtask that has not been completed yet in the following format: [ID: (id of next subtask)]. End your generation with EOS.""")
        #msg1 = "\n".join(msg1)
        
        #msg1.append()
        #msg1.append(f"""Here are the actions that have been executed so far:\n{prev_actions_str}""")
        #msg1.append(f"""Here is the current HTML:\n{current_obs}""")
        #msg1.append(f"""Write the next subtask that should be executed in the current page. End your generation with EOS.""")
        #msg1.append(f"""Using this information, plan the sequence of remaining subtasks. You can also choose to end the task, if you think the task is complete, by printing DONE. End your generation with EOS.""")
        #msg1.append(f"""Using this information, plan the sequence of remaining subtasks. Start each subtask with [Subtask] and end each subtask with \n. End your generation with EOS.""")

        #msg1.append(f"""Here are the subtasks that have been completed so far:\n{completed_subtasks_str}""")
        #msg1.append(f"""Write the next subtask that should be executed, as [ID: (id of next subtask)]. End your generation with EOS.""")
        #msg1 = "\n".join(msg1)

        #query_message.append(
        #    {"role": "user", 
        #     "content": f"""Here is the list of all subtasks:\n{subtasks_str}"""
        #    }
        #)
        
        #query_message.append(
        #    {"role": "user",
        #        "content": f"""Here are the actions that have been executed so far:\n{prev_actions_str}"""
        #        }
        #)
        query_message.append(
            {"role": "user",
                "content": f"""Here is the current HTML:\n{current_obs}"""
                }
        )

        query_message.append(
            {"role": "user",
                "content": f"""Here is the task:\n{task}"""
                }
        )

        query_message.append(
            {"role": "user",
                "content": f"""Here are the actions that have been executed so far:\n{prev_actions_str}"""
                }
        )

        query_message.append(
            {"role": "user",
                "content": f"""Using this information, plan the sequence of remaining subtasks. You can also choose to end the task, if you think the task is complete, by printing DONE. End your generation with EOS."""
                }
        )

        #ipdb.set_trace()

        if not completed_subtasks is None:
            #query_message.append(
            #    {"role": "user",
            #        "content": f"""Here are the subtasks that have been completed so far:\n{completed_subtasks_str}"""
            #    }
            #)
            #query_message.append(
            #    {"role": "user",
            #        "content": f"""Here are the subtasks that have not been completed yet:\n{remaining_subtasks_str}"""
            #    }
            #)
            query_message.append(   
                {"role": "user",
                    "content": f"""Here are the subtasks that have not been completed yet, in no particular order:\n{remaining_subtasks_str}"""
                }
            )

            # v1 order the subtasks
            #query_message.append(
            #    {"role": "user",
            #        "content": f"""Using this information, order the remaining subtasks. Start each subtask with [Subtask] and end each subtask with \n. End your generation with EOS."""
            #        }
            #)
            # v2 just return the next subtask

            #task_str = f"""Using this information, print the text of the next subtask that should be executed. End your generation with EOS."""
            task_str = f"""Of these subtasks, choose one that can be executed in the current page. End your generation with EOS."""
            query_message.append(
                {"role": "user",
                    "content": task_str
                    }
            )

            query_message.append(
                {"role": "user",
                    "content": "Subtask:"
                }
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
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
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
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
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
def parse_subtasks(input_string): 
    # Split the input string by newlines to get individual subtasks
    lines = input_string.split('\n')
    subtasks = []
    
    for line in lines:
        # Check if the line starts with "[Subtask]"
        if line.startswith("[Subtask]"):
            # Extract the subtask text by removing the "[Subtask]" part
            subtask = line[len("[Subtask]"):].strip()
            subtasks.append(subtask)
    
    return subtasks
     
def parse_subtask_strings1(strings):
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


# distance
