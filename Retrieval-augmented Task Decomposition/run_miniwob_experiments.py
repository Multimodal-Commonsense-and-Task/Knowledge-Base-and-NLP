import argparse
import ipdb
import os
from pathlib import Path
import subprocess

import logging



COMPWOB_KEYS=['two-way', 'three-way', 'n-way', 'transition', 'easy-medium two-way']



class MiniWoBSTasks(object):
    def __init__(self):
        self.compwob_tasks = []
        self.miniwob_tasks = []
        self.synapse_miniwob_tasks = []

        self.compwob_miniwob56 = ['book-flight', 'choose-date', 'choose-date-easy', 'choose-date-medium', 'choose-list', 'click-button', 'click-button-sequence', 'click-checkboxes', 'click-checkboxes-large', 'click-checkboxes-soft', 'click-checkboxes-transfer', 'click-collapsible', 'click-collapsible-2', 'click-color', 'click-dialog', 'click-dialog-2', 'click-link', 'click-menu', 'click-option', 'click-pie', 'click-scroll-list', 'click-shades', 'click-shape', 'click-tab', 'click-tab-2', 'click-tab-2-hard', 'click-test', 'click-test-2', 'click-widget', 'count-shape', 'email-inbox', 'email-inbox-forward-nl', 'email-inbox-forward-nl-turk', 'email-inbox-nl-turk', 'enter-date', 'enter-password', 'enter-text', 'enter-text-dynamic', 'enter-time', 'focus-text', 'focus-text-2', 'grid-coordinate', 'guess-number', 'identify-shape', 'login-user', 'login-user-popup', 'multi-layouts', 'multi-orderings', 'navigate-tree', 'search-engine', 'social-media', 'social-media-all', 'social-media-some', 'tic-tac-toe', 'use-autocomplete', 'use-spinner']

        self.compwob_compositional = {
            'two-way': ['click-button_click-checkboxes',
                        'click-button_click-checkboxes-transfer',
                        'click-button_click-dialog',
                        'click-button_click-link',
                        'click-button_click-option',
                        'click-button-sequence_click-checkboxes',
                        'click-button-sequence_click-option',
                        'click-button-sequence_login-user-popup',
                        'click-link_click-button',
                        'click-link_click-dialog',
                        'click-link_click-widget',
                        'click-link_enter-text',
                        'click-option_enter-text',
                        'click-option_login-user',
                        'click-option_navigate-tree',
                        'click-widget_enter-password',
                        'click-widget_multi-layouts',
                        'enter-password_click-option',
                        'login-user_navigate-tree',
                        'multi-layouts_login-user'
                        ],
            'three-way': ["click-button_click-option_login-user",
    "click-button-sequence_click-option_login-user",
    "click-checkboxes_click-widget_click-button-sequence",
    "click-checkboxes-transfer_click-button-sequence_enter-password",
    "click-checkboxes-transfer_enter-password_click-dialog",
    "click-dialog_click-button-sequence_enter-password",
    "click-dialog_click-checkboxes-transfer_click-widget",
    "click-link_click-button_click-dialog",
    "click-widget_click-option_click-dialog",
    "enter-password_click-checkboxes_login-user-popup"
    ],
            'n-way': [
    "click-button-sequence_click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog",
    "click-button-sequence_click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog_login-user",
    "click-link_click-button_click-checkboxes_click-dialog",
    "click-link_click-button_click-checkboxes_click-option_click-dialog",
    "click-widget_click-link_click-button_click-checkboxes_click-option_click-dialog"
],
        'transition': [
            "click-checkboxes-transfer_multi-layouts_email-inbox-forward-nl-transition",
    "click-option_login-user-transition",
    "click-option_multi-layouts_click-widget_login-user-popup-transition",
    "login-user_navigate-tree-transition",
    "login-user-popup_email-inbox-forward-nl-turk-transition"
        ],
        'easy-medium two-way': [
            "click-button_click-tab-2-hard",
    "click-button-sequence_use-autocomplete",
    "click-checkboxes-soft_enter-password",
    "click-checkboxes-soft_multi-layouts",
    "click-dialog_search-engine",
    "click-dialog-2_click-widget",
    "click-dialog-2_login-user-popup",
    "click-widget_click-checkboxes-soft",
    "enter-date_login-user",
    "use-autocomplete_click-dialog"
        ]
        }

    def get_tasks(self, name, reverse=False):
        if reverse:
            assert name == 'compwob_compositional'

        if name == 'compwob_miniwob56':
            logger.info("Dataset stats: compwob_miniwob56")
            logger.info(f"\t{len(self.compwob_miniwob56)}")
            return self.compwob_miniwob56
    
        if name == 'compwob_compositional':
            #ipdb.set_trace()

            if reverse:
                tasks = {}
                logger.info("Dataset stats: compwob_compositional (reverse)")
            else:
                tasks = self.compwob_compositional
                logger.info("Dataset stats: compwob_compositional")
            for k, v in self.compwob_compositional.items():
                if reverse:
                    _v = [f"{task}-reverse" for task in v]
                    tasks[k] = _v
                #ipdb.set_trace()
                logger.info(f"\t{k}: {len(v)}")
            #ipdb.set_trace()
            logger.info("\tTotal: " + str(sum([len(v) for k, v in self.compwob_compositional.items()])))
            return tasks

def get_html_task_names(path):

    # get all the html files from the dir
    html_files = list(Path(path).rglob('*.html'))

    # for each html file, get the task name
    task_names = []
    for html_file in html_files:
        task_name = html_file.stem
        task_names.append(task_name)

    return task_names

def run_task_list(task_list, args):
    logger.info(f"Running {len(task_list)} tasks...")
    for task in task_list:
        
        # NOTE for debug
        #task = 'click-link_enter-text'
        #task = 'click-option_enter-text'
        #task = 'click-link_click-button_click-dialog'
        #task = 'login-user-popup_email-inbox-forward-nl-turk-transition'
        #task = 'login-user_navigate-tree'
        #task = 'click-button_click-checkboxes-transfer'
        #task = 'click-button_click-option'
        #task = 'click-button_click-option_login-user'
        #task = 'click-button_click-link'

        #task = 'click-option_login-user-reverse'
        #task = 'click-button_click-checkboxes-reverse'
        #task = 'click-button_click-checkboxes-transfer-reverse'
        #task = 'click-link_click-button-reverse'
        #ipdb.set_trace()

        if args.dataset in ['compwob_compositional']:
            task = f"compositional.{task}"
        #logger.info(f"Running task: {task}")
        # run the task
        # subdomain is the task name
        #NOTE python run_miniwob.py --env_name $subdomain --seed 0 --num_episodes 50

        # run the task in a thread
        if args.subtask:
            python_file = "run_miniwob_subtask.py"
        else:
            python_file = "run_miniwob.py"
        
        command = f"python {python_file} --env_name {task} --seed {args.seed} --num_episodes {args.num_episodes} --log_dir {args.log_dir} --model {args.model} --temperature {args.temperature} --heuristic_termination {args.heuristic_termination} --api {args.api} --planner_type {args.planner_type}"
        if args.headless:
            command += " --headless"
        if args.comp_planner:
            command += " --comp_planner"
        if args.dataset in ['compwob_compositional']:
           command += f" --compwob_prompting_strategy {args.compwob_prompting_strategy}"
        if args.subtask:
            command += " --subtask"
        if args.reduce_if_fail:
            command += " --reduce_if_fail"
        if args.refine_verify:
            command += " --refine_verify"

        subprocess.run(command, shell=True)

    

def run(args):


    compwob_tasks = get_html_task_names(args.compwob_html_path)
    miniwob_tasks = get_html_task_names(args.miniwob_html_path)
    synapse_miniwob_tasks = get_html_task_names(args.synapse_miniwob_html_path)

    print("compwob_tasks: ", len(compwob_tasks))
    print("miniwob_tasks: ", len(miniwob_tasks))
    print("synapse_miniwob_tasks: ", len(synapse_miniwob_tasks))

    # get overlap between compwob and miniwob
    overlap = set(compwob_tasks).intersection(set(miniwob_tasks))
    print("overlap: ", len(overlap))

    # get overlap between compwob and synapse_miniwob
    overlap = set(compwob_tasks).intersection(set(synapse_miniwob_tasks))
    print("overlap: ", len(overlap))

    # get overlap between miniwob and synapse_miniwob
    overlap = set(miniwob_tasks).intersection(set(synapse_miniwob_tasks))
    print("overlap: ", len(overlap))

    # get diff between synapse_miniwob and miniwob
    diff = set(synapse_miniwob_tasks).difference(set(miniwob_tasks))
    print("diff: ", diff)

    compwob_miniwob56_tasks = MiniWoBSTasks().get_tasks('compwob_miniwob56')
    
    compwob_compositional_tasks = MiniWoBSTasks().get_tasks('compwob_compositional')

    compwob_compositional_tasks_reverse = MiniWoBSTasks().get_tasks('compwob_compositional', reverse=True)

    #ipdb.set_trace()

    # check each task in compwob_miniwob56_tasks is in synapse_miniwob_tasks
    # if all tasks are found, then print "all tasks found"
    all_tasks_found = True
    for task in compwob_miniwob56_tasks:
        if task not in synapse_miniwob_tasks:
            all_tasks_found = False
            print("task not found: ", task)
    if all_tasks_found:
        print("all tasks found")    
    #ipdb.set_trace()

    # select dataset
    if args.dataset == 'compwob_miniwob56':
        task_list = compwob_miniwob56_tasks
    elif args.dataset == 'compwob_compositional':
        task_list = []
        for k in COMPWOB_KEYS:
            if args.reverse:
                task_list += compwob_compositional_tasks_reverse[k]
            else:
                task_list += compwob_compositional_tasks[k]
        #ipdb.set_trace() 

    #ipdb.set_trace()
    # run task list
    run_task_list(task_list, args)

if __name__=="__main__":

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    #handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
    logger.addHandler(handler)

   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api", type=str, default="openai", choices=["openai", "azure1", "azure2"])

    parser.add_argument('--dataset', type=str, required=True, choices=['compwob_miniwob56', 'compwob_compositional'])

    parser.add_argument('--log_dir', type=str, required=True)

    # paths
    parser.add_argument('--compwob_html_path', type=str, default='/Users/minsookim/Workspace/web/compwob/miniwob-plusplus/html/compositional')
    parser.add_argument('--miniwob_html_path', type=str, default='/Users/minsookim/Workspace/web/compwob/miniwob-plusplus/html/miniwob')

    parser.add_argument('--synapse_miniwob_html_path', type=str, default='/Users/minsookim/Workspace/web/m2w2/synapse/envs/miniwob/html/miniwob')


    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no_filter", action="store_true", default=False)
    parser.add_argument("--no_memory", action="store_true", default=False)

    parser.add_argument("--compwob_prompting_strategy", type=str, default="first", choices=["first", "second", "combination"])

    parser.add_argument("--heuristic_termination", type=int, default=-1, help="Terminate if the same action is repeated n times in a row")

    parser.add_argument("--comp_planner", action="store_true", default=False, help="use compositional planning agent")
    parser.add_argument("--subtask", action="store_true", default=False, help="use subtask level action generation")

    parser.add_argument("--reduce_if_fail", action="store_true", default=False, help="reduce the context if the agent fails to generate an action")


    parser.add_argument("--planner_type", type=str, default="planning", choices=["heuristic", "planning"])

    parser.add_argument("--refine_verify", action="store_true", default=False, help="use refine and verify")

    parser.add_argument("--reverse", action="store_true", default=False, help="reverse the task")



    args = parser.parse_args()


    run(args)