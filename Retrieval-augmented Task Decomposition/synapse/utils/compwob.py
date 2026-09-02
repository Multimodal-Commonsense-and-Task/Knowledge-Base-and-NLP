import ipdb
import re

COMPWOB_SUBTASK_DELIMITER = 'and then'

def get_subtasks(task):
    # split task by compwob subtask delimiter
    subtasks = task.split(COMPWOB_SUBTASK_DELIMITER)
    #ipdb.set_trace()
    return subtasks

def get_subtasks_from_env_name(env_name):
    # split task by compwob subtask delimiter
    subtasks = env_name.split('compositional.')[1]
    subtasks = subtasks.split('_')
    
    return subtasks