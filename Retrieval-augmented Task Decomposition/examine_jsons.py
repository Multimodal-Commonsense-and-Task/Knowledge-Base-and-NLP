import json
import ipdb
import os
from pathlib import Path

from synapse.utils.llm import (
    generate_response,
    extract_from_response,
    num_tokens_from_messages,
    MAX_TOKENS,
    extract_from_response, set_api
)

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

path = "/home/mnskim/workspace/web/Synapse/results/mind2web/decomp/gpt-35-turbo-16k-mnskim/test_domain"

#path = "/n05_ssd3/mnskim/home/mnskim/workspace/web/Synapse/results/mind2web/decomp/gpt-35-turbo-16k-mnskim/test_domain"

# get list of files in the directory
dir = Path(path)
files = os.listdir(dir)

# sort 0.json, 1.json, 2.json, ...
files = sorted(files, key=lambda x: int(x.split(".")[0]))
print(f"## Directory: {dir}")
print(f"## Number of files in dir: {len(files)}\n")




#ipdb.set_trace()
# read each json file
trigger_map = {}
for idx, file in enumerate(files):

    #if idx < 800:
    #    continue

    path = dir / file
    with open(path
                , "r") as f:
            data = json.load(f)

    exemplars_tasks = data["exemplars_tasks"]
    # if string starts with Task: remove it
    exemplars_tasks = [exemplar_task[6:] if exemplar_task.startswith("Task:") else exemplar_task for exemplar_task in exemplars_tasks]
    task = data["task"]

    # get max semantic similarity
    max_score = 0
    max_idx = 0
    if False:
        for idx, exemplar_task in enumerate(exemplars_tasks):
            score = util.pytorch_cos_sim(
                model.encode([task]),
                model.encode([exemplar_task])
            )
            if score > max_score:
                max_score = score
                max_idx = idx

        max_score = max_score.item()


    exemplars_tasks_str = "[Exemplar task] ".join(exemplars_tasks)

    task_len = len(task.split())
    avg_exeplar_task_len = sum([len(exemplar_task.split()) for exemplar_task in exemplars_tasks]) / len(exemplars_tasks)

    max_exeplar_task_len = max([len(exemplar_task.split()) for exemplar_task in exemplars_tasks])
    min_exeplar_task_len = min([len(exemplar_task.split()) for exemplar_task in exemplars_tasks])

    #ipdb.set_trace()

    response = ''
    if False:
        query_message = []
        query_message.append(
                {"role": "user", "content": "Here is the task:\n" + task}
            )
        query_message.append(
            {"role": "user", "content": "You will be able to see exemplar demonstrations for the task. Given the list of exemplar demonstrations, determine if the task is similar enough to any of the exemplar demonstrations, such that the task can be completed by following the exemplar demonstrations. If not, select 'No', and you will be able to search for new demonstrations after decomposing the task further. End your response with 'EOS'."
            }
        )
        query_message.append(
            {"role": "user", "content": "Here are the exemplar demonstrations:\n" + exemplars_tasks_str
            }
        )
        message = query_message

        set_api('azure1')
        llm = 'gpt-35-turbo-16k-mnskim'
        response, info = generate_response(
                messages=message,
                model=llm,
                temperature=0,
                stop_tokens=['EOS'],
            )


        ipdb.set_trace()

    
    print(f"## File {idx}: {file}")
    if task_len > min_exeplar_task_len*1.5:
        print(f"Trigger task: {task}")
        print(f"\tExemplar tasks: {exemplars_tasks}")
        trigger_map[idx] = True
    else:
        trigger_map[idx] = False


    #print(f"\tmax score: {max(data['scores'])}, max semantic similarity: {max_score}, response: {response}")
    #ipdb.set_trace()

# Print trigger map infos
print(f"## Trigger map infos")
# how mnay Triggers below idx 100
print(f"## Number of triggers below idx 100: {sum([trigger_map[idx] for idx in range(100)])}")
# how mnay Triggers above idx 100
print(f"## Number of triggers above idx 100: {sum([trigger_map[idx] for idx in range(100, len(trigger_map))])}")

# dump trigger_map
with open("trigger_map.json", "w") as f:
    json.dump(trigger_map, f, indent=2)
#ipdb.set_trace()