import json
from tqdm import tqdm

original_input = []

with open("output/qgen/stepback/Meta-Llama-3-8B-Instruct/240709Jul-0_prompt_stepback_Meta-Llama-3-8B-Instruct.json") as f:
    questions = [json.loads(line) for line in f]

with open("dataset/test.hard.json") as f:
    inputs = [json.loads(line) for line in f]
inputs = [i for i in inputs if i['idx'].endswith("#0")]

filtered_q = {}
for q in tqdm(questions):
    question = q['answer']
    question = question.split("\n")[0].strip()
    filtered_q[q['idx']] = question

with open("dataset/test.stepback_question.json", "w") as fw:
    for i in inputs:
        idx = i['idx']
        question = filtered_q[idx]
        i['question'] = question
        i.pop("paragraphs")
        i.pop("targets")
        fw.write(json.dumps(i) + "\n")