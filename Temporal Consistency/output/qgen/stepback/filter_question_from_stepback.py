import json

original_file = "output/qgen/stepback/Meta-Llama-3-8B-Instruct/240709Jul-0_prompt_stepback_Meta-Llama-3-8B-Instruct.json"
filtered_file = "output/qgen/stepback/Meta-Llama-3-8B-Instruct/prompt_stepback_Meta-Llama-3-8B-Instruct_filter.json"

with open(original_file, "r") as f:
    data = [json.loads(line) for line in f]

with open(filtered_file, "w") as f:
    for line in data:
        stepback_q = line["answer"]
        stepback_q = stepback_q.split("?")[0] + "?"
        line["answer"] = stepback_q
        f.write(json.dumps(line) + "\n")
