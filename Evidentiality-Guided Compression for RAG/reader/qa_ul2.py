import json
import os
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

import src.data, src.evaluation


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int, default=1728143)
    parser.add_argument("--eval_data", type=str, default="../data/NQ/test.json")
    parser.add_argument("--model_name", type=str, default="google/flan-ul2")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--n_context", type=int, default=1)
    parser.add_argument("--num_shot", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    
    # 0) setting
    output_path = os.path.join(args.output_path, args.name)
    if args.start == None or args.end == None:
        output_path = os.path.join(output_path)
    else:
        output_path = os.path.join(output_path, f'{args.start}_{args.end}')
    os.makedirs(output_path, exist_ok=True)
    
    # 1) prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    eval_examples = src.data.load_data(args.eval_data)
    if args.start != None and args.end != None:
        eval_examples = eval_examples[args.start:args.end]
    eval_dataset = src.data.Dataset(eval_examples, tokenizer, args.n_context, args.max_length)

    # 2) load model
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        device_map="auto",
        load_in_8bit=True,
    )

    # 3) inference
    cnt = 0
    exactmatch = []
    pred_list = []
    score_list = []
    result = {}
    for i in tqdm(range(len(eval_dataset))):
        example = eval_dataset[i]
        output = model.generate(example['input_ids'].cuda(), attention_mask=example['attention_mask'].cuda(), max_new_tokens=100, output_scores=True, return_dict_in_generate=True)
        pred = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        # calculate score
        logits = output.scores
        probs = [torch.nn.functional.softmax(logit, dim=-1) for logit in logits]
        top_log_probs = [torch.max(prob, dim=-1)[0] for prob in probs]  # 각 timestep에서 최고 확률의 로그 확률
        average_log_prob = torch.mean(torch.stack(top_log_probs))  # 평균 로그 확률
        
        gold = eval_dataset.get_example(i)['answers']
        score = src.evaluation.ems(pred, gold)
        cnt += 1
        exactmatch.append(score)
        pred_list.append(pred)
        score_list.append(average_log_prob.item())
        if i % 100 == 0:
            result['total_em'] = sum(exactmatch)/cnt
            result['exactmatch'] = exactmatch
            result['pred'] = pred_list
            result['score'] = score_list
            with open(os.path.join(output_path, f'result_{i}.json'), 'w') as file:
                json.dump(result, file)


    result['total_em'] = sum(exactmatch)/cnt
    result['exactmatch'] = exactmatch
    result['pred'] = pred_list
    result['score'] = score_list
    
    # print the result
    print(f"Total EM: {result['total_em']}")
    with open(os.path.join(output_path, 'result.json'), 'w') as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()