import json
import os
import argparse
from tqdm import tqdm

import torch
from transformers import BitsAndBytesConfig
import openai
import src.data, src.evaluation
from src.llm import llm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1728143)
    parser.add_argument("--eval_data", type=str, default="./data/NQ/test.json")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--n_context", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    # 0) setting
    output_path = os.path.join(args.output_path, args.name)
    if args.start == None or args.end == None:
        output_path = os.path.join(output_path)
    else:
        output_path = os.path.join(output_path, f'{args.start}_{args.end}')
    os.makedirs(output_path, exist_ok=True)
    
    # 1) prepare dataset
    
    
    
    eval_examples = src.data.load_data(args.eval_data)
    if args.start != None and args.end != None:
        eval_examples = eval_examples[args.start:args.end]
        
    eval_dataset = src.data.GPTDataset(eval_examples, args.n_context, args.max_length)

    # 2) load model


    # 3) inference
    cnt = 0
    exactmatch = []
    pred_list = []
    result = {}
    for i in tqdm(range(len(eval_dataset))):
        example = eval_dataset[i]
        pred = llm(example, args.temperature)        
        gold = eval_dataset.get_example(i)['answers']
        score = src.evaluation.ems(pred, gold)
        cnt += 1
        exactmatch.append(score)
        pred_list.append(pred)
        
        if i % 100 == 0:
            result['total_em'] = sum(exactmatch)/cnt
            result['exactmatch'] = exactmatch
            result['pred'] = pred_list
            
            with open(os.path.join(output_path, f'result_{i}.json'), 'w') as file:
                json.dump(result, file)


    result['total_em'] = sum(exactmatch)/cnt
    result['exactmatch'] = exactmatch
    result['pred'] = pred_list
    
    # print the result
    print(f"Total EM: {result['total_em']}")
    with open(os.path.join(output_path, 'result.json'), 'w') as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()