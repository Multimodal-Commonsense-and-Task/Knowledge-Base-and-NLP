import argparse
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def get_data_evidentiality(file, args):
    content = []
    label = []
    print('loading data...')
    objects = json.load(open(file, "r", encoding="utf-8"))
    print('data loaded')
    
    for d in objects:
        for iteration in range(args.max_iters):
            q = d['question']
            total_list = d['ctxs'][iteration * args.sents_per_iter:(iteration + 1) * args.sents_per_iter]
            total_list = sorted(total_list, key=lambda x: x['r_score'], reverse=True)
            c = q + ' [sep] '
            for ctx_idx, ctx in enumerate(total_list):
                if 'title' in ctx:
                    c += f' [{ctx_idx + 1}] ' + ctx['title'] + ' ' + ctx['text']
                else:
                    c += f' [{ctx_idx + 1}] ' + ctx['text']
            l = '<TOT>'
            content.append(c)
            label.append(l)
    
    assert len(content) == len(label)
    assert len(content) == args.max_iters * len(objects)
    return content, label


def get_data_for_output(file, args):
    out_data = []
    print('loading data...')
    objects = json.load(open(file, "r", encoding="utf-8"))
    print('data loaded')
    
    for d in objects:
        for iteration in range(args.max_iters):
            q = d['question']
            total_list = d['ctxs'][iteration * args.sents_per_iter:(iteration + 1) * args.sents_per_iter]
            total_list = sorted(total_list, key=lambda x: x['r_score'], reverse=True)
            out_data.append({'question': q, 'ctxs': total_list, 'answers': d['answers']})
    
    assert len(out_data) == args.max_iters * len(objects)
    return out_data


def prepare_data(file, tokenizer, args):
    content, labels = get_data_evidentiality(file, args)
    questions, contexts = zip(*[c.split(' [sep] ') for c in content])
    return list(questions), list(contexts), list(labels)


class EvidentialityDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, questions, contexts, labels, args):
        self.tokenizer = tokenizer
        self.questions = questions
        self.contexts = contexts
        self.labels = labels
        self.args = args

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        context = self.contexts[index]
        label = self.labels[index]

        input_text = f"question: {question} context: {context}"
        target_text = f"{label}"

        encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )

        inputs = {key: val.squeeze() for key, val in encoding.items()}
        targets = {key: val.squeeze() for key, val in target_encoding.items()}

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }


def merge_data(data):
    new_data = []
    q_anchor = ''
    for i in range(len(data)):
        q = data[i]['question']
        if q != q_anchor:
            q_anchor = q
            new_data.append(data[i])
        else:
            new_data[-1]['ctxs'] += data[i]['ctxs']
    return new_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./output/toy.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_data', type=str, default='../data/evaluator/test.json')
    parser.add_argument('--weight_path', type=str, default='../checkpoints/evaluator/NQ',
                        help='a checkpoint-XXXX directory written by train_evaluator.py')
    parser.add_argument('--base_model', type=str, default='google/flan-t5-large',
                        help='tokenizer the evaluator was trained with')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--sents_per_iter', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.7)
    args = parser.parse_args()

    print(args)

    if os.path.dirname(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    tokenizer.add_tokens(['<EVI>', '<NOT>'])
    model = T5ForConditionalGeneration.from_pretrained(args.weight_path, device_map='auto')

    output_data = get_data_for_output(args.eval_data, args)
    test_questions, test_contexts, test_labels = prepare_data(args.eval_data, tokenizer, args)
    test_dataset = EvidentialityDataset(tokenizer, test_questions, test_contexts, test_labels, args)

    filtered_output_data = []
    
    skip_until_new_question = False
    current_question = None

    for i in tqdm(range(len(test_dataset))):
        # If the previous iteration was evidential, skip the current iteration
        if skip_until_new_question and output_data[i]['question'] == current_question:
            continue

        with torch.no_grad():
            input_ids = test_dataset[i]['input_ids'].unsqueeze(0).cuda()
            attention_mask = test_dataset[i]['attention_mask'].unsqueeze(0).cuda()
            decoder_input_ids = torch.tensor([model.config.decoder_start_token_id]).unsqueeze(0).cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            next_token_probabilities = probabilities[0, -1, :]
            evi_prob = next_token_probabilities[tokenizer.convert_tokens_to_ids('<EVI>')].item()

            # If the current iteration is evidential, turn on the flag and append this iteration to the filtered_output_data
            if evi_prob > args.threshold:
                current_question = output_data[i]['question']
                skip_until_new_question = True
                filtered_output_data.append(output_data[i])
            else:
                filtered_output_data.append(output_data[i])

    # Merge data with the same question
    filtered_output_data = merge_data(filtered_output_data)

    # Final save
    json.dump(filtered_output_data, open(args.output_path, "w"), indent=2)


if __name__ == "__main__":
    main()