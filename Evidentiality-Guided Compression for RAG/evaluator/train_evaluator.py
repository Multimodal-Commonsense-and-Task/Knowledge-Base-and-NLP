import argparse
import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# sorted when hard neg
def sorted_from_list(data, num_samples):
    if num_samples > len(data):
        return data
    return data[:num_samples]

# sample when not hard neg
def sample_from_list(data, num_samples, args):
    if args.hardneg:
        return sorted_from_list(data, num_samples)        
    if num_samples > len(data):
        return data
    return random.sample(data, num_samples)


def get_data_evidentiality(file, args, tokenizer, mode='val'):
    content = []
    label = []
    print('loading data...')
    objects = json.load(open(file, "r", encoding="utf-8"))
    print('data loaded')
    
    if mode == 'train':
        objects = objects[args.start:args.end]
    elif mode == 'val':
        objects = objects[:args.val_num]
    for d in objects:
        q = d['question']
        pos_list = sample_from_list(d[args.pos_key], args.pos_cnt, args)
        neg_list = sample_from_list(d[args.neg_key], args.neg_cnt, args)
            
        for pos in pos_list:
            if args.notitle:
                c = q + ' [sep] ' + pos['text']
                text = f'question: {q} context: {pos["text"]}'
            else:
                c = q + ' [sep] ' + pos['title'] + ' ' + pos['text']
                text = f'question: {q} context: {pos["title"]} {pos["text"]}'
            l = '<EVI>'
            length = len(tokenizer(text)['input_ids'])
            if length > args.max_length:
                continue
            content.append(c)
            label.append(l)
        for neg in neg_list:
            if args.notitle:
                c = q + ' [sep] ' + neg['text']
                text = f'question: {q} context: {neg["text"]}'
            else:
                c = q + ' [sep] ' + neg['title'] + ' ' + neg['text']
                text = f'question: {q} context: {neg["title"]} {neg["text"]}'
            length = len(tokenizer(text)['input_ids'])
            l = '<NOT>'
            if length > args.max_length:
                continue
            content.append(c)
            label.append(l)

    return content, label

def prepare_data(file, tokenizer, args, mode='val'):
    data = get_data_evidentiality(file, args, tokenizer, mode)
    content, labels = data
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./dataset/preprocessed/train.json')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_file', type=str, default='./dataset/preprocessed/dev.json')
    parser.add_argument('--pos_key', type=str, default='positive_ctxs')
    parser.add_argument('--neg_key', type=str, default='negative_ctxs')
    parser.add_argument('--model_path', type=str, default='google/flan-t5-large')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100000000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--val_num', type=int, default=500)
    parser.add_argument('--notitle', action='store_true')
    parser.add_argument('--pos_cnt', type=int, default=4)  # Number of positive samples
    parser.add_argument('--neg_cnt', type=int, default=12)  # Number of negative samples
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--hardneg', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    train_file = args.train_file
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model_path = args.model_path

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(['<EVI>', '<NOT>'])
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map='auto')
    model.resize_token_embeddings(len(tokenizer))
    

    train_questions, train_contexts, train_labels = prepare_data(train_file, tokenizer, args, mode='train')
    val_questions, val_contexts, val_labels = prepare_data(args.val_file, tokenizer, args, mode='val')


    train_dataset = EvidentialityDataset(tokenizer, train_questions, train_contexts, train_labels, args)
    val_dataset = EvidentialityDataset(tokenizer, val_questions, val_contexts, val_labels, args)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



    
    training_args = TrainingArguments(
        output_dir=args.save_path,
        evaluation_strategy="steps", 
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10, 
        save_strategy="steps", 
        save_steps=args.eval_steps, 
        eval_steps=args.eval_steps, 
        seed=SEED, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()