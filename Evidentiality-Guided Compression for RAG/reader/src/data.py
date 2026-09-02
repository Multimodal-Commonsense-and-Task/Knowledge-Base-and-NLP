import torch
import random
import json
import numpy as np

from transformers import DataCollatorForSeq2Seq

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples




class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 n_context=None,
                 max_length=2048,
                 question_prefix='Question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.tokenizer = tokenizer
        self.n_context = n_context
        self.max_length = max_length
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']#self.question_prefix + " " + example['question']
        target = self.get_target(example)
        few_shot = "who won a million on deal or no deal\nAnswer: Tomorrow Rodriguez\n\nwho is the woman washing the car in cool hand luke\nAnswer: Joy Harmon\n\nwho is the actor that plays ragnar on vikings\nAnswer: Travis Fimmel\n\nwho said it's better to have loved and lost\nAnswer: Alfred , Lord Tennyson\n\nname the first indian woman to be crowned as miss world\nAnswer: Reita Faria\n\nBongo Botrako known as Amparanoia.\nyo la tengo theres a riot going on release date\n"
        if len(example['ctxs']) == 0:
            passages = ['']
        elif 'title' not in example['ctxs'][0] or example['ctxs'][0]['title'] == '':
            contexts = example['ctxs'][:self.n_context]
            passages = [example['ctxs'][0]['text']]
        else:
            contexts = example['ctxs'][:self.n_context]
            #passages = [c['text'] for c in contexts]
            # if 'title' in contexts[0]:passages = [c['title'] + ' ' + c['text'] for c in contexts]else:passages = [c['text'] for c in contexts]
            passages = [c['title'] + ' ' + c['text'] if 'title' in c else c['text'] for c in contexts]

        total_input = few_shot
        for p in passages:
            total_input += p + " "
        total_input += f"\n{question}\nAnswer: "
    
        
        model_inputs = self.tokenizer(total_input, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        #labels = self.tokenizer(target, max_length=128, truncation=True, padding='max_length')
        #model_inputs['labels'] = labels['input_ids']
        

        return model_inputs

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

    def append_question(self, example):
        if example['passages'] is None:
            return [example['question']]
        return [example['question'] + " " + t for t in example['passages']]

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 max_length=2048,
                 question_prefix='Question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.max_length = max_length
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']#self.question_prefix + " " + example['question']
        target = self.get_target(example)
        few_shot = "who won a million on deal or no deal\nAnswer: Tomorrow Rodriguez\n\nwho is the woman washing the car in cool hand luke\nAnswer: Joy Harmon\n\nwho is the actor that plays ragnar on vikings\nAnswer: Travis Fimmel\n\nwho said it's better to have loved and lost\nAnswer: Alfred , Lord Tennyson\n\nname the first indian woman to be crowned as miss world\nAnswer: Reita Faria\n\nBongo Botrako known as Amparanoia.\nyo la tengo theres a riot going on release date\n"
        
        if len(example['ctxs']) == 0:
            passages = ['']
        
        elif 'title' not in example['ctxs'][0] or example['ctxs'][0]['title'] == '':
            contexts = example['ctxs'][:self.n_context]
            passages = [example['ctxs'][0]['text']]
        
        else:
            contexts = example['ctxs'][:self.n_context]
            passages = [c['title'] + ' ' + c['text'] if 'title' in c else c['text'] for c in contexts]

        total_input = few_shot
        
        for p in passages:
            total_input += p + " "
        total_input += f"\n{question}\nAnswer: "
    
        return total_input

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

    def append_question(self, example):
        if example['passages'] is None:
            return [example['question']]
        return [example['question'] + " " + t for t in example['passages']]
