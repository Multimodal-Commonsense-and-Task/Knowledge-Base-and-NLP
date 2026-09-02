# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import random
import json
import sys
import numpy as np
from src import normalize_text


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            gold = random.choice(example["positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []

        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]

        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        return batch

class PredictDataset(Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
        topk=1,
    ):
        super().__init__(datapaths, negative_ctxs, negative_hard_ratio, negative_hard_min_idx, training, global_rank, world_size, maxload, normalize)
        self._load_data(datapaths, global_rank, world_size, maxload)
        self.topk = topk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        ctxs = []
        for ctx in example['ctxs'][:self.topk]:
            if 'title' in ctx:
                ctxs.append(ctx['title'] + ' ' + ctx['text'])
            else:
                ctxs.append(ctx['text'])
        

        example = {
            "question": self.normalize_fn(question),
            "context": [self.normalize_fn(ctx) for ctx in ctxs][0],
        }
        return example


class HNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            gold = random.choice(example["positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
            # loss for hard neg
            negatives_for_hn = []
            if len(example["hard_negative_ctxs"]) > 0:
                hard_negative = random.choice(example["hard_negative_ctxs"])
            else:
                hard_negative = random.choice(example["positive_ctxs"])
            random_hard_negatives_for_hn = random.sample(example["negative_ctxs"], n_hard_negatives+n_random_negatives)
            negatives_for_hn += random_hard_negatives_for_hn
                
        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []

        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]

        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]

        hard_negative = hard_negative["title"] + " " + hard_negative["text"] if "title" in hard_negative and len(hard_negative["title"]) > 0 else hard_negative["text"]
        negatives_for_hn = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives_for_hn
        ]


        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
            "hard_negative": self.normalize_fn(hard_negative),
            "negatives_for_hn": [self.normalize_fn(n) for n in negatives_for_hn],
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives

class HNCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        # loss for hard neg
        hard_negatives = [ex["hard_negative"] for ex in batch]
        negatives_for_hn = [item for ex in batch for item in ex["negatives_for_hn"]]
        allpassages_for_hn = hard_negatives + negatives_for_hn
        kout_for_hn = self.tokenizer.batch_encode_plus(
            allpassages_for_hn,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        k_tokens_for_hn, k_mask_for_hn = kout_for_hn["input_ids"], kout_for_hn["attention_mask"].bool()
        g_tokens_for_hn, g_mask_for_hn = k_tokens_for_hn[: len(hard_negatives)], k_mask_for_hn[: len(hard_negatives)]
        n_tokens_for_hn, n_mask_for_hn = k_tokens_for_hn[len(hard_negatives) :], k_mask_for_hn[len(hard_negatives) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "k_tokens_for_hn": k_tokens_for_hn,
            "k_mask_for_hn": k_mask_for_hn,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
            "g_tokens_for_hn": g_tokens_for_hn,
            "g_mask_for_hn": g_mask_for_hn,
            "n_tokens_for_hn": n_tokens_for_hn,
            "n_mask_for_hn": n_mask_for_hn,
        }

        return batch


class PairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            gold = random.choice(example["positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []

        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]

        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]
        s_labels_neg = torch.zeros(n_hard_negatives+n_random_negatives, dtype=torch.float32)
        s_labels_neg[:n_hard_negatives] = 0.5
        s_labels_pos = torch.ones(1, dtype=torch.float32)
        #s_labels[0] = 1.0

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
            "s_labels_pos": s_labels_pos,
            "s_labels_neg": s_labels_neg,
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives


class PairCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]] # 
        s_labels_pos = [ex["s_labels_pos"].item() for ex in batch]
        s_labels_neg = [item.item() for ex in batch for item in ex["s_labels_neg"]]
        allpassages = golds + negs
        s_labels_org = s_labels_pos + s_labels_neg
        

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "s_labels_org": s_labels_org,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        return batch