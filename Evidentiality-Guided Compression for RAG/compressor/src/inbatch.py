# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np
import math
import random
import transformers
import logging
import torch.distributed as dist

from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)


class InBatch(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats


class HNInBatch(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(HNInBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, k_tokens_for_hn, k_mask_for_hn, stats_prefix="", iter_stats={}, **kwargs):
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        kemb_for_hn = self.encoder(input_ids=k_tokens_for_hn, attention_mask=k_mask_for_hn, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)
        gather_kemb_for_hn = gather_fn(kemb_for_hn)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
        scores_for_hn = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb_for_hn)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        loss_for_hn = torch.nn.functional.cross_entropy(scores_for_hn, labels, label_smoothing=self.label_smoothing)
        loss = loss + loss_for_hn

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats

class HNInBatchClass(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(HNInBatchClass, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        self.binary_classifier = torch.nn.Linear(768*2, 1)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, k_tokens_for_hn, k_mask_for_hn, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        kemb_for_hn = self.encoder(input_ids=k_tokens_for_hn, attention_mask=k_mask_for_hn, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)
        gather_kemb_for_hn = gather_fn(kemb_for_hn)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
        scores_for_hn = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb_for_hn)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        loss_for_hn = torch.nn.functional.cross_entropy(scores_for_hn, labels, label_smoothing=self.label_smoothing)
        loss = loss + loss_for_hn
        binary_labels = torch.zeros(qemb.size(0)*kemb.size(0), dtype=torch.long).cuda()
        binary_logits = torch.zeros(qemb.size(0)*kemb.size(0), dtype=torch.float).cuda()
        for i in range(scores.size(0)):
            for j in range(scores.size(1)):
                binary_labels[i*scores.size(0)+j] = 1 if i == j else 0
                binary_logits[i*scores.size(0)+j] = scores[i, j]
        binary_loss = torch.nn.BCEWithLogitsLoss()(binary_logits.view(-1, 1), binary_labels.float().view(-1, 1))
        #binary_loss = torch.nn.functional.cross_entropy(binary_logits.view(-1, 1), binary_labels, label_smoothing=self.label_smoothing)
        loss = loss + binary_loss

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats


class PairInBatch(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(PairInBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, s_labels_org, stats_prefix="", iter_stats={}, **kwargs):
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        # pairwise logistic loss
        pairwise_loss = 0.0
        s_labels = torch.zeros_like(scores, dtype=torch.float32)

        '''for i in range(scores.size(0)):
            s_labels[i, i] = 1.0
            temp_tensor = torch.tensor(s_labels_org[bsz+i*(bsz-1):bsz+(i+1)*(bsz-1)], dtype=torch.float).cuda()
            s_labels[i, bsz+i*(bsz-1):bsz+(i+1)*(bsz-1)] = temp_tensor
            for j in range(scores.size(1)):
                for k in range(scores.size(1)):
                    if s_labels[i, j] > s_labels[i, k]:
                        pairwise_loss += torch.log(1 + torch.exp(-scores[i, j] + scores[i, k]))

        num_comparisons = scores.size(0) * scores.size(1) * (scores.size(1) - 1) / 2'''
        
        for i in range(scores.size(0)):
            s_labels[i, i] = 1.0
            temp_tensor = torch.tensor(s_labels_org[bsz+i*(bsz-1):bsz+(i+1)*(bsz-1)], dtype=torch.float).cuda()
            s_labels[i, bsz+i*(bsz-1):bsz+(i+1)*(bsz-1)] = temp_tensor
            for j in range(scores.size(1)):
                for k in range(scores.size(1)):
                    if i == j or (j >= bsz+i*(bsz-1) and j < bsz+(i+1)*(bsz-1)) or (k >= bsz+i*(bsz-1) and k < bsz+(i+1)*(bsz-1)) or k == i:
                        if s_labels[i, j] > s_labels[i, k]:
                            pairwise_loss += torch.log(1 + torch.exp(-scores[i, j] + scores[i, k]))
                    else:
                        continue
        num_comparisons = scores.size(0) * (scores.size(0) - 1)

        pairwise_loss /= num_comparisons
        loss += pairwise_loss

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
