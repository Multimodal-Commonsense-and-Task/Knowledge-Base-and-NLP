import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint, pickle_dump, pickle_load


def load_model(args, do_print=True):
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation,
                                      qidf=pickle_load(args.qidf) if args.qidf is not None else None,
                                      )
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint
