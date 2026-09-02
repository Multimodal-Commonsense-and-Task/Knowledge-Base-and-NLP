import logging
import os
from tqdm import tqdm
import numpy as np
import json

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from src.options import Options
from src import data, slurm, dist_utils, utils, contriever, finetuning_data, inbatch

logger = logging.getLogger(__name__)


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def main():

    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    # 1) load model
    retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    opt.retriever_model_id = retriever_model_id
    # set device to cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
    # load weight from pth file
    if opt.weight_path is not None:
        # fail loudly: silently scoring with an untrained encoder looks like a bad
        # checkpoint rather than a missing one
        if not os.path.exists(opt.weight_path):
            raise FileNotFoundError(f"compressor checkpoint not found: {opt.weight_path}")
        model = AutoModel.from_pretrained(opt.model_path).to(device)
        state_dict = torch.load(opt.weight_path)['model']
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
    else:
        logger.warning("no --weight_path given, scoring with the untrained encoder")
        model = AutoModel.from_pretrained(opt.model_path).to(device)
        optimizer, scheduler = utils.set_optim(opt, model)

    logger.info(utils.get_parameters(model))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
    # 2) load dataset
    dataset = finetuning_data.PredictDataset(
        datapaths=opt.eval_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
    )
    # 3) predict score
    model.eval()
    score_list = []
    #device = torch.device('cuda:{}'.format(opt.local_rank))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            for question, context in zip(batch['question'], batch['context']):
                inputs = tokenizer([question, context], padding=True, truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                score = (embeddings[0] @ embeddings[1]).item()
                #sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
                score_list.append(score)
    score_path = os.path.join(opt.output_score_path, 'score.json')
    score_dict = {'score':score_list}
    os.makedirs(opt.output_score_path, exist_ok=True)
    with open(score_path, 'w') as f:
        json.dump(score_dict, f)
            


if __name__ == "__main__":
    main()