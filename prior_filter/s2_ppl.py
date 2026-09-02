
import os
os.environ['TORCH_NCCL_TIMEOUT_MS'] = '18000000'  #30분에서 5시간으로
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning as L
import omegaconf

import dataloader
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import sys
import json
import transformers
import torch.nn.functional as F
from datetime import timedelta

overrides = sys.argv[1:]

GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="ppl", overrides=overrides)




rand_value = config.rand_value



#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ['TORCH_NCCL_TIMEOUT_MS'] = '18000000'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=7200000))

def cleanup():
    dist.destroy_process_group()

#################################


    
@torch.no_grad()
def calcul_ppl(model, tokenizer, samples, rank):
    max_length = 512
    attn_mask = torch.ones(samples.shape).to(rank)
    eval_context_size = samples.shape[-1]
    batch_size = samples.shape[0] 
    num_batches = samples.shape[0] // batch_size
    weight_value = torch.zeros(samples.size(0), dtype=torch.get_default_dtype(), device=rank)
    mean_value = torch.zeros(samples.size(0), dtype=torch.get_default_dtype(), device=rank)
    with torch.no_grad():
        logits = model(
            samples, attention_mask=attn_mask)[0]
        logits = logits.transpose(-1, -2)

        nlls = F.cross_entropy(logits[..., :-1],
                                samples[..., 1:],
                                reduction='none')
        first_eos = (samples == tokenizer.eos_token_id).cumsum(-1) == 1  # 전부 False임
        token_mask = (samples!= tokenizer.eos_token_id)
        weight= first_eos[..., 1:] + token_mask[..., 1:]
        mean_value += (nlls * weight).sum(dim=-1)
        weight_value += weight.sum(dim=-1)
        ppl =  torch.exp(mean_value / weight_value).cpu()
    return ppl  # [batch_size]
    
            
def demo_basic(rank, world_size, test_data, generated_result, generated_results_dict): # run()
    print(f"Running basic DDP example on rank {rank}.")
    
    if rank==0:
        print(f"""
model : {config.model_name}
batchsize : {config.batch_size}
output : ppls/{config.dataset}/
      """)

        if 'test_size' in config:
            print(f"restrict test size into {config.test_size}")
    
    
    setup(rank, world_size)
    torch.cuda.set_device(rank) ##


    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2-large')
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name).to(rank)

    data_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False) 
    dataloader_test = DataLoader(test_data, batch_size = config.batch_size,  sampler=data_sampler, drop_last=False)
    
    # ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    ddp_model = model
    ddp_model.eval()
    tqdm_disable = False if rank==0 else True

    #####
    
    dataset=[]
    total_steps = len(dataloader_test) 
    for step, batch_data in tqdm(enumerate(dataloader_test) , total=total_steps, disable=tqdm_disable):
        samples = batch_data['input_ids'].to(rank)
        with torch.no_grad():
            ppls = calcul_ppl(model, tokenizer, samples, rank)
        
        for i in range(batch_data['id'].size(0)):
            dic ={}
            dic['id'] = batch_data['id'][i].to('cpu').tolist()
            dic['ppl'] = ppls[i].to('cpu').tolist()
            dataset.append(dic)

    dist.barrier()
    torch.save(dataset, f"./tmp/{rand_value}_{rank}")
    cleanup()

"""
batch_data:
{'id':[], 'ppl':[]}
"""

def run_demo(demo_fn, world_size, test_data, generated_results, generated_results_dict):
    mp.spawn(demo_fn,  # demo_fn  이  5번파일 run() 과 같음
            args=(world_size, test_data, generated_results, generated_results_dict),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    
    print(overrides)
    
    
    def ready_data(example, idx):
        example["id"] = idx
            
        return example

    """Main entry point for training."""
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count() # 타이탄 서버는 3개
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    world_size = n_gpus
    
    ### ready for dataset
    ds = load_from_disk(config.dataset_path)
    ds.set_format(type="torch")
    if 'test_size' in config:
        ds = ds.select(range(config.test_size))
    test_data = ds.map(ready_data, with_indices=True)
    ########



    with mp.Manager() as manager:
        generated_results = manager.list()
        generated_results_dict = manager.dict()

        run_demo(demo_basic, world_size, test_data, generated_results, generated_results_dict)

        generated_results = []
        for i in range(world_size):
            shard = torch.load(f"tmp/{rand_value}_{i}")
            generated_results.extend(shard)


            os.remove(f"tmp/{rand_value}_{i}")

        print(len(generated_results))
        print(generated_results[0]['ppl'])

        dics = {}
        for line in generated_results:  #중복제거
            dics[line['id']] = line  
        sorted_values = [value['ppl'] for key, value in sorted(dics.items())]  # 정렬


    ppls = torch.tensor(sorted_values)
    torch.save(ppls, f"ppls/{config.dataset}.pt" )


