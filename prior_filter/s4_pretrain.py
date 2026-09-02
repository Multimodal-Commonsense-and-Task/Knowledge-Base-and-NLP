
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
import argparse
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GPT2TokenizerFast

import pickle 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.utils.data.distributed import DistributedSampler
import lightning as L
import omegaconf

import dataloader
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import sys
import json
import numpy as np
import torch.nn.functional as F
import math
from collections import OrderedDict
from dataloader import get_dataloaders
from pytorch_lightning import Trainer
from pretrain_tool import Model

overrides = sys.argv[1:]
GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)

###########################
omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

original_cwd = os.getcwd()



#####################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

########## evaluation 관련




def demo_basic(rank, world_size, train_data, valid_data, trainer): # run()
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank) ##

    if rank==0:
        print(overrides)

    model = Model(config).to(rank)
    
    if config.init_from_checkpoint.bool:
      state_dict = torch.load(config.init_from_checkpoint.init_file)
      missing_keys, unexpected_keys = model.load_state_dict(state_dict['state_dict'] ,strict=False)
      if rank==0:
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
    
    
    model.trainer = trainer
    

    
    #####################

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    dataloader_train = DataLoader(train_data,batch_size=config.loader.batch_size,
      # num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      # persistent_workers=True,  
      sampler=train_sampler)
    
    valid_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank, shuffle=False, seed=config.seed) 
    dataloader_valid= DataLoader(valid_data,
                                 batch_size=config.loader.eval_batch_size,
      # num_workers=config.loader.num_workers,  
      pin_memory=config.loader.pin_memory,
      generator=None,
      sampler=valid_sampler)

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    for name, param in ddp_model.module.backbone.named_parameters():
      param.requires_grad_()
    param_list = []
    for name, param in ddp_model.module.backbone.named_parameters():
        if param.requires_grad:
            param_list.append(param)

    optimizer = torch.optim.AdamW(param_list,
      lr=config.optim.lr,
      betas=(config.optim.beta1,
             config.optim.beta2),
      eps=config.optim.eps,
      weight_decay=config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      config.lr_scheduler, optimizer=optimizer)

    tqdm_disable = False if rank==0 else True
    
    ####### if resume ##########
    epoch = 0
    global_step =0
    resume_epoch=0
    resume_global_step =0
    if 'resume_global_step' in config:
      resume_global_step = config.resume_global_step
    if 'resume_epoch' in config:
      resume_epoch = config.resume_epoch
    
    ###################################
    if resume_epoch ==0:
      with torch.no_grad():
        ddp_model.module.eval_step(rank, world_size, dataloader_valid, global_step, epoch, )
        dist.barrier()
        
    for epoch in range(1, 10):#args.max_epochs):  
        ddp_model.module.backbone.train()
        
        if epoch >= resume_epoch:

          optimizer.zero_grad()
          ddp_model.module.backbone.train()
          
          total_steps = len(dataloader_train)
          loss_list = []
          for step, batch_data in tqdm(enumerate(dataloader_train), disable=tqdm_disable, total=total_steps):
            if global_step >= resume_global_step:
              batch_data = {k: v.to(rank, non_blocking=True) for k, v in batch_data.items()}
              loss = ddp_model.module._compute_loss(batch_data)

              if torch.isnan(loss).any():
                  print("nan 있음")
              loss = loss / config.trainer.accumulate_grad_batches # default 1

              loss.backward()
              loss_list.append(loss.clone().detach())
              ddp_model.module.loss_metric.update(loss.detach(), torch.tensor(1.))
              if ((step + 1) % config.trainer.accumulate_grad_batches ==0) or (step + 1) == len(dataloader_train) :
                  optimizer.step()
                  scheduler.step()
                  optimizer.zero_grad()
                  dist.barrier()
                  global_step +=1
                  
                  logged_metrics = {}
                  current_lr = optimizer.param_groups[0]['lr']
                  logged_metrics['global_step'] = global_step
                  logged_metrics['learning_rate'] = current_lr
                  logged_metrics['epoch'] = epoch
                  with torch.no_grad():
                    loss_mean = (torch.tensor(loss_list).mean().item()  * config.trainer.accumulate_grad_batches )/ world_size
                  logged_metrics['train/loss'] = loss_mean
                  logged_metrics['train/loss_metric'] = ddp_model.module.loss_metric.compute()
                  loss_list=[]
                  ddp_model.module.loss_metric.reset()
                  ddp_model.module.trainer.logger.log_metrics(logged_metrics)
                  ddp_model.module.loss_metric.reset()
                    
                  dist.barrier()

                  if global_step % config.trainer.val_check_interval == 0:
                      with torch.no_grad():
                        ddp_model.module.eval_step(rank, world_size, dataloader_valid, global_step, epoch, )
                      

          if rank==0:
              print(f"{epoch} epoch / Loss : {loss.detach().item()}") ## loss.item() : loss 값
              

          dist.barrier()

      
          # ######eval#####
          ddp_model.module.backbone.eval()
          # torch.cuda.empty_cache()
          if rank==0:
              print("eval 시작")
              
          with torch.no_grad():
            ddp_model.module.eval_step(rank, world_size, dataloader_valid, global_step, epoch, )
           
                
          dist.barrier()
        
    cleanup()


def run_demo(demo_fn, world_size, train_data, valid_data,  trainer):
    mp.spawn(demo_fn,  # 
            args=(world_size, train_data, valid_data, trainer),
            nprocs=world_size,
            join=True)
    
 

if __name__ == "__main__":
    L.seed_everything(config.seed)
    n_gpus = torch.cuda.device_count() # 
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    os.makedirs(f'{config.output_dir}/{config.wandb.name}', exist_ok=True )
    
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
    trainer = Trainer(logger=wandb_logger, default_root_dir=os.getcwd())
    # trainer=None
    
    
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    train_ds, valid_ds = get_dataloaders(config, tokenizer) #
    
    filter_bool = torch.load(config.filter_file)
    print(f"!!!!! filter:  {config.filter_file}  !!!!!!!!")
    indices = filter_bool.nonzero().squeeze().tolist()
    train_ds = train_ds.select(indices)
      
      
    
    if 'valid_size' in config:
      print(f"!!!!!!valid size ->  {config.valid_size} !!!!!!")
      valid_ds = valid_ds.select(range(config.valid_size))
    

    with mp.Manager() as manager:
        learned_result = manager.dict()
        no_forget_result = manager.dict()
        loss_result = manager.dict()

        unchanged_ppl = manager.list()

        run_demo(demo_basic, world_size, train_ds, valid_ds,  trainer)

