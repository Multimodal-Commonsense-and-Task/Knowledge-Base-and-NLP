# config 불러오기
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
import omegaconf
import torch
import dataloader
from tqdm import tqdm
import transformers
import datasets
import sys

overrides = sys.argv[1:]



omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)  
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

GlobalHydra.instance().clear()
initialize(config_path="./configs/", version_base=None)
config = compose(config_name="config", overrides=overrides)



config.data.tokenizer_name_or_path ='gpt2-large'

config.data.wrap= True
config.data.streaming= False


tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2-large')
print(tokenizer)

train_ds = datasets.load_from_disk(config.data_path).with_format('torch')

train_loader = torch.utils.data.DataLoader(train_ds, 32, shuffle=False)

print(train_ds)

data_name = config.data.train.split('-')[0]
print(data_name)

pallet = torch.zeros((2, tokenizer.vocab.__len__()))
i=0
for data0 in tqdm(train_ds):
    x = data0['input_ids']
    pallet[0] += torch.cat((torch.bincount(x), torch.zeros((  pallet.size(1) - torch.bincount(x).size(0) ))))
    pallet[1,x] +=1
    
mul0 = pallet[0] * pallet[1]
sum0 = mul0.sum()
prob0 = mul0 / sum0
torch.save(prob0.cpu(),f'prior_filter/{data_name}/all_prior_tfdf.pt')


mul0 = pallet[0] 
sum0 = mul0.sum()
prob0_1 = mul0 / sum0
torch.save(prob0_1.cpu(),f'prior_filter/{data_name}/all_prior_tf.pt')


mul0 = pallet[1] 
sum0 = mul0.sum()
prob0_1 = mul0 / sum0
torch.save(prob0_1.cpu(),f'prior_filter/{data_name}/all_prior_df.pt')


#### assess   mu_d
logs=[]
for data in tqdm(train_loader):
    a = prob0[data['input_ids']].log().sum(dim=-1)
    logs.extend(a)
    # break
logs = torch.stack(logs) / data['input_ids'].size(-1)
torch.save(logs,f'prior_filter/{data_name}/{config.data.train}/all_data_means.pt')

### assess   sigma_d
train_loader = torch.utils.data.DataLoader(train_ds, 32, shuffle=False)
stds=[]
for data in tqdm(train_loader):
    a = (prob0[data['input_ids']] * 1000).std(dim=-1)
    stds.extend(a)
    # break
stds = torch.stack(stds)
torch.save(stds,f'prior_filter/{data_name}/{config.data.train}/all_data_stds.pt')

