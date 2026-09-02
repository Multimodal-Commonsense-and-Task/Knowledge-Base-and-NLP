from dataloader import get_dataset
import transformers
from transformers import logging
logging.set_verbosity_error() 

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2-large')
dataset = get_dataset(dataset_name='dolma_flatten-valid',
            tokenizer =tokenizer,
            mode='train',
            streaming=False,
            wrap=True,
            cache_dir='dataset/dolma/dolma_flatten-valid',
            block_size=512,
            )
dataset = get_dataset(dataset_name='dolma_flatten-train',
            tokenizer =tokenizer,
            mode='train',
            streaming=False,
            wrap=True,
            cache_dir='dataset/dolma/dolma_flatten-train',
            block_size=512,
            )