import re
import os
import sys
import time
import torch
import pickle
import logging
import numpy as np
import transformers

from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Union, Tuple, Dict, List, Optional

from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from alignment import (
    DataArguments,
    # DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from utils.preprocess import get_raw_datasets


logger = logging.getLogger(__name__)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    dataset_dir: Optional[str] = field(
        default="", metadata={"help": "the dataset name"}
    )
    n_toy: Optional[int] = field(
        default=-1, metadata={"help": "the number of datasets to be used"}
    )
    eval_split: Optional[str] = field(
        default="test", metadata={"help": "the dataset name"}
    )
    chat: bool = field(
        default=False, metadata={"help": "whether to use chat version or not"}
    )


def main():
    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()
    
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed for reproducibility
    set_seed(training_args.seed)


    # Load tokenizer and process datasets
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    # Load datasets
    raw_datasets = get_raw_datasets(script_args, data_args.dataset_name, tokenizer)

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision):
        # load the model, merge the adapter weights and unload the adapter
        # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
        logger.info(f"Merging peft adapters for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model, model_args.model_name_or_path, revision=model_args.model_revision
        )
        model.eval()
        model = model.merge_and_unload()
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    training_args.model_init_kwargs = model_kwargs
    training_args.ref_model_init_kwargs = ref_model_kwargs

    # Instantiate DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )


    # Training loop
    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    )
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info("*** Training complete ***")


    # Save model and create model card
    dpo_trainer.save_model(training_args.output_dir)
    
    # # Save everything else on main process
    # if accelerator.is_main_process:
    #     kwargs = {
    #         "finetuned_from": model_args.model_name_or_path,
    #         "dataset": list(data_args.dataset_mixer.keys()),
    #         "dataset_tags": list(data_args.dataset_mixer.keys()),
    #         "tags": ["alignment-handbook"],
    #     }
    #     dpo_trainer.create_model_card(**kwargs)
    #     # Restore k,v cache for fast inference
    #     dpo_trainer.model.config.use_cache = True
    #     dpo_trainer.model.config.save_pretrained(training_args.output_dir)
    #     if training_args.push_to_hub is True:
    #         dpo_trainer.push_to_hub()

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")




if __name__ == "__main__":
    main()
