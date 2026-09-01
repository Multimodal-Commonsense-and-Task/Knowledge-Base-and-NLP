#!/usr/bin/env python3
"""
Training script for EXIT model.
This script handles the full training pipeline including model setup,
training configuration, and experiment tracking.
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Optional, List
import torch
import wandb
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_id: str = "google/gemma-2b-it"
    train_dataset: str = None
    test_dataset: str = None
    output_dir: str = None
    cache_dir: str = "./cache"
    resume_from_checkpoint: Optional[str] = None
    
    # LoRA parameters
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training parameters
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_steps: int = -1
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    
    # Evaluation parameters
    eval_steps: int = 300
    save_steps: int = 300
    save_total_limit: int = 3
    
    # Experiment tracking
    wandb_project: Optional[str] = None
    experiment_name: Optional[str] = None

class MetricCallback(TrainerCallback):
    """Callback for logging metrics to wandb."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
            
        if wandb.run is not None:
            wandb.log({
                "eval/loss": metrics.get("eval_loss", 0),
                "eval/step": state.global_step,
            })

def parse_args() -> TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EXIT model")
    
    # Model and data arguments
    parser.add_argument("--model_id", type=str, default="google/gemma-2b-it")
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--resume_from_checkpoint", type=str)
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    
    # Evaluation arguments
    parser.add_argument("--eval_steps", type=int, default=300)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    # Experiment tracking
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--experiment_name", type=str)
    
    args = parser.parse_args()
    return TrainingConfig(**vars(args))

def setup_experiment_tracking(config: TrainingConfig) -> None:
    """Setup wandb experiment tracking."""
    if config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.experiment_name,
            config=vars(config)
        )
        logger.info(f"Initialized wandb project: {config.wandb_project}")

def setup_tokenizer(config: TrainingConfig) -> AutoTokenizer:
    """Setup and configure tokenizer."""
    logger.info(f"Loading tokenizer: {config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        add_eos_token=True,
        cache_dir=config.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def find_all_linear_names(model: AutoModelForCausalLM) -> List[str]:
    """Find all linear layer names for LoRA configuration."""
    cls = torch.nn.Linear
    names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            splits = name.split(".")
            names.add(splits[0] if len(splits) == 1 else splits[-1])
            
    if "lm_head" in names:
        names.remove("lm_head")
        
    return list(names)

def setup_model(config: TrainingConfig) -> tuple:
    """Setup and configure model with LoRA."""
    logger.info(f"Loading model: {config.model_id}")
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    target_modules = find_all_linear_names(model)
    logger.info(f"Target modules for LoRA: {target_modules}")
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log parameter counts
    trainable_params, total_params = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"Total parameters: {total_params:,}")
    
    return model, lora_config

def train(config: TrainingConfig) -> None:
    """Main training function."""
    try:
        # Setup experiment tracking
        setup_experiment_tracking(config)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup tokenizer and model
        tokenizer = setup_tokenizer(config)
        model, lora_config = setup_model(config)
        
        # Load datasets
        logger.info("Loading datasets")
        train_data = load_from_disk(config.train_dataset)
        test_data = load_from_disk(config.test_dataset)
        
        # Setup training arguments
        training_args = SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_ratio=config.warmup_ratio,
            max_steps=config.max_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            fp16=True,
            logging_steps=10,
            output_dir=config.output_dir,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            eval_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            weight_decay=config.weight_decay,
            report_to="wandb" if config.wandb_project else None
        )
        
        # Setup trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=test_data,
            dataset_text_field="prompt",
            max_seq_length=8192,
            peft_config=lora_config,
            data_collator=DataCollatorForCompletionOnlyLM(
                tokenizer=tokenizer,
                mlm=False,
                response_template="<start_of_turn>model\n"
            ),
            args=training_args,
            callbacks=[MetricCallback()]
        )
        
        # Start training
        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        
        # Save final model
        final_model_path = os.path.join(config.output_dir, "final_model")
        trainer.model.save_pretrained(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    config = parse_args()
    train(config)