"""Refiner implementation for abstractive compression."""

import torch
from typing import List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
from ...base import BaseCompressor, SearchResult

class RefinerCompressor(BaseCompressor):
    """Refiner: Abstractive document compression."""
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-chat-hf",
        adapter: str = "al1231/Refiner-7B",
        device: str = "cuda",
        max_tokens: int = 3500,
        cache_dir: str = "./cache"
    ):
        """Initialize Refiner compressor.
        
        Args:
            base_model: Base model path
            adapter: Path to trained adapter
            device: Device to use
            max_tokens: Maximum tokens for context
            cache_dir: Cache directory for models
        """
        self.device = device
        self.max_tokens = max_tokens
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Load adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter,
            is_trainable=False
        )
        self.model.eval()
        
        # Prompt template
        self.template = (
            "[INST]<<SYS>>"
            "[MONITOR]{context}"
            "<</SYS>>{question}[/INST] "
        )
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum tokens."""
        tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False
        )
        if len(tokens) <= self.max_tokens:
            return context
            
        truncated_tokens = tokens[:self.max_tokens]
        return self.tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=True
        )
    
    def _prepare_input(self, query: str, context: str) -> dict:
        """Prepare model input."""
        context = self._truncate_context(context)
        prompt = self.template.format(
            question=query,
            context=context
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        return {
            k: v.to(self.device)
            for k, v in inputs.items()
        }
    
    def _generate(self, inputs: dict) -> List[str]:
        """Generate compressed output."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                top_p=1,
                temperature=None,
                do_sample=False,
                max_new_tokens=8096,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True
            )
        
        # Extract generated tokens
        pred_tokens = outputs.sequences[
            :,
            inputs['input_ids'].shape[1]:
        ]
        return self.tokenizer.batch_decode(
            pred_tokens,
            skip_special_tokens=True
        )
    
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using Refiner."""
        # Combine documents into context
        context = "\n".join(
            f"## {doc.title}\n{doc.text}"
            for doc in documents
            if doc.text.strip()
        )
        
        # Prepare model input
        inputs = self._prepare_input(query, context)
        
        # Generate compressed text
        outputs = self._generate(inputs)
        compressed_text = outputs[0].strip()
        
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=compressed_text,
            score=1.0
        )]