"""EXIT implementation for context-aware extractive compression."""

import torch
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.cuda.amp import autocast
import numpy as np
from functools import lru_cache
from ...base import BaseCompressor, SearchResult

class EXITCompressor(BaseCompressor):
    """EXIT: Context-aware extractive compression."""
    
    def __init__(
        self,
        base_model: str = "google/gemma-2b-it",
        checkpoint: str = None,
        device: str = None,
        cache_dir: str = "./cache",
        batch_size: int = 8,
        threshold: float = 0.5
    ):
        """Initialize EXIT compressor.
        
        Args:
            base_model: Base model path
            checkpoint: Path to trained checkpoint
            device: Device to use (None for auto)
            cache_dir: Cache directory for models
            batch_size: Batch size for processing
            threshold: Confidence threshold for selection
        """
        self.batch_size = batch_size
        self.threshold = threshold
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model
        model_kwargs = {
            "device_map": "auto" if device is None else device,
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
            "max_length": 4096,
        }
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )
        
        if checkpoint:
            self.peft_config = PeftConfig.from_pretrained(checkpoint)
            self.model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint
            )
        else:
            self.model = self.base_model
            
        # Prepare model
        self.model.eval()
        if hasattr(self.model, 'half'):
            self.model.half()
            
        # Cache device and token IDs
        self.device = next(self.model.parameters()).device
        self.yes_token_id = self.tokenizer.encode(
            "Yes",
            add_special_tokens=False
        )[0]
        self.no_token_id = self.tokenizer.encode(
            "No",
            add_special_tokens=False
        )[0]
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    @lru_cache(maxsize=1024)
    def _generate_prompt(
        self,
        query: str,
        context: str,
        sentence: str
    ) -> str:
        """Generate prompt for relevance classification."""
        return (
            f'<start_of_turn>user\n'
            f'Query:\n{query}\n'
            f'Full context:\n{context}\n'
            f'Sentence:\n{sentence}\n'
            f'Is this sentence useful in answering the query? '
            f'Answer only "Yes" or "No".<end_of_turn>\n'
            f'<start_of_turn>model\n'
        )
    
    def _predict_batch(
        self,
        queries: List[str],
        contexts: List[str],
        sentences: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict relevance for a batch of sentences."""
        prompts = [
            self._generate_prompt(query, context, sentence)
            for query, context, sentence
            in zip(queries, contexts, sentences)
        ]
        
        with torch.cuda.amp.autocast():
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
                return_attention_mask=True
            )
            
            inputs = {
                k: v.to(self.device, non_blocking=True)
                for k, v in inputs.items()
            }
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = self.model(**inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                relevant_logits = torch.stack([
                    next_token_logits[:, self.yes_token_id],
                    next_token_logits[:, self.no_token_id]
                ], dim=1)
                
                probs = torch.softmax(relevant_logits, dim=1)
                predictions = [
                    "Yes" if p else "No"
                    for p in probs.argmax(dim=1).cpu().numpy()
                ]
        
        return predictions, probs
    
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using context-aware extraction.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        # Prepare full context
        context = "\n".join(
            f"{doc.title}\n{doc.text}"
            for doc in documents
        )
        
        selected_texts = []
        current_doc_id = None
        current_texts = []
        
        # Process each document while maintaining order
        for doc in documents:
            # Start new document
            if current_doc_id != doc.evi_id:
                if current_texts:
                    doc_text = " ".join(current_texts)
                    if doc_text.strip():
                        selected_texts.append(doc_text)
                current_doc_id = doc.evi_id
                current_texts = []
            
            # Get predictions for current document
            predictions, probs = self._predict_batch(
                [query],
                [context],
                [doc.text]
            )
            
            # Add text if above threshold
            if probs[0, 0].item() >= self.threshold:
                current_texts.append(doc.text)
        
        # Add last document if exists
        if current_texts:
            doc_text = " ".join(current_texts)
            if doc_text.strip():
                selected_texts.append(doc_text)
        
        # Combine all selected texts
        compressed_text = "\n\n".join(selected_texts)
        
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=compressed_text,
            score=1.0
        )]