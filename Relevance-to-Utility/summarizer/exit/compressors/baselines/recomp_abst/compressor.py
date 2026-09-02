"""RECOMP Abstractive implementation for document compression."""

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ...base import BaseCompressor, SearchResult

class RecompAbstractiveCompressor(BaseCompressor):
    """RECOMP: Abstractive document compression using seq2seq models."""
    
    def __init__(
        self,
        model_name: str = "fangyuan/nq_abstractive_compressor",
        cache_dir: str = "./cache",
        max_length: int = 8192,
        device: str = None
    ):
        """Initialize RECOMP Abstractive compressor.
        
        Args:
            model_name: Model identifier from HuggingFace
            cache_dir: Directory for model caching
            max_length: Maximum sequence length
            device: Device to run model on (None for auto)
        """
        self.max_length = max_length
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        
        self.model.eval()

    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using abstractive summarization.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        # Combine all documents into single passage
        passage = "\n".join([doc.text for doc in documents])
        
        # Create prompt
        prompt = f"Question: {query}\n Document: {passage}\n Summary: "
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=300,
                early_stopping=True,
                num_beams=5,
                length_penalty=2.0
            )
            
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
        
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=summary,
            score=1.0
        )]