"""LongLLMLingua implementation for token-level compression."""

from typing import List, Optional
from llmlingua import PromptCompressor
from ...base import BaseCompressor, SearchResult

class LongLLMLinguaCompressor(BaseCompressor):
    """LongLLMLingua: Dynamic token-level compression."""
    
    def __init__(
        self,
        device_map: str = "auto",
        cache_dir: str = "./cache",
        compression_ratio: float = 0.4,
        context_budget: str = "+100",
        compression_rate: float = 0.2,
        use_sentence_level: bool = False,
        reorder_context: str = "sort"
    ):
        """Initialize LongLLMLingua compressor.
        
        Args:
            device_map: Device mapping for model
            cache_dir: Cache directory for models
            compression_ratio: Dynamic compression ratio
            context_budget: Additional context budget
            compression_rate: Base compression rate
            use_sentence_level: Whether to use sentence-level filtering
            reorder_context: Context reordering strategy
        """
        self.device_map = device_map
        self.compression_ratio = compression_ratio
        self.context_budget = context_budget
        self.compression_rate = compression_rate
        self.use_sentence_level = use_sentence_level
        self.reorder_context = reorder_context
        
        # Initialize LLMLingua
        self.compressor = PromptCompressor(
            device_map=device_map,
            cache_dir=cache_dir
        )
        
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using LongLLMLingua.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        # Prepare documents
        prompt_list = [
            f"{doc.title}\n{doc.text}"
            for doc in documents
        ]
        
        # Compress using LLMLingua
        result = self.compressor.compress_prompt(
            prompt_list,
            instruction=(
                "Given the context information and not prior knowledge, "
                "answer the query. Do not provide any explanation."
            ),
            question=query,
            rate=self.compression_rate,
            condition_compare=True,
            condition_in_question="after",
            rank_method="longllmlingua",
            use_sentence_level_filter=self.use_sentence_level,
            context_budget=self.context_budget,
            dynamic_context_compression_ratio=self.compression_ratio,
            reorder_context=self.reorder_context
        )
        
        # Return compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=result["compressed_prompt"],
            score=1.0
        )]
    
    def set_compression_ratio(self, ratio: float) -> None:
        """Update dynamic compression ratio."""
        self.compression_ratio = ratio
    
    def set_context_budget(self, budget: str) -> None:
        """Update context budget."""
        self.context_budget = budget
    
    def set_compression_rate(self, rate: float) -> None:
        """Update base compression rate."""
        self.compression_rate = rate