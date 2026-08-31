import os
import time
import torch

from torch.utils.data import Dataset
from typing import Union, Any, Dict, List


class ListwiseRerankDataset(Dataset):
    def __init__(self, queries: List[str], documents: List[List[Dict[str, str]]]):
        assert len(queries) == len(documents), f"{len(queries)}; {len(documents)}"
        self.queries = queries
        self.documents = documents

    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "docs": self.documents[idx],
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    queries = [item["query"] for item in batch]
    docs = [item["docs"] for item in batch]

    return {
        "query": queries,
        "docs": docs,
    }