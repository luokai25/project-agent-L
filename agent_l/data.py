"""
Data utilities for Agent L.

Provides:
- Dataset utilities for language modeling
- Data loading and preprocessing
- Text chunking and tokenization
- Streaming data support
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Iterator, Callable, Any
from dataclasses import dataclass
import random


@dataclass
class TextDatasetConfig:
    """Configuration for text datasets."""
    max_seq_len: int = 2048
    stride: int = 1024  # Stride for overlapping chunks
    add_bos: bool = True
    add_eos: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


class TextDataset(Dataset):
    """
    Dataset for text files.
    
    Reads text files, tokenizes them, and creates overlapping chunks.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer_fn: Callable[[str], List[int]],
        config: TextDatasetConfig,
    ):
        self.config = config
        self.chunks = []
        
        # Read and tokenize file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        tokens = tokenizer_fn(text)
        
        # Create overlapping chunks
        stride = config.stride
        max_len = config.max_seq_len
        
        for i in range(0, len(tokens) - max_len + 1, stride):
            chunk = tokens[i:i + max_len]
            
            # Add BOS/EOS
            if config.add_bos:
                chunk = [config.bos_token_id] + chunk
            if config.add_eos:
                chunk = chunk + [config.eos_token_id]
            
            # Pad if needed
            if len(chunk) < max_len + 2:
                chunk = chunk + [config.pad_token_id] * (max_len + 2 - len(chunk))
            
            self.chunks.append(chunk[:max_len])
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


class JSONLDataset(Dataset):
    """
    Dataset for JSONL files (one JSON object per line).
    
    Each line should have 'text' field or be configurable.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer_fn: Callable[[str], List[int]],
        config: TextDatasetConfig,
        text_field: str = "text",
    ):
        self.config = config
        self.chunks = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get(text_field, "")
                tokens = tokenizer_fn(text)
                
                # Chunk if needed
                for i in range(0, len(tokens), config.stride):
                    chunk = tokens[i:i + config.max_seq_len]
                    if len(chunk) < config.max_seq_len // 2:
                        continue  # Skip very short chunks
                    
                    if config.add_bos:
                        chunk = [config.bos_token_id] + chunk
                    if config.add_eos:
                        chunk = chunk + [config.eos_token_id]
                    
                    chunk = chunk[:config.max_seq_len]
                    self.chunks.append(chunk)
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int):
        chunk = self.chunks[idx]
        # Pad
        if len(chunk) < self.config.max_seq_len:
            chunk = chunk + [self.config.pad_token_id] * (self.config.max_seq_len - len(chunk))
        
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large text files.
    
    Yields chunks on-the-fly without loading entire file.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer_fn: Callable[[str], List[int]],
        config: TextDatasetConfig,
        buffer_size: int = 10000,
    ):
        self.file_path = file_path
        self.tokenizer_fn = tokenizer_fn
        self.config = config
        self.buffer_size = buffer_size
    
    def __iter__(self):
        buffer = []
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer_fn(line)
                buffer.extend(tokens)
                
                while len(buffer) >= self.config.max_seq_len:
                    chunk = buffer[:self.config.max_seq_len]
                    buffer = buffer[self.config.stride:]
                    
                    if self.config.add_bos:
                        chunk = [self.config.bos_token_id] + chunk[:-1]
                    if self.config.add_eos:
                        chunk = chunk[:-1] + [self.config.eos_token_id]
                    
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults for language modeling.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def collate_fn(batch):
    """
    Collate function for variable-length sequences.
    
    Pads sequences to the same length.
    """
    inputs, targets = zip(*batch)
    
    # Pad to max length in batch
    max_len = max(len(seq) for seq in inputs)
    
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tgt)] = tgt
    
    return padded_inputs, padded_targets


__all__ = [
    "TextDatasetConfig",
    "TextDataset",
    "JSONLDataset",
    "StreamingTextDataset",
    "create_dataloader",
    "collate_fn",
]
