from abc import ABC

import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset, ABC):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode_plus(text,
                                             add_special_tokens=True,
                                             max_length=self.max_len,
                                             padding='max_length',
                                             return_attention_mask=True,
                                             return_tensors='pt',
                                             )

        return encoded['input_ids'], encoded['attention_mask'], label
