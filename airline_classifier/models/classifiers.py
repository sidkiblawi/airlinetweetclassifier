import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import FlaxAutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


class TweetClassifier(torch.nn.Module):
    def __init__(self, model):
        super(TweetClassifier, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]
