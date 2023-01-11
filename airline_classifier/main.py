from tweet_data.utils import split_dataset
from data.airline_tweet_dataset import CSVDataset
from models.classifiers import TweetClassifier
from train.trainer import Trainer
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup


train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset('../tweet_data/Tweets.csv')


model = TweetClassifier(AutoModelForSequenceClassification.from_pretrained('bert-base-uncased'))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

train_dataset = CSVDataset(train_data,train_labels,tokenizer)
val_dataset = CSVDataset(val_data,val_labels,tokenizer)
test_dataset = CSVDataset(test_data,test_labels,tokenizer)

trainer = Trainer(model)
trainer.fit(train_dataset,val_dataset,test_dataset)

