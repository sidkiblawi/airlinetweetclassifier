import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(filepath='./tweet_data/Tweets.csv'):
    df = pd.read_csv(filepath)
    texts = df['text'].tolist()
    labels = pd.factorize(df['airline_sentiment'])[0].tolist()

    train_end = int(len(texts) * 0.8)
    val_end = int(len(texts) * 0.9)
    return texts[:train_end], \
        labels[:train_end], \
        texts[train_end:val_end], \
        labels[train_end:val_end], \
        texts[val_end:], \
        labels[val_end:]
