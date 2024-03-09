from utils import word_embeddings, df, Vocab
import torch
from torch.utils.data import DataLoader

PAD_INDEX = Vocab.word2id["<pad>"]

reviews = df["vi_review"].values[:20000]
sentiments = df["sentiment"].values[:20000]
sentiment2id = {"positive": 1, "negative": 0}

def load_dataset(data_reviews, data_sentiments):
  data_infor = list()
  data_reviews = sorted(data_reviews, key = lambda x : len(Vocab.corpus_to_tensor(x)), reverse = True)

  reviews_length = [len(Vocab.corpus_to_tensor(review)) for review in data_reviews]
  our_sentiments = [sentiment2id[sentiments[id]] for id in range(len(data_sentiments))]

  reviews_tensor = [torch.tensor(Vocab.corpus_to_tensor(review)) for review in data_reviews]
  reviews_tensor = nn.utils.rnn.pad_sequence(reviews_tensor,
                                             batch_first = True,
                                             padding_value = PAD_INDEX)

  data_infor = [{"review": (reviews_tensor[id], torch.tensor(reviews_length[id])), \
                 "sentiment": torch.tensor(our_sentiments[id])} \
                                          for id in range(len(data_reviews))]
  return data_infor

train_reviews, train_sentiments = reviews[:16000], sentiments[:16000]
test_reviews, test_sentiments = reviews[16000: 18000], sentiments[16000: 18000]
val_reviews, val_sentiments = reviews[18000: 20000], sentiments[18000: 20000]

train_dataset = load_dataset(train_reviews, train_sentiments)
test_dataset = load_dataset(test_reviews, test_sentiments)
val_dataset = load_dataset(val_reviews, val_sentiments)

BATCH_SIZE = 100
train_loader = DataLoader(train_dataset,
                          batch_size = BATCH_SIZE,
                          shuffle = False)

test_loader = DataLoader(test_dataset,
                         batch_size = BATCH_SIZE,
                         shuffle = False)

val_loader = DataLoader(val_dataset,
                        batch_size = BATCH_SIZE,
                        shuffle = False)