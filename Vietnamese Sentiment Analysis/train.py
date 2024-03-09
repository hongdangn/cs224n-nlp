from model import BiLSTM
from load_data import train_loader, test_loader, val_loader
from utils import Vocab, word_embeddings
import torch
import torch.nn as nn
import torch.optim as optim


model = BiLSTM(
    vocab_size = len(Vocab.word2id),
    embed_size = 100,
    hidden_size = 256,
    n_layers = 1,
    bidirectional = True,
    dropout = 0.2,
    pad_idx = Vocab.word2id["<pad>"]
)

def count_parameters(model):
    count = 0
    for p in model.parameters():
      if p.requires_grad:
        p_count = 1
        for dim in p.size():
          p_count *= dim
        count += p_count
    return count
print("the model has {} parameters".format(count_parameters(model)))

model.embedding.weight.data[0] = torch.zeros(100)
model.embedding.weight.data[1] = torch.zeros(100)
for id in range(len(word_embeddings.vectors)):
  model.embedding.weight.data[id + 2] = word_embeddings.vectors[id]

def get_accuracy(pred, sentiments):
  prediction = pred.argmax(dim = -1)
  prediction = [True if prediction[id] == sentiments[id] else False for id in range(len(prediction))]
  return np.sum(prediction) / len(pred)

def train(dataloader, optimizer, criterion, model):
  history_loss, history_acc = [], []
  for batch in dataloader:
    optimizer.zero_grad()
    reviews, reviews_length = batch["review"]
    pred = model(reviews, reviews_length)
    sentiments = batch["sentiment"]
    acc = get_accuracy(pred, sentiments)
    loss = criterion(pred, sentiments)
    loss.backward()
    optimizer.step()
    history_loss.append(loss.item()); history_acc.append(acc)

  return np.mean(history_loss), np.mean(history_acc)

def evaluate(dataloader, optimizer, criterion, model):
  history_loss, history_acc = [], []
  with torch.no_grad():
    for batch in dataloader:
      reviews, reviews_length = batch["review"]
      pred = model(reviews, reviews_length)
      sentiments = batch["sentiment"]
      acc = get_accuracy(pred, sentiments)
      loss = criterion(pred, sentiments)
      history_loss.append(loss.item()); history_acc.append(acc)

  return np.mean(history_loss), np.mean(history_acc)

n_epochs = 5
history_train_loss, history_train_acc = [], []
history_val_loss, history_val_acc = [], []

optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
  train_loss, train_acc = train(train_loader, optimizer, criterion, model)
  val_loss, val_acc = evaluate(train_loader, optimizer, criterion, model)

  history_train_loss.append(train_loss); history_train_acc.append(train_acc)
  history_val_loss.append(val_loss); history_val_acc.append(val_acc)

  print("epoch {} ---- training loss {} ---- training accuracy {}%".format(epoch, train_loss, train_acc))
  print("epoch {} ---- val loss {} ---- val accuracy {}%".format(epoch, val_loss, val_acc))
  print("_________________")