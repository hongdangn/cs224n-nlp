import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, path
from rnn import all_category, train_data, test_data
from rnn import rnn

def train(model, n_epochs, optimizer, criterion):
    history = []
    for epoch in range(n_epochs):
        for i, (name, label) in enumerate(train_data):
            hidden_state = model.init_hidden()
            for input in name:
                hidden_state, output = model(input, hidden_state)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            if i % 2000 == 0:
                print("i = {}, loss = {}".format(i, loss.item()))
    return history

model = rnn(input_size = len(ALL_LETTERS), hidden_size = 250, output_size = len(all_category))
optimizer = optim.Adam(model.parameters(), lr = 0.002)
criterion = nn.CrossEntropyLoss()

history = train(model, n_epochs = 3, optimizer = optimizer, criterion = criterion)

## inference
with torch.no_grad():
  pred = 0
  for name, label in test_data:
    hidden_state = model.init_hidden()
    for input in name:
      hidden_state, output = model(input, hidden_state)
    pred_label = torch.argmax(output)
    if pred_label == label[0]:
      pred += 1
  print("The accuracy is {}%".format(pred/len(test_data) * 100))