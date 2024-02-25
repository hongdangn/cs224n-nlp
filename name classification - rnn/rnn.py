import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utils import ALL_LETTERS, path
from utils import letter_to_tensor, line_to_tensor, load_data
from sklearn.model_selection import train_test_split

all_category, all_names, all_targets = load_data(path)
train_idx, test_idx = train_test_split(range(len(all_names)), shuffle = True, \
                                       test_size = 0.2, random_state = 42)
train_data = [
    (all_names[id], all_targets[id]) for id in train_idx
]
test_data = [
    (all_names[id], all_targets[id]) for id in test_idx
]

class rnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def init_hidden(self):
        return nn.init.xavier_uniform_(torch.empty(1, self.hidden_size))
    
    def forward(self, input, pre_hidden):
        input = self.hidden1(input)
        pre_hidden = self.hidden2(pre_hidden)

        hidden = torch.sigmoid(input + pre_hidden)
        output = self.output(hidden)
        return hidden, output
    
        