import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
  def __init__(self, embeddings, n_features = 36, hidden_size = 200, n_classes = 3, dropout_prob = 0.5):
    super(ParserModel, self).__init__()
    """ Initialize the parser model.

        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
    """
    self.n_features = n_features
    self.n_classes = n_classes
    self.dropout_prob = dropout_prob
    self.embed_size = embeddings.shape[1]
    self.hidden_size = hidden_size

    self.embeddings = embeddings
    self.dropout = nn.Dropout(dropout_prob)
    self.relu = nn.ReLU()

    U = nn.Parameter(torch.empty(self.n_features * self.embed_size, self.hidden_size))
    b_1 = nn.Parameter(torch.empty(self.hidden_size))

    self.embed_to_hidden_weight = nn.init.xavier_uniform_(U)
    self.embed_to_hidden_bias = nn.init.uniform_(b_1)

    W = nn.Parameter(torch.empty(self.hidden_size, self.n_classes))
    b_2 = nn.Parameter(torch.empty(self.n_classes))

    self.hidden_to_logits_weight = nn.init.xavier_uniform_(W)
    self.hidden_to_logits_bias = nn.init.uniform_(b_2)

  def embedding_lookup(self, w):
    # param w (Tensor): input tensor of word indices (batch_size, n_features)
    x = self.embeddings[w]
    x = torch.tensor(x)
    x = x.view(x.size(0), -1)
    return x

  def forward(self, w):
    x = self.embedding_lookup(w)
    x = (torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias).reshape(w.shape[0], self.hidden_size)
    x = self.relu(x)
    x = self.dropout(x)
    x = torch.matmul(x, self.hidden_to_logits_weight) + self.hidden_to_logits_bias
    return x