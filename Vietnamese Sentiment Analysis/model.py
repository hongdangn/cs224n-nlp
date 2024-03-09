import torch
import torch.nn as nn

class BiLSTM(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, n_layers, bidirectional, dropout, pad_idx):
    super(BiLSTM, self).__init__()
    """
        @param vocab_size (int)
        @param embedding_dim (int)
        @param hidden_dim (int)
        @param n_layers (int)
        @param bidirectional (bool)
        @param dropout (float)
    """
    self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)

    self.LSTM = nn.LSTM(embed_size,
                        hidden_size = hidden_size,
                        num_layers = n_layers,
                        bidirectional = bidirectional,
                        dropout = dropout)

    self.dropout = nn.Dropout(dropout)

    self.fc = nn.Linear(2 * hidden_size, 2)

  def forward(self, reviews, reviews_length):
    # reviews: (batch_size, seq_length)
    reviews_embed = self.dropout(self.embedding(reviews))

    # embed: (batch_size, seq_length, embed_size)
    packed_embed = nn.utils.rnn.pack_padded_sequence(reviews_embed,
                                                     reviews_length,
                                                     batch_first = True,
                                                     enforce_sorted = False)

    # hidden = [n layers * n directions, batch size, hidden dim]
    # cell = [n layers * n directions, batch size, hidden dim]
    packed_output, (hidden, cell) = self.LSTM(packed_embed)

    # output = [batch size, seq len, hidden dim * n directions]
    output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)

    # hidden = [batch size, hidden dim * 2]
    out_hidden = self.dropout(torch.cat((hidden[-1], hidden[-2]), dim = -1))

    #final output = [batch_size, 2]
    return self.fc(out_hidden)