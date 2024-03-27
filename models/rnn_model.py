import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import math

OUTPUT_LENGTH = 2 # x and y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough 'pe' matrix that can be sliced according to actual sequence lengths
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return x

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, Y_target="end", model_type="lstm", num_heads=4, num_layers=2):
        super(Model, self).__init__()
        self.Y_target = Y_target
        self.model_type = model_type  # Ensure model_type is always set


# class Model(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, Y_target="end", model_type="lstm"):
#         super(Model, self).__init__()
#         self.Y_target = Y_target

        if model_type == "lstm":
            self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif model_type == "rnn":
            self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif model_type == "gru":
            self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif model_type == "transformer":
            self.embedding = nn.Linear(input_dim, hidden_dim)  # Embedding layer
            self.pos_encoder = PositionalEncoding(hidden_dim)
            encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim * 2)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)



        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim * 2, bias=True)
        self.bn = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.relu = torch.nn.ReLU(inplace=False)
        self.fc2 = torch.nn.Linear(hidden_dim * 2, OUTPUT_LENGTH, bias=True)

    def forward(self, x):
        # x, _status = self.rnn(x)
        if self.model_type in ["lstm", "rnn", "gru"]:
            x, _status = self.rnn(x)
        elif self.model_type == "transformer":
            x = self.embedding(x)  # Embedding input
            x = x.transpose(0, 1)  # Adjust shape to [seq_len, batch, features] for Transformer
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.transpose(0, 1)  # Switch back to [batch, seq_len, features]

        if self.Y_target == "end":
            x = x[:, -1]
            x = self.relu(x)
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
        elif self.Y_target == "all":
            x = self.relu(x)
            bs, seq, hs = x.size()
            x = x.reshape(bs * seq, hs)
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = x.view(bs, seq, OUTPUT_LENGTH)
        else:
            raise RuntimeError("Not implemented!!")

        return x

# if __name__ == "__main__":
#     bs = 256
#     seq_len = 12
#     input_size = 8
#     hs = 128
#     lstm = Model(input_size, hs, "all")
#     inputs = torch.randn(bs, seq_len, input_size)  # make a sequence of length 5
#     #
#     print(inputs.size())
#     out = lstm(inputs)
#     print(out.size())

if __name__ == "__main__":
    bs = 256
    seq_len = 12
    input_size = 8
    hs = 128
    model_type = "transformer"  # Change to 'lstm', 'rnn', 'gru', or 'transformer' as needed
    model = Model(input_size, hs, "all", model_type=model_type)
    inputs = torch.randn(bs, seq_len, input_size)
    out = model(inputs)
    print(out.size())