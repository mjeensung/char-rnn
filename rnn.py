import torch
import torch.nn as nn
import pdb

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x2h = nn.Embedding(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size,hidden_size)
        self.h2y = nn.Linear(hidden_size,input_size)
        
    def forward(self, x):
        _h = self.h2h(self.h)
        _x = self.x2h(x)
        self.h = nn.Tanh()(_h + _x)
        _y = self.h2y(self.h)
        return _y

    def init_hidden(self):
        self.h = torch.zeros(self.hidden_size)