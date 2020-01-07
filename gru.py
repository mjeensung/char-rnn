import torch
import torch.nn as nn
import pdb

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.h = torch.zeros(self.hidden_size)

        # reset gate
        self.x2r = nn.Embedding(input_size, hidden_size)
        self.h2r = nn.Linear(hidden_size, hidden_size)

        # update gate
        self.x2z = nn.Embedding(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size)

        # hidden unit
        self.x2h = nn.Embedding(input_size, hidden_size)
        self.rh2h = nn.Linear(hidden_size, hidden_size)

        # output
        self.h2y = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        _r = self.sigmoid(self.x2r(x) + self.h2r(self.h)) # reset gates
        # print("_r.shape=",_r.shape)
        _z = self.sigmoid(self.x2z(x) + self.h2z(self.h)) # update gates
        # print("_z.shape=",_z.shape)

        _h_hat = self.tanh(self.x2h(x) + self.rh2h(_r*self.h))
        # print("_h_hat.shape=",_h_hat.shape)
        self.h = _z * self.h + (1 - _z) * _h_hat
        # print("_h.shape=",self.h.shape)

        _y = self.h2y(self.h)
        # print("_y.shape=",_y.shape)
        return _y

    def init_hidden(self):
        self.h = torch.zeros(self.hidden_size)