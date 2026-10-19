"""Hydrological model implementation."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from types import SimpleNamespace

class LeadData(Dataset):

    def __init__(self, shared, indices, lead, length):
        self.hist, self.rain, self.y, self.tf = shared
        self.lead, self.length = (lead, length)
        self.target_indices = indices[indices - lead >= length - 1]

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, j):
        target = int(self.target_indices[j])
        end = target - self.lead
        return (self.hist[end - self.length + 1:end + 1], self.rain[end + 1:target + 1, None], self.tf[target], self.y[target])

class Forecast(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.enc = nn.LSTM(features, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.rain_enc = nn.Linear(1, 64)
        self.future = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(66, 64), nn.Tanh(), nn.Linear(64, 1), nn.Softplus())
        for names in self.enc._all_weights:
            for name in names:
                if 'bias' in name:
                    with torch.no_grad():
                        getattr(self.enc, name)[64:128].fill_(1.0)

    def forward(self, x, r, t):
        _, (h, _) = self.enc(x)
        _, hf = self.future(torch.tanh(self.rain_enc(r)), h[-1:])
        logq = self.head(torch.cat([hf[0], t], -1)).squeeze(-1).clamp(max=np.log1p(1000000.0))
        return torch.expm1(logq)
