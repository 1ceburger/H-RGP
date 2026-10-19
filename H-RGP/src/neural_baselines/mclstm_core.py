"""Hydrological model implementation."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from types import SimpleNamespace
training = SimpleNamespace(AREA=None)
GRID_SCALER = None

class MassForecast(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.hidden = 16
        sc = GRID_SCALER
        self.mu = float(sc.mean_[0])
        self.sd = float(sc.scale_[0])
        width = features + self.hidden
        self.allocate = nn.Linear(width, self.hidden)
        self.redistribute = nn.Linear(width, self.hidden * self.hidden)
        self.release = nn.Linear(width, self.hidden)
        for layer in [self.allocate, self.redistribute, self.release]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.release.bias, -1.0)
        self.register_buffer('identity', torch.eye(self.hidden))
        self.capture = False
        self.max_relative_closure_error = 0.0

    def forward(self, x, r, t):
        rain = torch.cat([x[:, :, :1], r], dim=1) * self.sd + self.mu
        rain = rain.clamp_min(0)
        aux = torch.cat([x[:, :, 1:], x[:, -1:, 1:].expand(-1, r.shape[1], -1)], dim=1)
        state = x.new_zeros(len(x), self.hidden)
        released = torch.zeros_like(state)
        for day in range(rain.shape[1]):
            mass = rain[:, day]
            normalized = state / state.sum(-1, keepdim=True).clamp_min(1e-08)
            features = torch.cat([mass, aux[:, day], normalized], dim=-1)
            gate = torch.sigmoid(self.allocate(features))
            gate = gate / gate.sum(-1, keepdim=True)
            matrix = torch.relu(self.redistribute(features).view(-1, self.hidden, self.hidden))
            sums = matrix.sum(-1, keepdim=True)
            matrix = torch.where(sums > 0, matrix / sums.clamp_min(1e-08), self.identity[None])
            available = torch.bmm(state[:, None], matrix).squeeze(1) + mass * gate
            released = available * torch.sigmoid(self.release(features))
            old = state
            state = available - released
            if self.capture:
                relative = (state.sum(-1) + released.sum(-1) - old.sum(-1) - mass[:, 0]).abs() / (old.sum(-1) + mass[:, 0]).clamp_min(1.0)
                self.max_relative_closure_error = max(self.max_relative_closure_error, float(relative.max().detach()))
        return released[:, 1:].sum(-1) * (training.AREA / 86.4)
