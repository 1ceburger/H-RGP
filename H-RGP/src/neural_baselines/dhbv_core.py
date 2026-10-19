"""Hydrological model implementation."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from types import SimpleNamespace
BOUNDS = np.array([[50, 700], [0.5, 5], [0.3, 1], [0.05, 1], [0.005, 0.5], [0.0001, 0.1], [0, 10], [0, 50], [1, 7]], float)
tr = SimpleNamespace(AREA=None)

def numpy_step(p, e, soil, upper, lower, params):
    fc, beta, lp, k0, k1, k2, perc, uzl, mb = params.T
    recharge = p * np.clip(soil / fc, 0, 1) ** beta
    added = soil + p - recharge
    overflow = np.maximum(added - fc, 0)
    soil = np.minimum(added, fc)
    ea = np.minimum(soil, e * np.clip(soil / (lp * fc), 0, 1))
    soil = soil - ea
    upper = upper + recharge + overflow
    q0 = np.minimum(upper, k0 * np.maximum(upper - uzl, 0))
    upper -= q0
    transfer = np.minimum(upper, perc)
    upper -= transfer
    lower += transfer
    q1 = np.minimum(upper, k1 * upper)
    upper -= q1
    q2 = np.minimum(lower, k2 * lower)
    lower -= q2
    return (soil, upper, lower)

class WarmData(Dataset):

    def __init__(self, hist, forcing, y, indices, lead):
        self.hist, self.forcing, self.y = (hist, forcing, y)
        self.lead = lead
        self.target_indices = indices[indices - lead >= 364]

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, j):
        target = int(self.target_indices[j])
        end = target - self.lead
        return (self.hist[end - 89:end + 1], self.forcing[end - 364:target + 1], self.y[target])

class DiffHBV(nn.Module):

    def __init__(self, features, initial):
        super().__init__()
        self.parameter_net = nn.Sequential(nn.Linear(features, 32), nn.Tanh(), nn.Linear(32, 9))
        nn.init.zeros_(self.parameter_net[-1].weight)
        nn.init.zeros_(self.parameter_net[-1].bias)
        fraction = np.clip((np.asarray(initial) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]), 0.001, 0.999)
        self.register_buffer('initial_logits', torch.tensor(np.log(fraction / (1 - fraction)), dtype=torch.float32))
        self.register_buffer('lower_bound', torch.tensor(BOUNDS[:, 0], dtype=torch.float32))
        self.register_buffer('width', torch.tensor(BOUNDS[:, 1] - BOUNDS[:, 0], dtype=torch.float32))
        self.capture = False
        self.max_balance_error = 0.0

    def forward(self, hist, forcing):
        raw = self.initial_logits + self.parameter_net(hist.mean(1))
        params = self.lower_bound + self.width * torch.sigmoid(raw)
        par = params.detach().numpy().astype(np.float64)
        rain = forcing.detach().numpy().astype(np.float64)
        soil = 0.5 * par[:, 0]
        upper = np.zeros(len(hist))
        lower = np.zeros(len(hist))
        for day in range(275):
            soil, upper, lower = numpy_step(rain[:, day, 0], rain[:, day, 1], soil, upper, lower, par)
        soil = torch.tensor(soil, dtype=hist.dtype)
        upper = torch.tensor(upper, dtype=hist.dtype)
        lower = torch.tensor(lower, dtype=hist.dtype)
        fc, beta, lp, k0, k1, k2, perc, uzl, mb = params.unbind(-1)
        discharges = []
        for day in range(275, len(forcing[0])):
            p, e = forcing[:, day].unbind(-1)
            old = soil + upper + lower
            recharge = p * (soil / fc).clamp(0, 1).pow(beta)
            added = soil + p - recharge
            overflow = (added - fc).clamp_min(0)
            soil = torch.minimum(added, fc)
            ea = torch.minimum(soil, e * (soil / (lp * fc)).clamp(0, 1))
            soil = soil - ea
            upper = upper + recharge + overflow
            q0 = torch.minimum(upper, k0 * (upper - uzl).clamp_min(0))
            upper = upper - q0
            transfer = torch.minimum(upper, perc)
            upper = upper - transfer
            lower = lower + transfer
            q1 = torch.minimum(upper, k1 * upper)
            upper = upper - q1
            q2 = torch.minimum(lower, k2 * lower)
            lower = lower - q2
            discharge = q0 + q1 + q2
            discharges.append(discharge)
            if self.capture:
                relative = (old + p - ea - discharge - soil - upper - lower).abs() / (old + p).clamp_min(1)
                self.max_balance_error = max(self.max_balance_error, float(relative.max()))
        weights = []

        def primitive(x):
            x = torch.minimum(torch.full_like(mb, float(x)), mb)
            return torch.where(x <= mb / 2, 2 * x.square() / mb.square(), 1 - 2 * (mb - x).square() / mb.square())
        for lag in range(7):
            weights.append(primitive(lag + 1) - primitive(lag))
        q = sum((weight * discharges[-1 - lag] for lag, weight in enumerate(weights)))
        return q * (tr.AREA / 86.4)
