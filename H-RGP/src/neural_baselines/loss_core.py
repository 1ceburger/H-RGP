"""Hydrological model implementation."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from types import SimpleNamespace

class HydroLoss(nn.Module):

    def __init__(self, alpha=2.0, beta=0.1, q_ref=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')
        if q_ref is None:
            self.register_buffer('q_ref', torch.tensor(0.0))
            self.use_fixed_ref = False
        else:
            self.register_buffer('q_ref', torch.tensor(float(q_ref)))
            self.use_fixed_ref = True

    def forward(self, pred_log, target_log):
        pred_phys = torch.expm1(pred_log)
        target_phys = torch.expm1(target_log)
        if self.use_fixed_ref and float(self.q_ref) > 0:
            ref = self.q_ref.to(device=target_phys.device, dtype=target_phys.dtype) + 1e-06
        else:
            ref = target_phys.max().detach() + 1e-06
        weights = 1.0 + self.alpha * (target_phys / ref)
        loss_pixel = self.mse(pred_log, target_log)
        loss_weighted = (loss_pixel * weights).mean()
        vol_pred = pred_phys.sum()
        vol_target = target_phys.sum()
        loss_vol = torch.abs(vol_pred - vol_target) / (vol_target + 1e-06)
        return loss_weighted + self.beta * loss_vol
