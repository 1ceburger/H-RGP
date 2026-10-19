"""LSTM, MC-LSTM and differentiable HBV model interfaces."""
from pathlib import Path
import json
import numpy as np
import torch
import joblib
import runtime as rt
import lstm_core as lc
import mclstm_core as mc
import dhbv_core as dh
KINDS = ['lstm', 'lstm_nores', 'mclstm', 'dhbv']
NAMES = ['FC_mm', 'BETA', 'LP', 'K0_day', 'K1_day', 'K2_day', 'PERC_mm_day', 'UZL_mm', 'MAXBAS_day']

def setup(loaded):
    global SC, AREA, W, AVG, HIST, TF, Y, FORCING, INITIAL
    X, R = (loaded['train'].x, loaded['train'].r)
    assert X.shape[-1] == 29 and R.shape == (rt.N, 8)
    SC = joblib.load(rt.OUT / 'grid_scaler.pkl')
    AREA = float(rt.G.area_km2.sum())
    W = torch.tensor(rt.G.area_km2.to_numpy() / AREA, dtype=torch.float32)
    AVG = (X * W[None, :, None]).sum(1)
    HIST = torch.cat([AVG, R], -1)
    assert HIST.shape[-1] == 37
    angle = torch.tensor(rt.DATES.dayofyear.to_numpy() / 365.0, dtype=torch.float32) * 2 * np.pi
    TF = torch.stack([torch.sin(angle), torch.cos(angle)], 1)
    Y = loaded['train'].y
    FORCING = (AVG[:, :2] * torch.tensor(SC.scale_[:2], dtype=torch.float32) + torch.tensor(SC.mean_[:2], dtype=torch.float32)).clamp_min(0)
    params = json.loads((Path(__file__).parent / 'hbv_initial_parameters.json').read_text(encoding='utf-8'))
    INITIAL = np.asarray([params[k] for k in NAMES], dtype=np.float64)
    mc.GRID_SCALER = SC
    mc.training.AREA = AREA
    dh.tr.AREA = AREA

def construct(kind, device):
    if kind == 'lstm':
        m = lc.Forecast(37)
    elif kind == 'lstm_nores':
        m = lc.Forecast(29)
    elif kind == 'mclstm':
        m = mc.MassForecast(37)
    elif kind == 'dhbv':
        assert str(device) == 'cpu', 'Differentiable HBV uses detached NumPy state initialization on CPU'
        m = dh.DiffHBV(37, INITIAL)
    else:
        raise ValueError(kind)
    return m.to(device)

def dataset(kind, lead, split, memory=15):
    beg, end = {'train': (0, rt.TRAIN_END), 'val': (rt.TRAIN_END, rt.VAL_END), 'test': (rt.VAL_END, rt.N)}[split]
    ix = np.arange(beg + 14 + lead, end)
    if kind in ['lstm', 'lstm_nores', 'mclstm']:
        return lc.LeadData((AVG if kind == 'lstm_nores' else HIST, AVG[:, 0], Y, TF), ix, lead, memory)
    return dh.WarmData(HIST, FORCING, Y, ix, lead)

def target(kind, b):
    return b[2] if kind == 'dhbv' else b[3]

def forward(kind, m, b):
    return m(b[0], b[1]) if kind == 'dhbv' else m(b[0], b[1], b[2])

def augment(kind, b):
    return list(b)

def batch_size(kind):
    return 32

def eval_batch_size(kind):
    return 128

def memories(kind):
    return [15, 90] if kind in ['lstm', 'lstm_nores'] else [90]
