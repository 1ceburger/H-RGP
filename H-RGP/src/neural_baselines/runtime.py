"""Input verification and training-period preprocessing."""
from pathlib import Path
from types import SimpleNamespace
import json, random, hashlib, contextlib, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
PACKAGE = Path(__file__).resolve().parent
PROFILE = json.loads((PACKAGE / 'expected_data.json').read_text(encoding='utf8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)

@contextlib.contextmanager
def lock_output(out, name='run.lock'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = (out / name).open('a+b')
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        f.close()

def verify_source():
    manifest = json.loads((PACKAGE / 'package_manifest.json').read_text(encoding='utf8'))
    for name, expected in manifest['files'].items():
        assert sha(PACKAGE / name) == expected, 'Package changed: ' + name
    return manifest

def verify_data(folder):
    folder = Path(folder)
    actual = {name: sha(folder / name) for name in PROFILE['required_files']}
    assert actual == PROFILE['reference_hashes'], 'Input files do not match the study configuration'
    return actual

def load_data(data_dir, output_dir, device):
    global I, OUT, DEVICE, DATES, N, TRAIN_END, VAL_END, COLS, G, Q, Q95
    I, OUT, DEVICE = (Path(data_dir).resolve(), Path(output_dir).resolve(), torch.device(device))
    OUT.mkdir(parents=True, exist_ok=True)
    DATES = pd.DatetimeIndex(pd.read_csv(I / 'dates.csv').date)
    N = len(DATES)
    TRAIN_END, VAL_END = (int(0.7 * N), int(0.85 * N))
    COLS = json.loads((I / 'input_metadata.json').read_text(encoding='utf8'))['columns']
    assert COLS == PROFILE['columns']
    raw = np.load(I / 'grid_features_full.npy', mmap_mode='r')
    assert list(raw.shape) == PROFILE['reference_shape']
    G = pd.read_csv(I / 'static_grid_features.csv')
    assert len(G) == 634 and np.all(G.area_km2.to_numpy() > 0)
    Q = pd.read_csv(I / 'basin_daily.csv').q_obs_m3_s.to_numpy()
    assert len(Q) == N and np.isfinite(Q).all() and np.all(Q >= 0)
    assert DATES.is_unique and DATES.is_monotonic_increasing
    assert np.all(np.diff(DATES.values) == np.timedelta64(1, 'D'))
    Q95 = float(np.quantile(Q[:TRAIN_END], 0.95))
    sc = StandardScaler().fit(raw[:TRAIN_END].reshape(-1, len(COLS)))
    x = torch.from_numpy(sc.transform(raw.reshape(-1, len(COLS))).reshape(raw.shape).astype('float32'))
    res = pd.read_csv(I / 'reservoir_causal_daily.csv')[['res_level', 'delta_level', 'res_outflow', 'res_in_final', 'res_outflow_lag1']]
    missing = res.isna().to_numpy()
    means = res.iloc[:TRAIN_END].mean()
    assert np.isfinite(means).all()
    filled = res.fillna(means)
    rs = StandardScaler().fit(filled.iloc[:TRAIN_END])
    r = torch.from_numpy(np.column_stack([rs.transform(filled).astype('float32'), missing[:, [0, 2, 3]].astype('float32')]))
    assert r.shape == (N, 8) and torch.isfinite(r).all()
    joblib.dump(rs, OUT / 'reservoir_scaler.pkl')
    assert torch.isfinite(x).all()
    joblib.dump(sc, OUT / 'grid_scaler.pkl')
    return {split: Windows(x, r, 1, split) for split in ['train', 'val']}

class Windows(Dataset):

    def __init__(self, x, r, lead, split):
        self.x, self.r, self.lead = (x, r, lead)
        assert lead in [1, 3, 7]
        beg, end = {'train': (0, TRAIN_END), 'val': (TRAIN_END, VAL_END), 'test': (VAL_END, N)}[split]
        self.target_indices = np.arange(beg + 14 + lead, end)
        self.y = torch.tensor(np.log1p(Q), dtype=torch.float32)
        self.tf = torch.tensor(np.stack([DATES.dayofyear.to_numpy() / 365.0, DATES.month.to_numpy()], 1), dtype=torch.float32)

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, j):
        target = int(self.target_indices[j])
        end = target - self.lead
        return (self.x[end - 14:end + 1], self.r[end - 14:end + 1], self.y[target], self.x[end + 1:target + 1, :, 0].T, self.tf[target])

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class HydroLoss(nn.Module):

    def __init__(self, alpha=2.0, beta=0.1, q_ref=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer('q_ref', torch.tensor(float(q_ref)))

    def forward(self, pred_log, target_log):
        p, o = (torch.expm1(pred_log), torch.expm1(target_log))
        weights = 1 + self.alpha * o / (self.q_ref + 1e-06)
        return ((pred_log - target_log) ** 2 * weights).mean() + self.beta * abs(p.sum() - o.sum()) / (o.sum() + 1e-06)
layers = SimpleNamespace(HydroLoss=HydroLoss)

def metrics(p, o):
    p, o = (np.asarray(p, dtype=float), np.asarray(o, dtype=float))
    assert np.isfinite(p).all() and np.isfinite(o).all() and (p >= 0).all()
    rr = np.corrcoef(p, o)[0, 1]
    return dict(n=len(p), NSE=float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()), KGE=float(1 - np.sqrt((rr - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)), RMSE_m3_s=float(np.sqrt(np.mean((p - o) ** 2))), MAE_m3_s=float(np.mean(abs(p - o))), PBIAS_pct=float(100 * (p.sum() / o.sum() - 1)), observed_peak_day_bias_pct=float(100 * (p[o.argmax()] - o.max()) / o.max()))
