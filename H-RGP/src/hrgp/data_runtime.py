"""Input loading, training-period preprocessing and chronological windows."""
from pathlib import Path
import json, math, types, random
import numpy as np, pandas as pd, torch, joblib
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import layers as layers
import network as network
from storage_release import ConservativePowerCell
from routing_setup import ShrunkSAP, attach_execution
from grid_encoder import grid_forward
from types import SimpleNamespace
PROFILE = json.loads((Path(__file__).parent / 'expected_data.json').read_text(encoding='utf8'))
layers.Config = SimpleNamespace(PRIOR_MASKS={'SOF_mix': PROFILE['mask']}, RES_COLS=PROFILE['reservoir_columns'])

def load_data(data_dir, output_dir, device):
    global I, OUT, DEVICE, DATES, N, TRAIN_END, VAL_END, COLS, RAW, G, NODE, TOPO, B, Q, Q95
    I = Path(data_dir).resolve()
    OUT = Path(output_dir).resolve()
    DEVICE = torch.device(device)
    DATES = pd.DatetimeIndex(pd.read_csv(I / 'dates.csv').date)
    N = len(DATES)
    TRAIN_END = int(0.7 * N)
    VAL_END = int(0.85 * N)
    COLS = json.loads((I / 'input_metadata.json').read_text(encoding='utf8'))['columns']
    assert COLS == PROFILE['columns'], 'Input columns must match the declared feature order'
    RAW = np.load(I / 'grid_features_full.npy', mmap_mode='r')
    G = pd.read_csv(I / 'static_grid_features.csv')
    NODE = pd.read_csv(I / 'nodes.csv')
    TOPO = np.load(I / 'topology.npz')
    B = pd.read_csv(I / 'basin_daily.csv')
    Q = B.q_obs_m3_s.to_numpy()
    Q95 = float(np.quantile(Q[:TRAIN_END], 0.95))
    assert RAW.shape == (N, 634, 29) and len(NODE) == 15
    assert len(B) == N and np.isfinite(Q).all() and (Q >= 0).all()
    assert DATES.is_monotonic_increasing and DATES.is_unique
    assert (np.diff(DATES.values) == np.timedelta64(1, 'D')).all()
    assert np.allclose(TOPO['mapping'].sum(0), 1.0)
    x, r, sc = prepare()
    return {s: Windows(x, r, 1, s) for s in ['train', 'val']}

def lengths():
    nodes = NODE.set_index('node_id')
    distance = nodes.dist_to_outlet
    result = [float(distance[i] - distance[int(nodes.loc[i, 'downstream_node'])]) for i in range(14)]
    assert min(result) > 0
    boundary = json.loads((Path(__file__).parent / 'geometry.json').read_text(encoding='utf8'))
    result[13] = boundary['terminal13_to_dam_reach_m']
    return result

class Windows(Dataset):

    def __init__(self, x, r, lead, split):
        self.x = x
        self.r = r
        self.lead = lead
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

def prepare():
    sc = StandardScaler().fit(RAW[:TRAIN_END].reshape(-1, len(COLS)))
    x = torch.from_numpy(sc.transform(RAW.reshape(-1, len(COLS))).reshape(RAW.shape).astype('float32'))
    global INQ_TARGET, INQ_MASK, INQ95
    raw_res = pd.read_csv(I / 'reservoir_causal_daily.csv')
    INQ_TARGET = raw_res.res_in_final.to_numpy(float)
    INQ_MASK = np.isfinite(INQ_TARGET) & (raw_res.inflow_coverage.to_numpy() >= 0.95)
    INQ95 = float(np.quantile(INQ_TARGET[:TRAIN_END][INQ_MASK[:TRAIN_END]], 0.95))
    res = raw_res[layers.Config.RES_COLS]
    missing = res.isna().to_numpy()
    means = res.iloc[:TRAIN_END].mean()
    if not np.isfinite(means).all():
        raise RuntimeError('Reservoir training means unavailable')
    filled = res.fillna(means)
    rs = StandardScaler().fit(filled.iloc[:TRAIN_END])
    r = torch.from_numpy(np.column_stack([rs.transform(filled).astype('float32'), missing[:, [0, 2, 3]].astype('float32')]))
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(sc, OUT / 'grid_scaler.pkl')
    joblib.dump(rs, OUT / 'reservoir_scaler.pkl')
    return (x, r, sc)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_model():
    mapping = TOPO['mapping']
    count = len(NODE)
    feat = NODE[['dist_to_outlet', 'num_local_grids', 'upstream_area_km2', 'lon', 'lat']].to_numpy(np.float32)
    static = mapping @ G[['slope', 'flow_len']].to_numpy(np.float32) / mapping.sum(1, keepdims=True)
    vr = {'sorted_reservoirs': list(range(count)), 'edge_index': TOPO['node_edges'], 'grid_to_reservoir_matrix': mapping, 'reservoir_features': feat, 'num_reservoirs': count}
    edge = torch.tensor(TOPO['grid_edges'], dtype=torch.long, device=DEVICE)
    mask = [COLS.index(c) for c in layers.Config.PRIOR_MASKS['SOF_mix']]
    model = network.RunoffNetwork(edge, vr, len(COLS), 8, mask, len(G), [COLS.index('flow_len'), COLS.index('slope')], static, sm_col_idx=COLS.index('sm')).to(DEVICE)
    model.basin.axis_correct = True
    model.basin.sample_norm = True
    model.basin.forward = types.MethodType(grid_forward, model.basin)
    net = model.vr_net
    contributing_area = NODE.local_area_km2.to_numpy().copy()
    for node in range(count):
        for src, dst in TOPO['node_edges'].T:
            if int(dst) == node:
                contributing_area[node] += contributing_area[int(src)]
    assert abs(contributing_area[-1] - G.area_km2.sum()) < 0.0001
    refs = np.maximum(1.0, Q95 * contributing_area / G.area_km2.sum()).astype('float32')
    net.register_buffer('reference_flows', torch.tensor(refs, device=DEVICE))
    transformed = torch.log1p(net.static_sf.clamp_min(0))
    net.static_sf.copy_((transformed - transformed.mean(0)) / transformed.std(0, unbiased=False).clamp_min(0.01))
    net.river_features.copy_((net.river_features - net.river_features.mean(0)) / net.river_features.std(0, unbiased=False).clamp_min(0.01))
    net.vr_cells = nn.ModuleDict({str(n): ConservativePowerCell(feat.shape[-1], refs[n] * 86400.0) for n in net.sorted_nodes}).to(DEVICE)
    net.sap_policy = ShrunkSAP('constant').to(DEVICE)
    for p in net.spatial_param_mlp.parameters():
        p.requires_grad_(False)
    attach_execution(net, 'constant')
    return model

def metrics(p, o):
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    rr = float(np.corrcoef(p, o)[0, 1])
    av = float(p.std() / o.std())
    bv = float(p.mean() / o.mean())
    return {'n': len(p), 'NSE': float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)), 'KGE': float(1 - np.sqrt((rr - 1) ** 2 + (av - 1) ** 2 + (bv - 1) ** 2)), 'RMSE_m3_s': float(np.sqrt(np.mean((p - o) ** 2))), 'MAE_m3_s': float(np.mean(np.abs(p - o))), 'PBIAS_pct': float(100 * (p.sum() / o.sum() - 1)), 'observed_peak_day_bias_pct': float(100 * (p[o.argmax()] - o.max()) / o.max())}
