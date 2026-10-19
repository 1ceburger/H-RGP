"""Training and evaluation entry point."""
from pathlib import Path
import os, json, time
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
from numba import njit
from types import SimpleNamespace
import argparse, hashlib
AP = argparse.ArgumentParser()
AP.add_argument('--data', required=True)
AP.add_argument('--output', required=True)
AP.add_argument('--seed', type=int, choices=range(5), required=True)
AP.add_argument('--preflight-only', action='store_true')
AP.add_argument('--eval-only', action='store_true')
AP.add_argument('--evaluate-test', action='store_true')
args = AP.parse_args()
I = Path(args.data).resolve()
PROFILE = json.loads((Path(__file__).parent / 'expected_data.json').read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
INPUT_HASHES = {n: sha(I / n) for n in PROFILE['required_files']}
assert INPUT_HASHES == PROFILE['reference_hashes'], 'HBV input mismatch'
DATES = pd.DatetimeIndex(pd.read_csv(I / 'dates.csv').date)
G = pd.read_csv(I / 'static_grid_features.csv')
Q = pd.read_csv(I / 'basin_daily.csv').q_obs_m3_s.to_numpy(float)
rt = SimpleNamespace(I=I, DATES=DATES, G=G, Q=Q, N=len(DATES), TRAIN_END=int(0.7 * len(DATES)), VAL_END=int(0.85 * len(DATES)))
AREA = float(G.area_km2.sum())

def metrics(p, o):
    p, o = (np.asarray(p, float), np.asarray(o, float))
    assert np.isfinite(p).all() and np.isfinite(o).all() and (p >= 0).all()
    return dict(n=len(p), NSE=float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()), KGE=float(1 - np.sqrt((np.corrcoef(p, o)[0, 1] - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)), RMSE_m3_s=float(np.sqrt(np.mean((p - o) ** 2))), MAE_m3_s=float(np.mean(abs(p - o))), PBIAS_pct=float(100 * (p.sum() / o.sum() - 1)))
OUT = Path(args.output).resolve()
OUT.mkdir(parents=True, exist_ok=True)
FD = pd.DatetimeIndex(pd.read_csv(rt.I / 'forcing_dates.csv').date)
W = rt.G.area_km2.to_numpy() / AREA
P = np.load(rt.I / 'forcing_rain_mm.npy') @ W
E = np.load(rt.I / 'forcing_pet_mm.npy') @ W
OBS = np.full(len(FD), np.nan)
POS = FD.get_indexer(rt.DATES)
OBS[POS] = rt.Q
assert POS[0] == 365
NAMES = ['FC_mm', 'BETA', 'LP', 'K0_day', 'K1_day', 'K2_day', 'PERC_mm_day', 'UZL_mm', 'MAXBAS_day']
BOUNDS = np.array([[50, 700], [0.5, 5], [0.3, 1], [0.05, 1], [0.005, 0.5], [0.0001, 0.1], [0, 10], [0, 50], [1, 7]], float)

def target_indices(split, lead):
    beg, end = {'train': (0, rt.TRAIN_END), 'val': (rt.TRAIN_END, rt.VAL_END), 'test': (rt.VAL_END, rt.N)}[split]
    return POS[np.arange(beg + 14 + lead, end)]
TRAIN = target_indices('train', 1)
VAL = target_indices('val', 1)

@njit(cache=True)
def primitive(x, mb):
    x = min(x, mb)
    return 2 * x * x / (mb * mb) if x <= mb / 2 else 1 - 2 * (mb - x) ** 2 / (mb * mb)

@njit(cache=True)
def hbv_forward(params, p, e, area):
    n = len(p)
    pop = params.shape[1]
    out = np.zeros((n, pop))
    for j in range(pop):
        fc, beta, lp, k0, k1, k2, perc, uzl, mb = params[:, j]
        sm = 0.5 * fc
        upper = 0.0
        lower = 0.0
        queue = np.zeros(7)
        weights = np.array([primitive(k + 1.0, mb) - primitive(float(k), mb) for k in range(7)])
        for day in range(n):
            recharge = p[day] * max(0.0, min(sm / fc, 1.0)) ** beta
            added = sm + p[day] - recharge
            excess = max(added - fc, 0.0)
            sm = min(added, fc)
            ea = min(e[day] * min(sm / (lp * fc), 1.0), sm)
            sm -= ea
            upper += recharge + excess
            q0 = min(k0 * max(upper - uzl, 0.0), upper)
            upper -= q0
            transfer = min(perc, upper)
            upper -= transfer
            lower += transfer
            q1 = min(k1 * upper, upper)
            upper -= q1
            q2 = min(k2 * lower, lower)
            lower -= q2
            for k in range(6, 0, -1):
                queue[k] = queue[k - 1]
            queue[0] = q0 + q1 + q2
            out[day, j] = (queue * weights).sum() * area / 86.4
    return out

def simulate(params, n):
    params = np.asarray(params, float)
    if params.ndim == 1:
        params = params[:, None]
    return hbv_forward(params, P[:n], E[:n], AREA)

def nse(pred, obs):
    return 1 - ((pred - obs) ** 2).sum(axis=0) / ((obs - obs.mean()) ** 2).sum()

def objective(params):
    return 1 - nse(simulate(params, int(TRAIN[-1]) + 1)[TRAIN], OBS[TRAIN, None])

def main():
    records = []
    for seed in [args.seed]:
        folder = OUT / f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        path = folder / 'validation_metadata.json'
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        if args.eval_only:
            raise FileNotFoundError('No calibration metadata for evaluation: ' + str(path))
        best = -np.inf
        bad = 0
        generation = 0
        selected = None
        history = []
        start = time.time()

        def callback(x, convergence):
            nonlocal best, bad, generation, selected
            generation += 1
            q = simulate(x, int(VAL[-1]) + 1)[:, 0]
            val = float(nse(q[VAL], OBS[VAL]))
            if val > best + 1e-05:
                best = val
                bad = 0
                selected = x.copy()
            else:
                bad += 1
            row = {'model': 'hbv', 'seed': seed, 'generation': generation, 'train_NSE': float(nse(q[TRAIN], OBS[TRAIN])), 'val_NSE': val, 'best_val_NSE': best, 'bad': bad, 'seconds': time.time() - start}
            history.append(row)
            pd.DataFrame(history).to_csv(folder / 'calibration_history.csv', index=False)
            (OUT / 'status.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
            return bad >= 20
        result = differential_evolution(objective, BOUNDS, popsize=12, maxiter=300, tol=1e-07, atol=1e-08, seed=seed, callback=callback, polish=False, vectorized=True, updating='deferred')
        assert selected is not None
        meta = {'model': 'hbv', 'seed': seed, 'best_val_NSE': best, 'parameters': dict(zip(NAMES, map(float, selected))), 'generations': generation, 'objective': 'training NSE on main lead-1 target dates; 365-day pre-feature warmup', 'selection': 'validation NSE only', 'warmup_days': 365, 'area_km2': AREA, 'future_forcing': 'retrospective fused rain; estimated PET from retrospective meteorology', 'routing': 'exact day-integrated triangular weights, support at most 7 days', 'upper_store_sequence': 'q0, percolation, q1', 'input_hashes': INPUT_HASHES, 'termination': str(result.message)}
        path.write_text(json.dumps(meta, indent=2))
        records.append(meta)
    old = json.loads((OUT / 'calibration_parameters.json').read_text()) if (OUT / 'calibration_parameters.json').exists() else []
    merged = {r['seed']: r for r in old}
    merged.update({r['seed']: r for r in records})
    (OUT / 'calibration_parameters.json').write_text(json.dumps([merged[k] for k in sorted(merged)], indent=2))
    rows = []
    for meta in records:
        q = simulate([meta['parameters'][k] for k in NAMES], len(FD))[:, 0]
        assert np.isfinite(q).all() and (q >= 0).all()
        for lead in [1, 3, 7]:
            dst = OUT / f"lead_{lead}/seed_{meta['seed']}"
            dst.mkdir(parents=True, exist_ok=True)
            row = {'model': 'hbv', 'lead': lead, 'seed': meta['seed']}
            for split in ['val'] + (['test'] if args.evaluate_test else []):
                ix = target_indices(split, lead)
                sc = metrics(q[ix], OBS[ix])
                row.update({split + '_' + k: v for k, v in sc.items()})
                pd.DataFrame({'date': FD[ix], 'observed_m3_s': OBS[ix], 'predicted_m3_s': q[ix]}).to_csv(dst / f'{split}_predictions.csv', index=False)
            (dst / 'evaluation_metrics.json').write_text(json.dumps(row, indent=2))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'per_run_metrics.csv', index=False)
    print('CONTINUOUS_HBV_COMPLETE', flush=True)
if __name__ == '__main__':
    if args.preflight_only:
        q = simulate(BOUNDS.mean(1), int(VAL[-1]) + 1)[:, 0]
        assert np.isfinite(q).all() and (q >= 0).all()
        (OUT / 'preflight.json').write_text(json.dumps(dict(passed=True, no_optimization=True, no_test_scoring=True, validation_n=len(VAL))))
        print('HBV_PREFLIGHT_PASSED')
    else:
        main()
