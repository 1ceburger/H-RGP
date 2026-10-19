"""Evaluation metrics and five-seed aggregation."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
MODELS = ['hrgp', 'lstm', 'lstm_nores', 'mclstm', 'dhbv', 'hbv']
KEYS = ['NSE', 'KGE', 'RMSE_m3_s', 'MAE_m3_s', 'PBIAS_pct']

def metrics(p, o):
    p, o = (np.asarray(p, float), np.asarray(o, float))
    if len(p) < 2 or not np.isfinite(p).all() or (not np.isfinite(o).all()) or (p < 0).any():
        raise ValueError('Invalid predictions')
    return dict(NSE=float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()), KGE=float(1 - np.sqrt((np.corrcoef(p, o)[0, 1] - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)), RMSE_m3_s=float(np.sqrt(np.mean((p - o) ** 2))), MAE_m3_s=float(np.mean(abs(p - o))), PBIAS_pct=float(100 * (p.sum() / o.sum() - 1)))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    a = ap.parse_args()
    rows = []
    ref = {}
    for model in MODELS:
        for lead in [1, 3, 7]:
            for seed in range(5):
                p = (a.runs / 'hbv' / f'lead_{lead}' / f'seed_{seed}' if model == 'hbv' else a.runs / model / f'lead_{lead}' / f'seed_{seed}') / f'{a.split}_predictions.csv'
                df = pd.read_csv(p, parse_dates=['date'])
                if not df.date.is_unique or not df.date.is_monotonic_increasing:
                    raise ValueError('Invalid dates: ' + str(p))
                if lead not in ref:
                    ref[lead] = df[['date', 'observed_m3_s']].copy()
                else:
                    r = ref[lead]
                    if not df.date.equals(r.date) or not np.allclose(df.observed_m3_s, r.observed_m3_s, rtol=2e-06, atol=0.001):
                        raise ValueError('Unequal evaluation dates/observations: ' + str(p))
                pred, obs = (df.predicted_m3_s.to_numpy(float), df.observed_m3_s.to_numpy(float))
                if model in ['lstm', 'lstm_nores', 'mclstm', 'dhbv']:
                    pred, obs = (pred.astype('float32').astype(float), obs.astype('float32').astype(float))
                rows.append(dict(model=model, lead=lead, seed=seed, split=a.split, n=len(df), start=str(df.date.iloc[0].date()), end=str(df.date.iloc[-1].date()), **metrics(pred, obs)))
    frame = pd.DataFrame(rows)
    summary = []
    for (model, lead), g in frame.groupby(['model', 'lead'], sort=False):
        if sorted(g.seed) != list(range(5)):
            raise ValueError('Five seeds required')
        summary.append(dict(model=model, lead=int(lead), n_seeds=5, **{f'{k}_{s}': float(g[k].mean() if s == 'mean' else g[k].std(ddof=1)) for k in KEYS for s in ['mean', 'sample_sd']}))
    a.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.output / f'{a.split}_metrics_by_seed.csv', index=False)
    pd.DataFrame(summary).to_csv(a.output / f'{a.split}_metrics_mean_sd.csv', index=False)
    print(json.dumps(dict(passed=True, runs=len(rows), groups=len(summary), ddof=1)))
if __name__ == '__main__':
    main()
