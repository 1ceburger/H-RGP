"""Independent training and evaluation for a fixed lead time and seed."""
from pathlib import Path
import argparse, os, time, json, random, platform
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import runtime as rt
import models as md
from loss_core import HydroLoss
P = Path(__file__).resolve().parent

def stopped(out):
    return any(((p / 'STOP_REQUESTED').exists() for p in [P, out, *out.parents]))

def save(p, value):
    tmp = Path(str(p) + '.tmp')
    torch.save(value, tmp)
    tmp.replace(p)

def move(b, device):
    return [v.to(device) for v in b]

@torch.no_grad()
def evaluate(kind, m, ds, device):
    m.eval()
    pred = []
    for b in DataLoader(ds, batch_size=md.eval_batch_size(kind), shuffle=False):
        pred.append(md.forward(kind, m, move(b, device)).cpu().numpy())
    p = np.concatenate(pred)
    o = torch.expm1(ds.y[ds.target_indices]).cpu().numpy()
    assert np.isfinite(p).all() and np.all(p >= 0)
    return (rt.metrics(p, o), p, o)

def fit(kind, lead, seed, memory, out, device, max_epochs):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'fit_completed.json').exists():
        return json.loads((out / 'fit_completed.json').read_text(encoding='utf8'))
    if stopped(out):
        raise InterruptedError('STOP_REQUESTED')
    rt.seed_all(seed)
    m = md.construct(kind, device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.0005, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    generator = torch.Generator().manual_seed(seed)
    ds = md.dataset(kind, lead, 'train', memory)
    val = md.dataset(kind, lead, 'val', memory)
    loader = DataLoader(ds, batch_size=md.batch_size(kind), shuffle=True, generator=generator)
    criterion = HydroLoss(alpha=2.0, beta=0.1, q_ref=rt.Q95).to(device)
    epoch = 0
    best = -float('inf')
    bad = 0
    best_epoch = 0
    history = []
    if (out / 'last.pth').exists():
        s = torch.load(out / 'last.pth', weights_only=False, map_location=device)
        m.load_state_dict(s['model'])
        opt.load_state_dict(s['optimizer'])
        sched.load_state_dict(s['scheduler'])
        epoch, best, bad, best_epoch, history = (s['epoch'], s['best'], s['bad'], s['best_epoch'], s['history'])
        torch.set_rng_state(s['torch_rng'].cpu())
        np.random.set_state(s['numpy_rng'])
        random.setstate(s['python_rng'])
        generator.set_state(s['loader_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda_rng']])
    start = time.time()
    while epoch < max_epochs and bad < 20:
        if stopped(out):
            raise InterruptedError('STOP_REQUESTED; resume from last completed epoch')
        epoch += 1
        m.train()
        total = 0
        count = 0
        for b in loader:
            b = md.augment(kind, move(b, device))
            opt.zero_grad(set_to_none=True)
            pred = md.forward(kind, m, b)
            loss = criterion(torch.log1p(pred)[:, None], md.target(kind, b)[:, None])
            assert torch.isfinite(loss), 'nonfinite loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            assert torch.isfinite(norm)
            opt.step()
            total += float(loss.detach()) * len(pred)
            count += len(pred)
        score, _, _ = evaluate(kind, m, val, device)
        sched.step(score['NSE'])
        if score['NSE'] > best + 1e-05:
            best = score['NSE']
            best_epoch = epoch
            bad = 0
            save(out / 'best.pth', m.state_dict())
        else:
            bad += 1
        row = dict(model=kind, lead=lead, seed=seed, memory=memory, epoch=epoch, val_NSE=score['NSE'], best_val_NSE=best, best_epoch=best_epoch, bad=bad, loss=total / count, seconds=time.time() - start)
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
        save(out / 'last.pth', dict(model=m.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), epoch=epoch, best=best, best_epoch=best_epoch, bad=bad, history=history, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), loader_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None))
        rt.write(out / 'current_status.json', dict(state='training', pid=os.getpid(), **row))
        print(json.dumps(row), flush=True)
    meta = dict(memory=memory, epochs=epoch, best_epoch=best_epoch, best_val_NSE=best, checkpoint_sha256=rt.sha(out / 'best.pth'), parameters=sum((v.numel() for v in m.parameters())))
    rt.write(out / 'fit_completed.json', meta)
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', choices=md.KINDS, required=True)
    ap.add_argument('--lead', type=int, choices=[1, 3, 7], required=True)
    ap.add_argument('--seed', type=int, choices=range(5), required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--evaluate-test', action='store_true')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--smoke-epochs', type=int, default=0, help='Separate implementation smoke run only, NEVER a paper result')
    a = ap.parse_args()
    device = 'cpu' if a.model in ['mclstm', 'dhbv'] else a.device
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1 if a.model in ['mclstm', 'dhbv'] else 2 if device == 'cpu' else 4)
    with rt.lock_output(out):
        if stopped(out):
            raise InterruptedError('STOP_REQUESTED')
        rt.verify_source()
        data_hashes = rt.verify_data(a.data)
        loaded = rt.load_data(a.data, out, device)
        md.setup(loaded)
        protocol = dict(model=a.model, lead=a.lead, seed=a.seed, reservoir_input=a.model != 'lstm_nores', reservoir_scaler=True, source_hashes=json.loads((P / 'package_manifest.json').read_text())['files'], data_hashes=data_hashes, max_epochs=a.smoke_epochs or 160, patience=20, min_delta=1e-05, optimizer='AdamW', lr=0.0005, weight_decay=0.01, clip=1.0, batch_size=md.batch_size(a.model), memories=md.memories(a.model), memory_selection='validation NSE only', checkpoint_selection='validation NSE only', rain_noise=0.0, device=device, torch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda, smoke_only=bool(a.smoke_epochs))
        if (out / 'protocol.json').exists():
            old = json.loads((out / 'protocol.json').read_text())
            for k in ['model', 'lead', 'seed', 'data_hashes', 'source_hashes', 'max_epochs']:
                assert old[k] == protocol[k], k
        else:
            rt.write(out / 'protocol.json', protocol)
        if a.preflight_only:
            rt.seed_all(a.seed)
            m = md.construct(a.model, device)
            ds = md.dataset(a.model, a.lead, 'train', md.memories(a.model)[0])
            b = move(next(iter(DataLoader(ds, batch_size=32))), device)
            p = md.forward(a.model, m, b)
            loss = HydroLoss(alpha=2.0, beta=0.1, q_ref=rt.Q95).to(device)(torch.log1p(p)[:, None], md.target(a.model, b)[:, None])
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(p).all() and (p >= 0).all()
            assert all((v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()))
            rt.write(out / 'preflight.json', dict(passed=True, model=a.model, lead=a.lead, device=device, actual_dates=len(p), no_training_or_test=True))
            return
        if not a.eval_only:
            candidates = [fit(a.model, a.lead, a.seed, memory, out / f'memory_{memory}', device, a.smoke_epochs or 160) for memory in md.memories(a.model)]
            selected = max(candidates, key=lambda x: x['best_val_NSE'])
            rt.write(out / 'selection.json', dict(selection='validation NSE only', selected=selected, candidates=candidates))
        selection = json.loads((out / 'selection.json').read_text())['selected']
        checkpoint = out / f"memory_{selection['memory']}" / 'best.pth'
        assert rt.sha(checkpoint) == selection['checkpoint_sha256']
        rt.seed_all(a.seed)
        m = md.construct(a.model, device)
        m.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        report = dict(model=a.model, lead=a.lead, seed=a.seed, reservoir_input=a.model != 'lstm_nores', selected_memory=selection['memory'], checkpoint_sha256=rt.sha(checkpoint), smoke_only=bool(a.smoke_epochs), splits={})
        existing = json.loads((out / 'evaluation_metrics.json').read_text()) if (out / 'evaluation_metrics.json').exists() else None
        for split in ['val'] + (['test'] if a.evaluate_test else []):
            if existing and split in existing['splits']:
                assert existing['checkpoint_sha256'] == report['checkpoint_sha256']
                report['splits'][split] = existing['splits'][split]
                continue
            ds = md.dataset(a.model, a.lead, split, selection['memory'])
            score, p, o = evaluate(a.model, m, ds, device)
            path = out / f'{split}_predictions.csv'
            pd.DataFrame(dict(date=rt.DATES[ds.target_indices], observed_m3_s=o, predicted_m3_s=p)).to_csv(path, index=False)
            report['splits'][split] = dict(metrics=score, zero_days=int((p == 0).sum()), predictions_sha256=rt.sha(path))
        if existing:
            for s, v in existing['splits'].items():
                report['splits'].setdefault(s, v)
        rt.write(out / 'evaluation_metrics.json', report)
        rt.write(out / 'current_status.json', dict(state='completed', **report))
        print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
