"""Independent training and evaluation for a fixed lead time and seed."""
from pathlib import Path
import argparse, json, time, os, random, hashlib, contextlib, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import data_runtime as rt
import modeling as md
from ema_weights import EMAWeights
import history_encoder
history_encoder.install(md)

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)

def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

@contextlib.contextmanager
def output_lock(path):
    f = Path(path).open('a+b')
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    if os.name == 'nt':
        import msvcrt
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('This output directory is already running')
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        yield
    finally:
        f.seek(0)
        if os.name == 'nt':
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

def evaluate(m, ds, mode):
    return md.evaluate_window(m, ds) if mode == 'window' else md.evaluate_rolling(m, ds)

def train_window_epoch(m, loader, opt, lossfn, device):
    m.train()
    total = 0.0
    count = 0
    for batch in loader:
        x, r, y, rain, t = [v.to(device) for v in batch]
        opt.zero_grad(set_to_none=True)
        p = m(x, r, t, rain, lead_time=1)
        loss = lossfn(torch.log1p(p)[:, None], y[:, None]) + 0.01 * m.vr_net.sap_penalty
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        assert torch.isfinite(norm)
        opt.step()
        total += float(loss.detach()) * len(y)
        count += len(y)
    return total / count

def train_rolling_epoch(m, ds, opt, lossfn, device, streams, unroll, ema=None):
    streams = min(streams, len(ds))
    segments = np.array_split(np.arange(len(ds)), streams)
    first = [int(ds.target_indices[s[0]]) for s in segments]
    water = md.warmup(m, ds, first)
    m.train()
    total = 0.0
    count = 0
    for offset in range(0, max(map(len, segments)), unroll):
        length = min(unroll, max(map(len, segments)) - offset)
        valid = np.array([[offset + j < len(s) for j in range(length)] for s in segments])
        indices = np.array([[s[min(offset + j, len(s) - 1)] for j in range(length)] for s in segments])
        batch = md.batch_for(ds, indices, device)
        mask = torch.as_tensor(valid, device=device)
        opt.zero_grad(set_to_none=True)
        p, next_water, penalty, _ = md.block_prediction(m, batch, water, streams, length, mask)
        targets = batch[2].reshape(streams, length)
        loss = lossfn(torch.log1p(p[mask])[:, None], targets[mask, None]) + 0.01 * penalty
        label_indices = ds.target_indices[indices.reshape(-1)]
        observed_inflow = torch.as_tensor(np.nan_to_num(rt.INQ_TARGET[label_indices], nan=0.0), device=device, dtype=p.dtype)
        inflow_mask = torch.as_tensor(rt.INQ_MASK[label_indices], device=device) & mask.reshape(-1)
        if inflow_mask.any():
            criterion_inflow = rt.layers.HydroLoss(alpha=2.0, beta=0.1, q_ref=rt.INQ95).to(device)
            loss = loss + criterion_inflow(torch.log1p(m.latest_inflow[inflow_mask])[:, None], torch.log1p(observed_inflow[inflow_mask])[:, None])
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        assert torch.isfinite(norm)
        opt.step()
        if ema is not None:
            ema.update(m)
        water = next_water.detach()
        n = int(mask.sum())
        total += float(loss.detach()) * n
        count += n
    return total / count

def preflight(m, ds, args, device):
    batch = md.batch_for(ds['val'], np.arange(2), device)
    m.train()
    m.zero_grad(set_to_none=True)
    if args.mode == 'window':
        x, r, y, rain, t = batch
        p = m(x, r, t, rain, lead_time=1)
        penalty = m.vr_net.sap_penalty
    else:
        segments = np.array_split(np.arange(len(ds['train'])), args.streams)
        assert min(map(len, segments)) >= args.unroll, 'Too many streams/unroll for this dataset'
        indices = np.stack([s[:args.unroll] for s in segments])
        batch = md.batch_for(ds['train'], indices, device)
        water = md.warmup(m, ds['train'], ds['train'].target_indices[indices[:, 0]])
        p, _, penalty, balance = md.block_prediction(m, batch, water, args.streams, args.unroll, check=args.variant in ['full', 'no_hrc'])
        y = batch[2].reshape(args.streams, args.unroll)
    loss = (torch.log1p(p) - y).square().mean() + 0.01 * penalty
    loss.backward()
    assert torch.isfinite(p).all() and (p >= 0).all()
    assert all((v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters()))
    return dict(passed=True, mode=args.mode, variant=args.variant, finite_forward_backward=True, no_test_scoring=True, dates_per_block=args.streams * args.unroll if args.mode == 'rolling' else 2, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024 ** 3 if str(device).startswith('cuda') else None)

def make_optimizer(m, multiplier=20.0):
    assert multiplier > 0
    time_parameter = m.vr_net.total_time_raw
    condition = m.vr_net.time_condition_raw
    ordinary = [p for p in m.parameters() if p.requires_grad and p is not time_parameter and (p is not condition)]
    groups = [dict(params=ordinary, lr=0.0005, name='encoder_generator_reservoir')]
    if time_parameter.requires_grad:
        groups.append(dict(params=[time_parameter, condition], lr=0.0005 * multiplier, name='routing_base_and_daily_condition'))
    return torch.optim.AdamW(groups, lr=0.0005, weight_decay=0.01)

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Directory containing the model-ready input files')
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['rolling'], default='rolling')
    p.add_argument('--variant', choices=['full', 'no_routing', 'no_hrc', 'no_routing_no_hrc'], default='full')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lead', type=int, choices=[1, 3, 7], default=1, help='Independently fitted target lead; future branch commits day1 state only')
    p.add_argument('--max-epochs', type=int, default=160, help='0 means no epoch cap; patience still applies')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--min-delta', type=float, default=1e-05)
    p.add_argument('--batch-size', type=int, default=32, help='Window mode only')
    p.add_argument('--streams', type=int, default=8)
    p.add_argument('--unroll', type=int, default=4)
    p.add_argument('--frontend', choices=['local_mlp', 'local_cnn', 'local_lstm', 'mean_gnn'], default='local_mlp')
    p.add_argument('--travel-cap', type=float, choices=[4.0, 8.0], default=8.0)
    p.add_argument('--initial-time', type=float, choices=[1.0, 2.0, 4.0], default=1.0)
    p.add_argument('--time-policy', choices=['flow', 'flow_storage'], default='flow')
    p.add_argument('--routing-lr-multiplier', type=float, default=20.0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--evaluate-test', action='store_true')
    p.add_argument('--allow-data-mismatch', action='store_true', help='Explicitly allow data not matching the supplied reference hashes')
    args = p.parse_args()
    assert args.patience > 0 and args.max_epochs >= 0 and (args.streams > 0) and (args.unroll > 0)
    return args

def main():
    args = arguments()
    assert args.lead in [1, 3, 7] and args.variant in ['full', 'no_routing'], 'Selected causal hierarchy, independent leads1/3/7'
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if str(device).startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable; install matching PyTorch or use --device cpu')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output / 'run.lock'):
        hashes = {name: sha(Path(args.data) / name) for name in rt.PROFILE['required_files']}
        mismatched = [name for name, digest in hashes.items() if digest != rt.PROFILE['reference_hashes'][name]]
        if mismatched and (not args.allow_data_mismatch):
            raise ValueError('Input file hashes do not match the study configuration: ' + str(mismatched) + '. Check files before using --allow-data-mismatch.')
        torch.set_num_threads(2 if device == 'cpu' else 4)
        ds = rt.load_data(args.data, output, device)
        x, r = (ds['train'].x, ds['train'].r)
        ds = {split: rt.Windows(x, r, args.lead, split) for split in ['train', 'val']}
        config = dict(experiment='H-RGP', EMA_decay=0.99, EMA_scope='Trainable parameters averaged each optimizer step; fixed buffers exact; val/LR/earlystop/best use EMA model', inflow_auxiliary_weight=1.0, inflow_labels='masked actual INQ; no mean filling', inflow_reference95=rt.INQ95, operation_global_grid_context=False, historical_ShimenQ_input=False, grid_history_days=15, reservoir_history_days=15, history_GRU64=True, time_policy=args.time_policy, initial_time=args.initial_time, routing_lr_multiplier=args.routing_lr_multiplier, frontend=args.frontend, travel_cap=args.travel_cap, mode=args.mode, variant=args.variant, seed=args.seed, solver='paper_single_store_analytic_length_distributed_total_time', internal_substeps=1, patience=args.patience, max_epochs=args.max_epochs, min_delta=args.min_delta, batch_size=args.batch_size, streams=args.streams, unroll=args.unroll, data_hashes=hashes, source_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}, statistical_release_response=True, no_rain_noise=True, lead=args.lead, history_days=15, loss='OutletHydroLoss + maskedInflowHydroLoss; separate training Q95', train_end=str(rt.DATES[rt.TRAIN_END - 1].date()), val_dates=[str(rt.DATES[ds['val'].target_indices[0]].date()), str(rt.DATES[ds['val'].target_indices[-1]].date())], validation_n=len(ds['val']), test_scoring_requested=args.evaluate_test, state_protocol='Each stream warms once through origin; L-day forecast branches from carried origin storage; only first forecast-day state committed for next origin; TBPTT; validation continuous from first origin', history_information_end='target minus lead', future_drivers='Precipitation and calendar; no future observed hydrological states', lead_training='Each lead/seed has separate initialization, optimization and early stopping; no extracted horizons or relabeled lead1 outputs', calendar='Target calendar repeated during warmup' if args.mode == 'window' else 'Actualhistoricaltargetdatesineachwarmupstep')
        if (output / 'protocol.json').exists():
            old = json.loads((output / 'protocol.json').read_text(encoding='utf8'))
            keys = ['lead', 'time_policy', 'initial_time', 'routing_lr_multiplier', 'frontend', 'travel_cap', 'mode', 'variant', 'seed', 'solver', 'data_hashes', 'source_hashes']
            if not args.eval_only:
                keys += ['patience', 'min_delta', 'batch_size', 'streams', 'unroll']
            assert all((old[k] == config[k] for k in keys)), 'Output belongs to different code/data/model; use another output directory'
        else:
            write(output / 'protocol.json', config)
        write(output / 'runtime_options.json', dict(vars(args), resolved_device=device))
        rt.seed_all(args.seed)
        m = md.make_model(args.mode, args.variant, args.frontend, args.travel_cap, args.initial_time, args.time_policy)
        ema = EMAWeights(m, 0.99)
        if args.preflight_only:
            result = preflight(m, ds, args, device)
            write(output / 'preflight.json', result)
            print(json.dumps(result), flush=True)
            return
        if not args.eval_only and (output / 'best.pth').exists() and (not args.resume):
            raise RuntimeError('Existing weights found. Use --resume or a fresh --output.')
        if not args.eval_only:
            rt.seed_all(10000 + args.seed)
            criterion = rt.layers.HydroLoss(alpha=2.0, beta=0.1, q_ref=rt.Q95).to(device)
            opt = make_optimizer(m, args.routing_lr_multiplier)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
            generator = torch.Generator().manual_seed(args.seed)
            loader = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True, generator=generator)
            best = -np.inf
            best_epoch = 0
            bad = 0
            epoch = 0
            history = []
            if args.resume:
                checkpoint = torch.load(output / 'last.pth', map_location=device, weights_only=False)
                m.load_state_dict(checkpoint['model'])
                ema.load_state_dict(checkpoint['ema'])
                opt.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                best = checkpoint['best']
                best_epoch = checkpoint['best_epoch']
                bad = checkpoint['bad']
                epoch = checkpoint['epoch']
                history = checkpoint['history']
                torch.set_rng_state(checkpoint['torch_rng'].cpu())
                np.random.set_state(checkpoint['numpy_rng'])
                random.setstate(checkpoint['python_rng'])
                generator.set_state(checkpoint['loader_rng'].cpu())
                if str(device).startswith('cuda') and checkpoint.get('cuda_rng'):
                    torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
            started = time.time()
            try:
                while (args.max_epochs == 0 or epoch < args.max_epochs) and bad < args.patience:
                    if (output / 'STOP_REQUESTED').exists():
                        write(output / 'current_status.json', dict(state='stopped_by_user', last_completed_epoch=epoch))
                        print('STOP_REQUESTED: stopping at epoch boundary; saved checkpoints retained.', flush=True)
                        return
                    epoch += 1
                    t0 = time.time()
                    if args.mode == 'window':
                        loss = train_window_epoch(m, loader, opt, criterion, device)
                    else:
                        loss = train_rolling_epoch(m, ds['train'], opt, criterion, device, args.streams, args.unroll, ema=ema)
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    training_seconds = time.time() - t0
                    validation_start = time.time()
                    score, _, _, diag = evaluate(ema.model, ds['val'], args.mode)
                    scheduler.step(score['NSE'])
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    validation_seconds = time.time() - validation_start
                    if score['NSE'] > best + args.min_delta:
                        best = score['NSE']
                        best_epoch = epoch
                        bad = 0
                        save(output / 'best.pth', ema.model.state_dict())
                    else:
                        bad += 1
                    row = dict(time_policy=args.time_policy, daily_time_statistics=diag['daily_time_statistics'], frontend=args.frontend, travel_cap=args.travel_cap, variant=args.variant, epoch=epoch, val_NSE=score['NSE'], val_KGE=score['KGE'], val_RMSE_m3_s=score['RMSE_m3_s'], best_val_NSE=best, best_epoch=best_epoch, no_improvement=bad, train_loss=loss, training_seconds=training_seconds, validation_seconds=validation_seconds, learning_rate=opt.param_groups[0]['lr'], routing_learning_rate=opt.param_groups[-1]['lr'] if args.variant in ['full', 'no_hrc'] else 0.0, total_time_days=md.routing.describe(m.vr_net)['total_longest_path_mean_days'] if args.variant in ['full', 'no_hrc'] else 0.0, epoch_seconds=time.time() - t0, total_seconds=time.time() - started)
                    history.append(row)
                    rt.pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
                    write(output / 'current_status.json', dict(row, state='training', pid=os.getpid()))
                    save(output / 'last.pth', dict(model=m.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, best=best, best_epoch=best_epoch, bad=bad, history=history, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(), loader_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None))
                    print(json.dumps(row), flush=True)
            except KeyboardInterrupt:
                save(output / 'interrupted_weights.pth', m.state_dict())
                save(output / 'interrupted_ema.pth', ema.state_dict())
                write(output / 'current_status.json', dict(state='interrupted', epoch=epoch))
                print('Interruptedweights saved; --resume restarts from last completed epoch.', flush=True)
                return
        if not (output / 'best.pth').exists():
            raise RuntimeError('No completed best checkpoint; see training status')
        m.load_state_dict(torch.load(output / 'best.pth', map_location=device, weights_only=True))
        splits = ['val'] + (['test'] if args.evaluate_test else [])
        report = dict(lead=args.lead, mode=args.mode, frontend=args.frontend, travel_cap=args.travel_cap, variant=args.variant, seed=args.seed, best_checkpoint_sha256=sha(output / 'best.pth'), splits={})
        for split in splits:
            dataset = ds[split] if split == 'val' else rt.Windows(ds['train'].x, ds['train'].r, args.lead, 'test')
            scores, p, o, diag = evaluate(m, dataset, args.mode)
            assert np.isfinite(p).all() and (p >= 0).all()
            components = m.latest_evaluation_components
            rt.pd.DataFrame(dict(date=rt.DATES[dataset.target_indices], observed_m3_s=o, predicted_m3_s=p, predicted_inflow_m3_s=components[:, 0], predicted_reservoir_outflow_m3_s=components[:, 1], predicted_interval_m3_s=components[:, 2], observed_inflow_m3_s=rt.INQ_TARGET[dataset.target_indices], inflow_label_valid=rt.INQ_MASK[dataset.target_indices])).to_csv(output / f'{split}_predictions.csv', index=False)
            report['splits'][split] = dict(metrics=scores, diagnostics=diag)
        write(output / 'evaluation_metrics.json', report)
        write(output / 'current_status.json', dict(state='completed', **report))
        print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
