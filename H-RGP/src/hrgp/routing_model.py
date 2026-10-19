"""Routing model assembly, batching and state warmup."""
import copy
import numpy as np
import torch
import data_runtime as rt
import grid_frontend as front
import routing as routing

def make_model(mode, variant, frontend='local_mlp', travel_cap=4.0, initial_time=2.0, time_policy='flow'):
    assert mode == 'rolling' and variant in ['full', 'no_routing', 'no_hrc', 'no_routing_no_hrc']
    arm = variant
    variant = 'full' if arm in ['full', 'no_hrc'] else 'no_routing'
    m = rt.make_model()
    m.portable_mode = mode
    m.portable_variant = variant
    m.frontend = frontend
    m.ablation_arm = arm
    m.hrc_enabled = arm in ['full', 'no_routing']
    if not m.hrc_enabled:
        for parameter in m.real_res.parameters():
            parameter.requires_grad_(False)
    front.configure(m, frontend, rt.COLS)
    routing.configure(m, variant, rt.lengths(), travel_cap, initial_time, time_policy)
    return m

def encode(m, batch):
    x, r, y, rain, t = batch
    embed = m.time_enc(t[:, 0:1], t[:, 1:2])
    _, _, fused, hidden = m.basin(x, embed)
    b, n, d = fused.shape
    q, _ = m.future_gen.forward_step(rain[:, :, 0:1], fused.reshape(b * n, d))
    return (q @ m.agg_mask.T, hidden, embed, r, rain.sum(2).mean(1, keepdim=True), y)

def finish(m, flow, hidden, embed, res, rain_sum):
    if not m.hrc_enabled:
        return flow[:, -1].clamp(0, 1000000.0)
    return (flow[:, -1] + m.real_res(res, flow, rain_sum, hidden, embed)).clamp(0, 1000000.0)

def batch_for(ds, indices, device):
    samples = [ds[int(i)] for i in np.asarray(indices).reshape(-1)]
    return [torch.stack([s[k] for s in samples]).to(device) for k in range(5)]

def warmup(m, ds, first_targets):
    first = np.asarray(first_targets, dtype=int)
    device = next(m.parameters()).device
    water = torch.zeros(len(first), 15, device=device)
    if m.portable_variant == 'no_routing':
        return water
    old = m.training
    m.eval()
    history = torch.stack([ds.x[t - 15:t] for t in first]).to(device)
    with torch.no_grad():
        for day in range(1, 15):
            calendar = ds.tf[torch.as_tensor(first - 15 + day, dtype=torch.long)].to(device)
            embed = m.time_enc(calendar[:, 0:1], calendar[:, 1:2])
            _, _, fused, _ = m.basin(history[:, :day], embed)
            b, n, d = fused.shape
            q, _ = m.future_gen.forward_step(history[:, day, :, 0:1], fused.reshape(b * n, d))
            water, _, _ = routing.route(m.vr_net, q @ m.agg_mask.T, water)
    m.train(old)
    return water.detach()

def block_prediction(m, batch, water, streams, steps, valid=None, check=False):
    local, hidden, embed, res, rain_sum, y = encode(m, batch)
    local = local.reshape(streams, steps, 15)
    if valid is None:
        valid = torch.ones(streams, steps, dtype=torch.bool, device=local.device)
    flows = []
    maximum = 0.0
    for day in range(steps):
        active = valid[:, day].nonzero(as_tuple=True)[0]
        out = local.new_zeros(streams, 15)
        if len(active):
            new, q, balance = routing.route(m.vr_net, local[active, day], water[active], check)
            water = water.index_copy(0, active, new)
            out = out.index_copy(0, active, q)
            maximum = max(maximum, balance)
        flows.append(out)
    routed = torch.stack(flows, 1).reshape(-1, 15)
    prediction = finish(m, routed, hidden, embed, res, rain_sum).reshape(streams, steps)
    return (prediction, water, local.sum() * 0.0, maximum)

def evaluate_rolling(m, ds):
    from torch.utils.data import DataLoader
    m.eval()
    device = next(m.parameters()).device
    all_days = rt.Windows(ds.x, ds.r, 1, 'train')
    last = int(ds.target_indices[-1]) + 1
    all_days.target_indices = np.arange(15, last) if m.portable_variant == 'full' else ds.target_indices.copy()
    with torch.no_grad():
        water = warmup(m, ds, [15]).cpu()
        router = copy.deepcopy(m.vr_net).cpu()
        values = []
        for batch in DataLoader(all_days, batch_size=32, shuffle=False):
            z = encode(m, [v.to(device) for v in batch])
            values.append(tuple((v.cpu() for v in z[:-1])))
        local, hidden, embed, res, rain_sum = [torch.cat([v[i] for v in values]) for i in range(5)]
        wanted = set(map(int, ds.target_indices))
        selected = []
        flows = []
        time_values = []
        max_balance = 0.0
        for index, target in enumerate(all_days.target_indices):
            water, q, balance = routing.route(router, local[index:index + 1], water, check=int(target) in wanted)
            if int(target) in wanted:
                selected.append(index)
                flows.append(q)
                max_balance = max(max_balance, balance)
                if m.portable_variant == 'full':
                    time_values.append(float(router.last_total_time[0]))
        selected = torch.as_tensor(selected, dtype=torch.long)
        routed = torch.cat(flows)
        pred = []
        for start in range(0, len(selected), 32):
            pos = selected[start:start + 32]
            pred.append(finish(m, routed[start:start + 32].to(device), hidden[pos].to(device), embed[pos].to(device), res[pos].to(device), rain_sum[pos].to(device)).cpu().numpy())
    p = np.concatenate(pred)
    o = torch.expm1(ds.y[ds.target_indices]).numpy()
    assert np.isfinite(p).all() and (p >= 0).all()
    diag = dict(ablation_arm=m.ablation_arm, grid_context_correction_active=m.hrc_enabled, frontend=dict(m.basin.frontend_info), continuous=m.portable_variant == 'full', window_reset=False, physical_state_active=m.portable_variant == 'full', initial_warmup_days=14 if m.portable_variant == 'full' else 0, max_relative_node_balance=max_balance, zero_days=int((p == 0).sum()), negative_days=int((p < 0).sum()), nonfinite=int((~np.isfinite(p)).sum()), solver='daily_conditioned_single_store_analytic', path_budget=routing.describe(router), daily_time_statistics=dict(n=len(time_values), mean=float(np.mean(time_values)), std=float(np.std(time_values)), minimum=float(np.min(time_values)), maximum=float(np.max(time_values))) if time_values else None, analytic_scope='Exact single-node constant daily inflow; topological daily-mean cascade, not exact continuously coupled river ODE', finite_warmup_limitation='14days does not guarantee removal of initial-state influence')
    return (rt.metrics(p, o), p, o, diag)
