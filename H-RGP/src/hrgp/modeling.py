"""Multi-day forecast branches and reservoir release response."""
import copy
import numpy as np
import torch
import routing_model as routing_base
import data_runtime as rt
make_model = routing_base.make_model
batch_for = routing_base.batch_for
finish = routing_base.finish
routing = routing_base.routing

def encode_future(m, batch):
    x, res, y, rain, calendar = batch
    embed = m.time_enc(calendar[:, 0:1], calendar[:, 1:2])
    _, _, fused, hidden = m.basin(x, embed)
    b, n, d = fused.shape
    state = fused.reshape(b * n, d)
    local = []
    for k in range(rain.shape[2]):
        q, state = m.future_gen.forward_step(rain[:, :, k:k + 1], state)
        local.append(q @ m.agg_mask.T)
    return (torch.stack(local, 1), hidden, embed, res, rain.sum(2).mean(1, keepdim=True), y)

def warmup(m, ds, first_targets):
    return routing_base.warmup(m, ds, np.asarray(first_targets, dtype=int) - ds.lead + 1)

def block_prediction(m, batch, water, streams, steps, valid=None, check=False):
    local, hidden, embed, res, rain_sum, y = encode_future(m, batch)
    horizon = local.shape[1]
    local = local.reshape(streams, steps, horizon, 15)
    if valid is None:
        valid = torch.ones(streams, steps, dtype=torch.bool, device=local.device)
    flows = []
    maximum = 0.0
    for day in range(steps):
        active = valid[:, day].nonzero(as_tuple=True)[0]
        out = local.new_zeros(streams, 15)
        if len(active):
            branch = water[active]
            committed = None
            for k in range(horizon):
                branch, q, balance = routing.route(m.vr_net, local[active, day, k], branch, check)
                if k == 0:
                    committed = branch
                maximum = max(maximum, balance)
            water = water.index_copy(0, active, committed)
            out = out.index_copy(0, active, q)
        flows.append(out)
    routed = torch.stack(flows, 1).reshape(-1, 15)
    prediction = finish(m, routed, hidden, embed, res, rain_sum).reshape(streams, steps)
    return (prediction, water, local.sum() * 0.0, maximum)

def evaluate_rolling(m, ds):
    from torch.utils.data import DataLoader
    m.eval()
    device = next(m.parameters()).device
    lead = ds.lead
    days = rt.Windows(ds.x, ds.r, lead, 'train')
    last = int(ds.target_indices[-1]) + 1
    days.target_indices = np.arange(14 + lead, last) if m.portable_variant == 'full' else ds.target_indices.copy()
    with torch.no_grad():
        water = warmup(m, ds, [14 + lead]).cpu()
        router = copy.deepcopy(m.vr_net).cpu()
        values = []
        for batch in DataLoader(days, batch_size=32, shuffle=False):
            z = encode_future(m, [v.to(device) for v in batch])
            values.append(tuple((v.cpu() for v in z[:-1])))
        local, hidden, embed, res, rain_sum = [torch.cat([v[i] for v in values]) for i in range(5)]
        wanted = set(map(int, ds.target_indices))
        selected = []
        flows = []
        time_values = []
        maximum = 0.0
        for index, target in enumerate(days.target_indices):
            branch = water
            committed = None
            horizon = lead if int(target) in wanted else 1
            for k in range(horizon):
                branch, q, balance = routing.route(router, local[index:index + 1, k], branch, check=int(target) in wanted)
                if k == 0:
                    committed = branch
                if int(target) in wanted:
                    maximum = max(maximum, balance)
            water = committed
            if int(target) in wanted:
                selected.append(index)
                flows.append(q)
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
    diag = dict(ablation_arm=m.ablation_arm, grid_context_correction_active=m.hrc_enabled, lead_days=lead, continuous=m.portable_variant == 'full', window_reset=False, state_commitment='First forecast day only; terminal L-day state never committed', history_information_end='target minus lead', frontend=dict(m.basin.frontend_info), physical_state_active=m.portable_variant == 'full', initial_warmup_days=14 if m.portable_variant == 'full' else 0, max_relative_node_balance=maximum, zero_days=int((p == 0).sum()), negative_days=int((p < 0).sum()), nonfinite=int((~np.isfinite(p)).sum()), solver='daily_conditioned_single_store_analytic', path_budget=routing.describe(router), daily_time_statistics=dict(n=len(time_values), mean=float(np.mean(time_values)), std=float(np.std(time_values)), minimum=float(np.min(time_values)), maximum=float(np.max(time_values))) if time_values else None, time_statistics_scope='Terminal forecast-day scale only per scored origin, not mixed with internal days', analytic_scope='Exact single-node daily inflow; daily mean cascade approximation', finite_warmup_limitation='14days does not guarantee initial-state washout')
    return (rt.metrics(p, o), p, o, diag)

class ReservoirResponse(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(8, 64, 2, batch_first=True, dropout=0.2)
        self.inflow_proj = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.ReLU())
        self.fc = torch.nn.Sequential(torch.nn.Linear(64 + 8 + 64 + 1 + 16, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))

    def forward(self, res, inflow, rain, calendar):
        _, (h, _) = self.lstm(res)
        context = torch.cat([h[-1], res[:, -1], self.inflow_proj(inflow[:, None]), rain, calendar], 1)
        return (inflow + self.fc(context).squeeze(1)).clamp(0, 1000000.0)

def make_model(mode, variant, frontend='local_mlp', travel_cap=8.0, initial_time=1.0, time_policy='flow'):
    assert variant in ['full', 'no_routing']
    m = routing_base.make_model(mode, variant, frontend, travel_cap, initial_time, time_policy)
    fraction = np.load(__import__('pathlib').Path(__file__).parent / 'upstream_grid_fraction.npy')
    m.register_buffer('upstream_grid_fraction', torch.tensor(fraction, dtype=torch.float32, device=m.agg_mask.device))
    m.register_buffer('interval_grid_fraction', 1 - m.upstream_grid_fraction)
    m.agg_mask.mul_(m.upstream_grid_fraction[None, :])
    m.vr_net.reference_flows[-1] = rt.INQ95
    m.real_res = ReservoirResponse().to(m.agg_mask.device)
    m.trace_components = False
    m.component_chunks = []
    return m

def encode_future(m, batch):
    x, res, y, rain, calendar = batch
    embed = m.time_enc(calendar[:, 0:1], calendar[:, 1:2])
    _, _, fused, _ = m.basin(x, embed)
    b, n, d = fused.shape
    state = fused.reshape(b * n, d)
    local = []
    interval = []
    for k in range(rain.shape[2]):
        q, state = m.future_gen.forward_step(rain[:, :, k:k + 1], state)
        local.append(q @ m.agg_mask.T)
        interval.append((q * m.interval_grid_fraction[None, :]).sum(1))
    interval = torch.stack(interval, 1)[:, -1:]
    return (torch.stack(local, 1), interval, embed, res, rain.sum(2).mean(1, keepdim=True), y)

def finish(m, flow, interval, embed, res, rain_sum):
    inflow = flow[:, -1]
    outflow = m.real_res(res, inflow, rain_sum, embed)
    m.latest_inflow = inflow
    m.latest_outflow = outflow
    m.latest_interval = interval[:, 0]
    if m.trace_components:
        m.component_chunks.append(torch.stack([inflow, outflow, interval[:, 0]], 1).detach().cpu())
    return (outflow + interval[:, 0]).clamp(0, 1000000.0)
_evaluate_flow = evaluate_rolling

def evaluate_rolling(m, ds):
    m.component_chunks = []
    m.trace_components = True
    try:
        score, p, o, diag = _evaluate_flow(m, ds)
    finally:
        m.trace_components = False
    arrays = torch.cat(m.component_chunks).numpy()
    assert len(arrays) == len(ds)
    m.latest_evaluation_components = arrays
    mask = rt.INQ_MASK[ds.target_indices]
    observed = rt.INQ_TARGET[ds.target_indices]
    diag['inflow_metrics'] = rt.metrics(arrays[mask, 0], observed[mask]) if mask.sum() > 2 else None
    diag['frontend']['grid_context_correction_active'] = False
    diag['grid_context_correction_active'] = False
    diag['release_response_active'] = True
    diag['frontend']['limitation'] = 'Reservoir release is estimated from historical reservoir inputs, predicted inflow, precipitation and calendar.'
    diag['inflow_supervision'] = 'Masked inflow supervision with weight 1; target inflow is never a forecast input'
    diag['boundary'] = 'Upstream subgrid fractions at Sanjiangkou DEM proxy; reach13 shortened to dam; downstream interval generated separately'
    diag['operation_scope'] = 'Statistical release response; not certified reservoir-volume conservation or optimal dispatch'
    diag['operation_inputs'] = 'Past15day8channelreservoir; predicted inflow; forecastP; known calendar. No global grid latent and no historical ShimenQ.'
    diag['downstream_interval_scope'] = 'Terminal forecast-day generated interval runoff joins terminal release instantly at daily resolution; no calibrated interval travel time for any lead'
    return (score, p, o, diag)
