"""Differentiable daily storage and release routing."""
import math, types
import torch
from torch import nn

def release(water, inflow_volume, k):
    decay = torch.exp(-k)
    retention = -torch.expm1(-k) / k
    stored = water * decay + inflow_volume * retention
    q = (water + inflow_volume - stored) / 86400.0
    return (q, stored)

def configure(model, variant, lengths, travel_cap=4.0, initial_time=2.0, time_policy='flow'):
    net = model.vr_net
    for p in net.parameters():
        p.requires_grad_(False)
    assert travel_cap in [4.0, 8.0]
    net.travel_min = 0.75
    net.travel_cap = float(travel_cap)
    assert net.travel_min < initial_time < net.travel_cap
    net.initial_time = float(initial_time)
    initial_fraction = (initial_time - net.travel_min) / (net.travel_cap - net.travel_min)
    net.total_time_raw = nn.Parameter(torch.tensor(math.log(initial_fraction / (1 - initial_fraction)), device=model.agg_mask.device), requires_grad=variant == 'full')
    assert time_policy in ['flow', 'flow_storage']
    net.time_policy = time_policy
    net.time_condition_raw = nn.Parameter(torch.zeros(1 if time_policy == 'flow' else 2, device=model.agg_mask.device), requires_grad=variant == 'full')
    lengths = torch.as_tensor(lengths, dtype=torch.float32, device=model.agg_mask.device)
    assert lengths.shape == (14,) and lengths.min() > 0
    distance = model.vr_net.river_features
    downstream = {u: n for n in net.sorted_nodes for u in net.upstream_map[n]}
    paths = []
    path_lengths = []
    for source in range(14):
        nodes = []
        n = source
        while n != 14:
            nodes.append(n)
            n = downstream[n]
        paths.append(nodes)
        path_lengths.append(float(lengths[nodes].sum()))
    max_distance = max(path_lengths)
    net.register_buffer('reach_fraction_of_longest_path', lengths / max_distance)
    net.register_buffer('simple_reach_lengths_m', lengths)
    net.simple_paths = paths
    net.simple_max_path_m = max_distance
    adj = torch.zeros(15, 15, device=model.agg_mask.device)
    ancestry = torch.eye(15, device=model.agg_mask.device)
    for n in net.sorted_nodes:
        for u in net.upstream_map[n]:
            adj[n, u] = 1
            ancestry[n] += ancestry[u]
    net.register_buffer('simple_adjacency', adj)
    net.register_buffer('direct_ancestry', ancestry)
    depth = {n: 0 for n in net.sorted_nodes}
    for n in net.sorted_nodes:
        if net.upstream_map[n]:
            depth[n] = 1 + max((depth[u] for u in net.upstream_map[n]))
    net.simple_groups = [[n for n in net.sorted_nodes if depth[n] == d] for d in range(max(depth.values()) + 1)]
    net.simple_variant = variant

def daily_total(raw, coeff, local, water, reference_flow, fractions, minimum, cap, policy):
    base = minimum + (cap - minimum) * torch.sigmoid(raw)
    flow_context = torch.tanh(torch.log1p(local.sum(1) / reference_flow) - math.log(2.0))
    stock_context = torch.tanh(torch.log1p(water[:, :14].sum(1) / (reference_flow * 86400.0 * base)) - math.log(2.0))
    context = torch.stack([flow_context] if policy == 'flow' else [flow_context, stock_context], 1)
    delta = (0.5 * torch.tanh(coeff) * context).sum(1)
    return (minimum + (cap - minimum) * torch.sigmoid(raw + delta), context)

def route(net, local, water, check=False):
    if net.simple_variant == 'no_routing':
        return (water, local @ net.direct_ancestry.T, 0.0)
    total, context = daily_total(net.total_time_raw, net.time_condition_raw, local, water, net.reference_flows[-1], net.reach_fraction_of_longest_path, net.travel_min, net.travel_cap, net.time_policy)
    net.last_total_time = total.detach()
    net.last_time_context = context.detach()
    tau = total[:, None] * net.reach_fraction_of_longest_path[None, :]
    rates = 1.0 / tau
    outs = {}
    end = {}
    for group in net.simple_groups:
        upstream = torch.stack([sum((outs[u] for u in net.upstream_map[n]), torch.zeros_like(local[:, n], dtype=torch.double)) for n in group], 1)
        incoming = local[:, group].double() + upstream
        for j, n in enumerate(group):
            if n == 14:
                outs[n] = incoming[:, j]
                end[n] = torch.zeros_like(incoming[:, j])
            else:
                q, stored = release(water[:, n].double(), incoming[:, j] * 86400.0, rates[:, n].double())
                outs[n] = q
                end[n] = stored
    flow = torch.stack([outs[n] for n in net.sorted_nodes], 1).to(local.dtype)
    new = torch.stack([end[n] for n in net.sorted_nodes], 1).to(water.dtype)
    residual = 0.0
    if check:
        incoming = (local + flow @ net.simple_adjacency.T) * 86400.0
        residual = float(((water + incoming - new - flow * 86400.0).abs() / (water + incoming).clamp_min(1.0)).max().detach())
        assert residual < 1e-05 and new.min() >= 0 and (flow.min() >= 0) and torch.isfinite(new).all() and torch.isfinite(flow).all()
    return (new, flow, residual)

def describe(net):
    total = float((net.travel_min + (net.travel_cap - net.travel_min) * torch.sigmoid(net.total_time_raw)).detach())
    tau = (net.reach_fraction_of_longest_path.detach().cpu() * total).tolist()
    sums = [sum((tau[n] for n in path)) for path in net.simple_paths]
    assert max(sums) <= net.travel_cap + 1e-05
    return dict(time_policy=net.time_policy, condition_coefficients=(0.5 * torch.tanh(net.time_condition_raw.detach())).cpu().tolist(), reference_time_only=True, initial_time_days=net.initial_time, total_longest_path_mean_days=total, admissible_total_days=[net.travel_min, net.travel_cap], reach_tau_days=tau, path_tau_sums_days=sums, max_geometry_path_m=net.simple_max_path_m, outlet_additional_storage=False, learned_time_active=net.simple_variant == 'full', interpretation='Daily instantaneous sum of reach time scales bounded by cap; coefficients frozen within each day. Reference taus are not a fitted fixed kernel. Time-varying routing has no global mean-travel-time or hard-arrival guarantee; not measured river speed.')
