"""Storage-routing parameterization and execution helpers."""
import math, types
import torch
from torch import nn

class ShrunkSAP(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.baseline = nn.Parameter(torch.tensor([math.log(0.45 / 1.5)] if mode == 'k_only' else [math.log(0.45 / 1.5), 0.0]))
        self.residual = nn.Sequential(nn.Linear(3 if mode == 'anchored' else 6, 16), nn.Tanh(), nn.Linear(16, 1 if mode == 'k_only' else 2))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.last_penalty = None

    def forward(self, sf, sm, q=None, water=None, slow=None, ref=None):
        if self.mode == 'constant':
            raw = self.baseline.expand(*sm.shape, 2)
            self.last_penalty = raw.sum() * 0.0
        else:
            parts = [sf, sm[..., None].clamp(-5.0, 5.0)]
            if self.mode in ('state', 'k_only'):
                parts.extend([torch.log1p(q.clamp_min(0) / (ref.squeeze(-1) / 86400.0))[..., None], torch.log1p(water.clamp_min(0) / ref), torch.log1p(slow.clamp_min(0) / ref)])
            delta = 0.5 * torch.tanh(self.residual(torch.cat(parts, -1)))
            raw = self.baseline + delta
            self.last_penalty = delta.square().mean()
        k = 0.05 + 1.95 * torch.sigmoid(raw[..., 0:1])
        return (k, torch.ones_like(k) if self.mode == 'k_only' else 0.5 + torch.sigmoid(raw[..., 1:2]))

def attach_execution(net, variant):
    count = len(net.sorted_nodes)
    assert set(net.sorted_nodes) == set(range(count))
    depth = {n: 0 for n in net.sorted_nodes}
    for n in net.sorted_nodes:
        if net.upstream_map[n]:
            depth[n] = 1 + max((depth[u] for u in net.upstream_map[n]))
    groups = [[n for n in net.sorted_nodes if depth[n] == d] for d in range(max(depth.values()) + 1)]

    def linear(values, layers):
        weight = torch.stack([l.weight for l in layers])
        bias = torch.stack([l.bias for l in layers])
        return (torch.bmm(values.transpose(0, 1), weight.transpose(1, 2)) + bias[:, None, :]).transpose(0, 1)

    def initialize(this, h):
        cells = [this.vr_cells[str(n)] for n in range(count)]
        state = torch.nn.functional.softplus(linear(h, [c.init_net[0] for c in cells]))
        reference = torch.stack([c.reference_m3 for c in cells])[None, :, None]
        state = state * reference
        return {n: (state[:, n, 0:1], state[:, n, 1:2]) for n in range(count)}

    def step(this, Q, states, rain, sm):
        cells = [this.vr_cells[str(n)] for n in range(count)]
        water = torch.stack([states[n][0] for n in range(count)], 1)
        slow = torch.stack([states[n][1] for n in range(count)], 1)
        reference = torch.stack([c.reference_m3 for c in cells])[None, :, None]
        feat = this.river_features[None, :, :].expand(len(Q), -1, -1)
        hidden = torch.tanh(linear(torch.cat([feat, slow / reference], -1), [c.phys_param_net[0] for c in cells]))
        params = torch.sigmoid(linear(hidden, [c.phys_param_net[2] for c in cells]))
        kslow = 0.01 + 0.49 * params[:, :, 1:2]
        ratio = 0.25 * params[:, :, 2:3] * (1 - torch.tanh(slow / reference))
        slow_release = slow * kslow / 86400.0
        sf = this.static_sf[None, :, :].expand(len(Q), -1, -1)
        if variant in ('spatial', 'matched_init'):
            raw = this.spatial_param_mlp(torch.cat([sf, sm[:, :, None]], -1))
            k = 0.05 + 1.95 * torch.sigmoid(raw[:, :, 0:1])
            alpha = 0.5 + torch.sigmoid(raw[:, :, 1:2])
        elif variant == 'fixed':
            k = torch.full_like(water, 0.5)
            alpha = torch.ones_like(water)
        elif variant not in ('state', 'k_only'):
            k, alpha = this.sap_policy(sf, sm)
        outs = {}
        new = {}
        penalties = []
        from storage_release import implicit_power_release
        for nodes in groups:
            upstream = torch.stack([sum((outs[u] for u in this.upstream_map[n]), torch.zeros_like(Q[:, n])) for n in nodes], 1)
            q = (Q[:, nodes] + upstream).clamp_min(0)[:, :, None]
            if variant in ('state', 'k_only'):
                kg, ag = this.sap_policy(sf[:, nodes], sm[:, nodes], q.squeeze(-1), water[:, nodes], slow[:, nodes], reference[:, nodes])
                penalties.append(this.sap_policy.last_penalty * len(nodes) / count)
            else:
                kg, ag = (k[:, nodes, :], alpha[:, nodes, :])
            transfer = q * ratio[:, nodes, :]
            slow_new = slow[:, nodes, :] * (1 - kslow[:, nodes, :]) + transfer * 86400.0
            available = water[:, nodes, :] + (q - transfer + slow_release[:, nodes, :]) * 86400.0
            qout, stored = implicit_power_release(available, kg, ag, reference[:, nodes, :])
            for j, n in enumerate(nodes):
                outs[n] = qout[:, j, 0]
                new[n] = (stored[:, j, :], slow_new[:, j, :])
                if cells[n].capture:
                    ref = reference[:, n, :].expand_as(stored[:, j, :])
                    cells[n].trace = {'qin': q[:, j, :].detach(), 'water_old': water[:, n, :].detach(), 'slow_old': slow[:, n, :].detach(), 'transfer': transfer[:, j, :].detach(), 'slow_release': slow_release[:, n, :].detach(), 'available': available[:, j, :].detach(), 'qout': qout[:, j, :].detach(), 'water_new': stored[:, j, :].detach(), 'slow_new': slow_new[:, j, :].detach(), 'k_day': kg[:, j, :].detach(), 'alpha': ag[:, j, :].detach(), 'reference_m3': ref.detach(), 'power_q': (kg[:, j, :] / 86400.0 * ref * (stored[:, j, :] / ref).clamp_min(0).pow(ag[:, j, :])).detach()}
        this.sap_penalty = sum(penalties) if penalties else this.sap_policy.last_penalty if hasattr(this, 'sap_policy') else Q.sum() * 0.0
        return (torch.stack([outs[n] for n in this.sorted_nodes], -1), new)
    net.initialize_states = types.MethodType(initialize, net)
    net.forward_step = types.MethodType(step, net)
