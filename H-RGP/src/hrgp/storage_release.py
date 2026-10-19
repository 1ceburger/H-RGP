"""Hydrological model implementation."""
import torch
from torch import nn
import torch.nn.functional as F

def _root_newton(a, k, p, iterations=5):
    a = a.clamp_min(0)
    upper = torch.minimum(a, (a / k).clamp_min(0).pow(1 / p))
    u = torch.where(p < 1, upper * 0.25, upper)
    for _ in range(iterations):
        z = u.clamp_min(1e-60)
        f = z + k * z.pow(p) - a
        d = 1 + k * p * z.pow(p - 1)
        proposal = z - f / d
        u = torch.where(proposal > 0, proposal, z * 0.5)
    return torch.where(a > 0, u, torch.zeros_like(u))

def _solve_root(a, k, p):
    return _root_newton(a, k, p)

class _ImplicitPower(torch.autograd.Function):

    @staticmethod
    def forward(ctx, available, k_day, alpha):
        a, k, p = [t.double() for t in (available, k_day, alpha)]
        a = a.clamp_min(0)
        u = _solve_root(a, k, p)
        ctx.save_for_backward(u, k, p, a)
        ctx.input_dtype = available.dtype
        return u.to(available.dtype)

    @staticmethod
    def backward(ctx, grad):
        u, k, p, a = ctx.saved_tensors
        z = u.clamp_min(1e-60)
        den = 1 + k * p * z.pow(p - 1)
        g = grad.double()
        positive = a > 0
        at_zero = torch.where(p < 1, torch.zeros_like(p), torch.where(p == 1, 1 / (1 + k), torch.ones_like(p)))
        ga = torch.where(positive, g / den, g * at_zero)
        gk = torch.where(positive, -g * z.pow(p) / den, torch.zeros_like(g))
        gp = torch.where(positive, -g * k * z.pow(p) * torch.log(z) / den, torch.zeros_like(g))
        return tuple((v.to(ctx.input_dtype) for v in (ga, gk, gp)))

def implicit_power_release(available_m3, k_day, alpha, reference_m3):
    normalized = available_m3 / reference_m3
    stored = _ImplicitPower.apply(normalized, k_day, alpha) * reference_m3
    q = (available_m3 - stored) / 86400.0
    return (q, stored)

class ConservativePowerCell(nn.Module):

    def __init__(self, feat_dim, reference_m3):
        super().__init__()
        self.register_buffer('reference_m3', torch.tensor(float(reference_m3)))
        self.phys_param_net = nn.Sequential(nn.Linear(feat_dim + 1, 32), nn.Tanh(), nn.Linear(32, 3), nn.Sigmoid())
        self.init_net = nn.Sequential(nn.Linear(128, 2), nn.Softplus())
        self.capture = False

    def forward(self, q_in, river_feat, state, k_day, alpha):
        water, slow = state
        ref = self.reference_m3.to(water)
        params = self.phys_param_net(torch.cat([river_feat, slow / ref], -1))
        k_slow = 0.01 + 0.49 * params[:, 1:2]
        transfer_ratio = 0.25 * params[:, 2:3] * (1 - torch.tanh(slow / ref))
        transfer = q_in[:, None] * transfer_ratio
        slow_release = slow * k_slow / 86400.0
        slow_new = slow * (1 - k_slow) + transfer * 86400.0
        available = water + (q_in[:, None] - transfer + slow_release) * 86400.0
        q, water_new = implicit_power_release(available, k_day, alpha, ref)
        if self.capture:
            self.trace = {key: value.detach() for key, value in {'qin': q_in[:, None], 'water_old': water, 'slow_old': slow, 'transfer': transfer, 'slow_release': slow_release, 'available': available, 'qout': q, 'water_new': water_new, 'slow_new': slow_new, 'k_day': k_day, 'alpha': alpha, 'reference_m3': ref.expand_as(q), 'power_q': k_day / 86400.0 * ref * (water_new / ref).clamp_min(0).pow(alpha)}.items()}
        return (q.squeeze(1), (water_new, slow_new))
