"""Temporal grid history encoding."""
import types
import torch
from torch import nn
STATIC = ('elev', 'slope', 'ksat', 'poro', 'cn', 'fc', 'wp')

def attach(m):
    assert m.frontend == 'local_mlp'
    basin = m.basin
    basin.local_history_gru = nn.GRU(64, 64, batch_first=True).to(m.agg_mask.device)
    import data_runtime as rt
    static_indices = [rt.COLS.index(c) for c in STATIC]

    def forward(this, xg, time_embed):
        x = xg * this.frontend_gate[None, None, None, :]
        b, t, n, d = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().reshape(b * n, t, d)
        pointwise = this.local_mlp(seq)
        _, h = this.local_history_gru(pointwise)
        temporal = h[-1].reshape(b, n, 64)
        static = this.local_static(x[:, 0, :, static_indices])
        path = static.new_zeros(b, n, 32)
        fused = this.fusion(torch.cat([static, temporal, path, time_embed[:, None, :].expand(-1, n, -1)], -1))
        mask = this.agg_mask / this.agg_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = torch.einsum('ng,bgd->bnd', mask, fused)
        return (fused.new_zeros(b, n), pooled, fused, fused.mean(1))
    basin.forward = types.MethodType(forward, basin)
    basin.frontend_info = dict(basin.frontend_info, temporal='shared pointwiseMLP64 then local historyGRU64 over all15known grid days', grid_history_days=15, reservoir_history_days=15)
    return m

def install(md):
    if hasattr(md, 'history_base_make_model'):
        return
    base_constructor = md.make_model
    md.history_base_make_model = base_constructor

    def make(*args, **kwargs):
        return attach(base_constructor(*args, **kwargs))
    md.make_model = make
