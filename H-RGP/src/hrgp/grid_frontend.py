"""Grid feature encoding with a fixed input mask."""
import types
import torch
from torch import nn
import torch.nn.functional as F
FRONTS = ('local_mlp', 'local_cnn', 'local_lstm', 'mean_gnn')
MASKED = ('flow_len', 'tc_sof', 'tc_if', 'lon', 'lat')
STATIC = ('elev', 'slope', 'ksat', 'poro', 'cn', 'fc', 'wp')

class LocalCausalCNN(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(width, 64, 3, dilation=1), nn.Conv1d(64, 64, 3, dilation=2), nn.Conv1d(64, 64, 3, dilation=4)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = F.gelu(layer(F.pad(x, (2 * layer.dilation[0], 0))))
        return x[:, :, -1]

def configure(model, mode, columns):
    assert mode in FRONTS
    basin = model.basin
    device = model.agg_mask.device
    for module in [basin.gnn, basin.gnn_norm, basin.path_enc, basin.route_attn, basin.runoff_head]:
        for p in module.parameters():
            p.requires_grad_(False)
    if mode in ['local_mlp', 'local_cnn']:
        for p in basin.lstm.parameters():
            p.requires_grad_(False)
    basin.local_static = nn.Sequential(nn.Linear(len(STATIC), 128), nn.GELU(), nn.LayerNorm(128)).to(device)
    basin.local_mlp = nn.Sequential(nn.Linear(len(columns), 64), nn.GELU(), nn.LayerNorm(64)).to(device)
    basin.local_cnn = LocalCausalCNN(len(columns)).to(device)
    for p in basin.local_mlp.parameters():
        p.requires_grad_(mode == 'local_mlp')
    for p in basin.local_cnn.parameters():
        p.requires_grad_(mode == 'local_cnn')
    gate = torch.ones(len(columns), device=device)
    for name in MASKED:
        gate[columns.index(name)] = 0
    basin.register_buffer('frontend_gate', gate)
    static_indices = [columns.index(c) for c in STATIC]
    edge = basin.edge_index
    loops = torch.arange(model.agg_mask.shape[1], device=device)
    edges = torch.cat([edge, torch.stack([loops, loops])], 1)
    basin.register_buffer('static_mean_edges', edges)
    degree = torch.bincount(edges[1], minlength=len(loops)).float().clamp_min(1)
    basin.register_buffer('static_mean_degree', degree)

    def forward(this, xg, time_embed):
        x = xg * this.frontend_gate[None, None, None, :]
        B, T, N, D = x.shape
        seq = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
        if mode == 'local_mlp':
            temporal = this.local_mlp(seq[:, -1])
        elif mode == 'local_cnn':
            temporal = this.local_cnn(seq)
        else:
            _, (h, _) = this.lstm(seq)
            temporal = h[-1]
        temporal = temporal.reshape(B, N, 64)
        stat = x[:, 0, :, static_indices]
        if mode == 'mean_gnn':
            src, dst = this.static_mean_edges
            stat = torch.zeros_like(stat).index_add(1, dst, stat[:, src]) / this.static_mean_degree[None, :, None]
        static = this.local_static(stat)
        path = static.new_zeros(B, N, 32)
        fused = this.fusion(torch.cat([static, temporal, path, time_embed[:, None, :].expand(-1, N, -1)], -1))
        mask = this.agg_mask / this.agg_mask.sum(1, keepdim=True).clamp_min(1)
        pooled = torch.einsum('ng,bgd->bnd', mask, fused)
        return (fused.new_zeros(B, N), pooled, fused, fused.mean(1))
    basin.forward = types.MethodType(forward, basin)
    basin.frontend_info = dict(mode=mode, dynamic_GAT=False, attention_pooling=False, temporal='last known day MLP with existing antecedent hydrologic features' if mode == 'local_mlp' else 'causal temporalCNN receptive15days' if mode == 'local_cnn' else 'local15day LSTM', spatial='fixed neighbor mean of7 STATIC attributes only' if mode == 'mean_gnn' else 'no cross-grid messages', masked_columns=list(MASKED), static_attributes=list(STATIC), normalization='per-grid feature LayerNorm', limitation='The grid encoder represents antecedent catchment conditions.')
