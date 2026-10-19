"""Neural layers and hydrological training loss."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm

class HydroLoss(nn.Module):

    def __init__(self, alpha=2.0, beta=0.1, q_ref=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss(reduction='none')
        if q_ref is None:
            self.register_buffer('q_ref', torch.tensor(0.0))
            self.use_fixed_ref = False
        else:
            self.register_buffer('q_ref', torch.tensor(float(q_ref)))
            self.use_fixed_ref = True

    def forward(self, pred_log, target_log):
        pred_phys = torch.expm1(pred_log)
        target_phys = torch.expm1(target_log)
        if self.use_fixed_ref and float(self.q_ref) > 0:
            ref = self.q_ref.to(device=target_phys.device, dtype=target_phys.dtype) + 1e-06
        else:
            ref = target_phys.max().detach() + 1e-06
        weights = 1.0 + self.alpha * (target_phys / ref)
        loss_pixel = self.mse(pred_log, target_log)
        loss_weighted = (loss_pixel * weights).mean()
        vol_pred = pred_phys.sum()
        vol_target = target_phys.sum()
        loss_vol = torch.abs(vol_pred - vol_target) / (vol_target + 1e-06)
        return loss_weighted + self.beta * loss_vol

class TimeFeatureEncoder(nn.Module):

    def __init__(self, embed_dim=16):
        super().__init__()
        self.month_emb = nn.Embedding(13, embed_dim)
        self.doy_proj = nn.Linear(2, embed_dim)

    def forward(self, doy_norm, month_idx):
        doy_rad = doy_norm * 2 * np.pi
        doy_cyc = torch.cat([torch.sin(doy_rad), torch.cos(doy_rad)], dim=-1)
        doy_feat = self.doy_proj(doy_cyc)
        month_clamped = torch.clamp(month_idx.squeeze(-1).long(), 0, 12)
        month_feat = self.month_emb(month_clamped)
        return doy_feat + month_feat

class PathPositionalEncoding(nn.Module):

    def __init__(self, embed_dim=32):
        super().__init__()
        self.phys_proj = nn.Sequential(nn.Linear(2, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.Dropout(0.3))

    def forward(self, phys_feats):
        return self.phys_proj(phys_feats)

class GridFeatureTower(nn.Module):

    def __init__(self, grid_dim, mask_idx, grid_ei, agg_mask, N_grid, N_res, phys_col_idx):
        super().__init__()
        self.mask_idx = mask_idx
        self.phys_col_idx = phys_col_idx
        self.path_enc = PathPositionalEncoding(embed_dim=32)
        gnn_in_dim = len(mask_idx) + 32
        self.gnn = GATv2Conv(gnn_in_dim, 32, heads=4, concat=True)
        self.gnn_norm = LayerNorm(128)
        self.edge_index = grid_ei
        self.lstm = nn.LSTM(grid_dim, 64, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(128 + 64 + 32 + 16, 128), nn.GELU(), nn.Dropout(0.3))
        self.runoff_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        self.route_attn = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.register_buffer('agg_mask', torch.FloatTensor(agg_mask))

    def forward(self, xg, time_embed):
        B, T, N, n_feat = xg.shape
        phys_feats = xg[:, 0, :, :][:, :, self.phys_col_idx]
        path_emb = self.path_enc(phys_feats)
        xg_flat = xg.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, n_feat)
        _, (hn, _) = self.lstm(xg_flat)
        h_temp = hn.squeeze(0).reshape(B, N, -1)
        xg_last = xg[:, -1, :, :][:, :, self.mask_idx]
        gnn_in = torch.cat([xg_last, path_emb], dim=-1)
        h_spat_list = []
        for i in range(B):
            h = self.gnn(gnn_in[i], self.edge_index)
            h_spat_list.append(h)
        h_spat = torch.stack(h_spat_list)
        h_spat = torch.stack([self.gnn_norm(sample) for sample in h_spat], dim=0)
        time_embed_exp = time_embed.unsqueeze(1).expand(-1, N, -1)
        h_grid = torch.cat([h_spat, h_temp, path_emb, time_embed_exp], dim=-1)
        h_fused = self.fusion(h_grid)
        q_grid = self.runoff_head(h_fused).squeeze(-1)
        attn_scores = self.route_attn(h_fused).squeeze(-1)
        mask_exp = self.agg_mask.unsqueeze(0)
        scores_exp = attn_scores.unsqueeze(1)
        masked_scores = scores_exp.masked_fill(mask_exp == 0, -1000000000.0)
        weights = F.softmax(masked_scores, dim=-1)
        h_vr_pooled = torch.sum(h_fused.unsqueeze(1) * weights.unsqueeze(-1), dim=2)
        h_basin_global = torch.mean(h_fused, dim=1)
        return (q_grid.squeeze(-1), h_vr_pooled, h_fused, h_basin_global)

class RecurrentFutureGenerator(nn.Module):

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.gru_cell = nn.GRUCell(1, hidden_dim)
        self.out_net = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward_step(self, rain_step, h_state):
        B, N, _ = rain_step.shape
        rain_flat = rain_step.reshape(B * N, 1)
        h_new = self.gru_cell(rain_flat, h_state)
        q_flat = self.out_net(h_new)
        return (q_flat.reshape(B, N), h_new)

class PanoramicRealReservoirTower(nn.Module):

    def __init__(self, F_res=5, N_res=19):
        super().__init__()
        self.lstm = nn.LSTM(F_res, 64, 2, batch_first=True, dropout=0.2)
        self.vr_proj = nn.Linear(N_res, 64)
        self.fc = nn.Sequential(nn.Linear(64 + F_res + 64 + 1 + 128 + 16, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, xr, q_routed_vec, future_rain_sum, h_basin_global, time_embed):
        _, (hn, _) = self.lstm(xr)
        h_last = hn[-1]
        vr_context = F.relu(self.vr_proj(q_routed_vec))
        xr_last = xr[:, -1, :]
        x_decision = torch.cat([h_last, xr_last, vr_context, future_rain_sum, h_basin_global, time_embed], dim=1)
        return self.fc(x_decision).squeeze(1)
