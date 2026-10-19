"""Hydrological model implementation."""
import torch
import torch.nn.functional as F

def grid_forward(self, xg, time_embed):
    Bn, T, Nn, Fdim = xg.shape
    path = self.path_enc(xg[:, 0, :, :][:, :, self.phys_col_idx])
    seq = xg.permute(0, 2, 1, 3).contiguous().reshape(Bn * Nn, T, Fdim) if self.axis_correct else xg.reshape(Bn * Nn, T, Fdim)
    _, (hn, _) = self.lstm(seq)
    ht = hn.squeeze(0).reshape(Bn, Nn, -1)
    gin = torch.cat([xg[:, -1, :, :][:, :, self.mask_idx], path], dim=-1)
    edges = self.edge_index.to(xg.device)
    batched = (edges.unsqueeze(0) + torch.arange(Bn, device=xg.device).view(-1, 1, 1) * Nn).permute(1, 0, 2).reshape(2, -1)
    if getattr(self, 'capture_attention', False):
        hs, (ee, aa) = self.gnn(gin.reshape(Bn * Nn, -1), batched, return_attention_weights=True)
        self.attention = (ee.detach(), aa.detach())
    else:
        hs = self.gnn(gin.reshape(Bn * Nn, -1), batched)
    hs = hs.reshape(Bn, Nn, -1)
    if self.sample_norm:
        hs = self.gnn_norm(hs.reshape(Bn * Nn, -1), batch=torch.arange(Bn, device=xg.device).repeat_interleave(Nn)).reshape(Bn, Nn, -1)
    else:
        hs = self.gnn_norm(hs)
    fused = self.fusion(torch.cat([hs, ht, path, time_embed.unsqueeze(1).expand(-1, Nn, -1)], dim=-1))
    q = self.runoff_head(fused).squeeze(-1)
    score = self.route_attn(fused).squeeze(-1)
    weights = F.softmax(score.unsqueeze(1).masked_fill(self.agg_mask.unsqueeze(0) == 0, -1000000000.0), dim=-1)
    pooled = (fused.unsqueeze(1) * weights.unsqueeze(-1)).sum(2)
    return (q, pooled, fused, fused.mean(1))
