"""Grid, storage-routing and reservoir network components."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import layers as layers

class StorageCell(nn.Module):

    def __init__(self, river_feat_dim: int):
        super().__init__()
        self.phys_param_net = nn.Sequential(nn.Linear(river_feat_dim + 1, 32), nn.Tanh(), nn.Linear(32, 3), nn.Sigmoid())
        self.init_net = nn.Sequential(nn.Linear(128, 2), nn.Softplus())

    def forward(self, q_total_in, river_feat, state_pack, K_pow, alpha_pow):
        s_water, s_bed = state_pack
        param_input = torch.cat([river_feat, s_bed], dim=-1)
        params = self.phys_param_net(param_input)
        k_water, k_bed, alpha = (params[:, 0:1], params[:, 1:2], params[:, 2:3])
        s_bed_norm = torch.tanh(s_bed)
        loss_ratio = alpha * (1.0 - s_bed_norm)
        q_loss = q_total_in.unsqueeze(1) * loss_ratio
        q_net_in = q_total_in.unsqueeze(1) - q_loss
        s_bed_new = s_bed * (1.0 - k_bed) + q_loss
        s_potential = s_water + q_net_in
        s_pos = F.relu(s_potential) + 0.001
        q_phys = K_pow * torch.pow(s_pos, alpha_pow)
        q_out_final = torch.min(q_phys, s_potential)
        s_water_new = s_potential - q_out_final
        return (q_out_final.squeeze(1), (s_water_new, s_bed_new))

class StorageRoutingNetwork(nn.Module):

    def __init__(self, vr_data, static_sf: np.ndarray):
        super().__init__()
        self.sorted_nodes = vr_data['sorted_reservoirs']
        self.edge_index = np.asarray(vr_data['edge_index'])
        features = torch.FloatTensor(np.asarray(vr_data['reservoir_features'], dtype=np.float32))
        self.register_buffer('river_features', features)
        n_res = int(static_sf.shape[0])
        if n_res != vr_data['num_reservoirs']:
            raise ValueError('static_sf 第一维须等于 num_reservoirs')
        self.register_buffer('static_sf', torch.from_numpy(static_sf))
        input_dim = int(features.shape[1])
        self.spatial_param_mlp = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 2))
        self.vr_cells = nn.ModuleDict()
        for nid in self.sorted_nodes:
            self.vr_cells[str(nid)] = StorageCell(input_dim)
        self.upstream_map = {nid: [] for nid in self.sorted_nodes}
        if self.edge_index.shape[1] > 0:
            for src, dst in zip(self.edge_index[0], self.edge_index[1]):
                if dst in self.upstream_map:
                    self.upstream_map[dst].append(int(src))

    def forward_step(self, Q_agg_t: torch.Tensor, current_states: dict, future_rain_sum: torch.Tensor, sm_per_reservoir: torch.Tensor):
        B = Q_agg_t.shape[0]
        device = Q_agg_t.device
        q_outs_t: dict = {}
        new_states: dict = {}
        q_out_vec = torch.zeros(B, len(self.sorted_nodes), device=device)
        for idx, nid in enumerate(self.sorted_nodes):
            q_upstream = torch.zeros(B, device=device)
            for up_id in self.upstream_map[nid]:
                if up_id in q_outs_t:
                    q_upstream = q_upstream + q_outs_t[up_id]
            q_local = Q_agg_t[:, nid]
            q_total_in = torch.clamp(q_local + q_upstream, 0.0, 1000000.0)
            feat = self.river_features[nid].unsqueeze(0).expand(B, -1)
            prev = current_states[nid]
            sf = self.static_sf[nid].unsqueeze(0).expand(B, -1)
            sm_i = sm_per_reservoir[:, nid].unsqueeze(1)
            xa = torch.cat([sf, sm_i], dim=-1)
            raw = self.spatial_param_mlp(xa)
            K_pow = F.softplus(raw[:, 0:1]) + 0.0001
            alpha_pow = 0.15 + 2.85 * torch.sigmoid(raw[:, 1:2])
            q_out, new_sp = self.vr_cells[str(nid)](q_total_in, feat, prev, K_pow, alpha_pow)
            q_outs_t[nid] = q_out
            new_states[nid] = new_sp
            q_out_vec[:, idx] = q_out
        return (q_out_vec, new_states)

    def initialize_states(self, h_vr_pooled):
        states = {}
        for nid in self.sorted_nodes:
            h_node = h_vr_pooled[:, nid, :]
            s_init = self.vr_cells[str(nid)].init_net(h_node)
            states[nid] = (s_init[:, 0:1], s_init[:, 1:2])
        return states

class RunoffNetwork(nn.Module):

    def __init__(self, grid_ei, vr_data, grid_dim, res_dim, mask_idx, N_grid, phys_col_idx, static_sf: np.ndarray, sm_col_idx: int):
        super().__init__()
        self.sm_col_idx = sm_col_idx
        self.sm_relax = 0.92
        self.sm_rain_coef = 0.08
        self.time_enc = layers.TimeFeatureEncoder(embed_dim=16)
        self.basin = layers.GridFeatureTower(grid_dim, mask_idx, grid_ei, vr_data['grid_to_reservoir_matrix'], N_grid, vr_data['num_reservoirs'], phys_col_idx)
        self.vr_net = StorageRoutingNetwork(vr_data, static_sf)
        self.future_gen = layers.RecurrentFutureGenerator(hidden_dim=128)
        self.real_res = layers.PanoramicRealReservoirTower(res_dim, vr_data['num_reservoirs'])
        self.register_buffer('agg_mask', torch.FloatTensor(vr_data['grid_to_reservoir_matrix']))

    def _roll_sm_agg(self, sm_agg: torch.Tensor, rain_k: torch.Tensor) -> torch.Tensor:
        mask = self.agg_mask.to(device=rain_k.device, dtype=rain_k.dtype)
        grid_cnt = mask.sum(dim=1).clamp(min=1.0)
        rain_sum = torch.matmul(rain_k.squeeze(-1), mask.t())
        rain_agg = rain_sum / grid_cnt.unsqueeze(0)
        sm_next = self.sm_relax * sm_agg + self.sm_rain_coef * rain_agg
        return torch.clamp(sm_next, -5.0, 5.0)

    def forward(self, xg, xr, time_feats, future_rain_seq=None, lead_time=0, teacher_forcing_ratio=0.0):
        time_embed = self.time_enc(time_feats[:, 0:1], time_feats[:, 1:2])
        q_grid_0, h_vr_pooled, h_fused_0, h_basin_global = self.basin(xg, time_embed)
        q_grid_0 = torch.clamp(torch.expm1(torch.clamp(q_grid_0, -10, 20)), 0, 1000000.0)
        if future_rain_seq is not None:
            f_rain_sum = future_rain_seq.sum(dim=2).mean(dim=1, keepdim=True)
        else:
            f_rain_sum = torch.zeros(xg.shape[0], 1, device=xg.device)
        xg_last = xg[:, -1, :, :]
        mask = self.agg_mask.to(device=xg.device, dtype=xg.dtype)
        grid_cnt = mask.sum(dim=1).clamp(min=1.0)
        sm_sum = torch.matmul(xg_last[:, :, self.sm_col_idx], mask.t())
        sm_agg = sm_sum / grid_cnt.unsqueeze(0)
        current_vr_states = self.vr_net.initialize_states(h_vr_pooled)
        B, N, Fdim = h_fused_0.shape
        current_grid_state = h_fused_0.reshape(B * N, Fdim)
        if lead_time == 0:
            q_agg_0 = torch.matmul(q_grid_0, self.agg_mask.to(xg.device).t())
            q_routed_vec, _ = self.vr_net.forward_step(q_agg_0, current_vr_states, f_rain_sum, sm_agg)
        else:
            sm_roll = sm_agg
            if future_rain_seq is not None:
                for k in range(lead_time):
                    rain_k = future_rain_seq[:, :, k].unsqueeze(-1)
                    q_grid_pred, current_grid_state = self.future_gen.forward_step(rain_k, current_grid_state)
                    q_agg_fut = torch.matmul(q_grid_pred, self.agg_mask.to(xg.device).t())
                    q_routed_vec, current_vr_states = self.vr_net.forward_step(q_agg_fut, current_vr_states, f_rain_sum, sm_roll)
                    sm_roll = self._roll_sm_agg(sm_roll, rain_k)
            else:
                for _k in range(lead_time):
                    zeros = torch.zeros(xg.shape[0], self.agg_mask.shape[0], device=xg.device)
                    q_routed_vec, current_vr_states = self.vr_net.forward_step(zeros, current_vr_states, f_rain_sum, sm_roll)
        adj = self.real_res(xr, q_routed_vec, f_rain_sum, h_basin_global, time_embed)
        outlet_idx = len(self.vr_net.sorted_nodes) - 1
        base_pred = q_routed_vec[:, outlet_idx]
        return torch.clamp(base_pred + adj, 0, 1000000.0)
