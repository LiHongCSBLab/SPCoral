import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv


class HANConv(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads, meta_paths, ntypes):
        super(HANConv, self).__init__()
        self.meta_paths = meta_paths
        self.ntypes = ntypes
        self.gat_layers = nn.ModuleDict({
            mp: nn.ModuleDict({
                ntype: GATConv(
                    in_feats=in_feats[ntype],
                    out_feats=hidden_feats // num_heads,
                    num_heads=num_heads,
                    allow_zero_in_degree=True
                ) for ntype in ntypes
            }) for mp in meta_paths
        })
        self.semantic_attention = nn.ModuleDict({
            ntype: nn.Linear(hidden_feats, 1) for ntype in ntypes
        })
        self.fc = nn.ModuleDict({
            ntype: nn.Linear(hidden_feats, out_feats) for ntype in ntypes
        })
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h_dict):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for mp in self.meta_paths:
                meta_path = list(tuple(mp.split('_')))  # 假设 mp 是字符串，如 'paper_author_paper'
                self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(g, meta_path)

        h_out = {ntype: [] for ntype in h_dict}
        attn_out = {mp: {ntype: [] for ntype in h_dict} for mp in self.meta_paths}

        for mp in self.meta_paths:
            new_g = self._cached_coalesced_graph[mp]
            for ntype in h_dict:
                h, attn_weights = self.gat_layers[mp][ntype](new_g, h_dict[ntype], get_attention=True)
                attn_out[mp][ntype] = attn_weights

                h = h.flatten(1)
                h_out[ntype].append(h)

        for ntype in h_dict:
            h_out[ntype] = torch.stack(h_out[ntype], dim=1)
            attn_weights = torch.softmax(self.semantic_attention[ntype](h_out[ntype]), dim=1)
            h_out[ntype] = torch.sum(h_out[ntype] * attn_weights, dim=1)
            h_out[ntype] = self.fc[ntype](h_out[ntype])

        return h_out, attn_out