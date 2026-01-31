# import

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_code import GATConv

from typing import List, Optional
import warnings

from .utils import normalize_tensor

warnings.filterwarnings("ignore")

# WGAN GP

from math import ceil

class WDiscriminator(nn.Module):
    r"""
    WGAN Discriminator

    Parameters
    ----------
    hidden_size
        input dim
    hidden_size2
        hidden dim
    """

    def __init__(self, hidden_size: int, hidden_size2: Optional[int] = 512):
        super().__init__()
        self.hidden = nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = nn.Linear(hidden_size2, hidden_size2)
        self.output = nn.Linear(hidden_size2, 1)

    def forward(self, input_embd):
        return self.output(
            F.leaky_relu(
                self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)),
                0.2,
                inplace=True,
            )
        )


def gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty.mean()


def train_GAN(
        wdiscriminator: torch.nn.Module,
        optimizer_d: torch.optim.Optimizer,
        embds: List[torch.Tensor],
        device: torch.device = 'cpu',
        batch_d_per_iter: Optional[int] = 5,
        anchor_scale: Optional[float] = 0.8,
        lambda_gp: Optional[float] = 2.0,
) -> torch.Tensor:
    embd0, embd1 = [embd.clone() for embd in embds]
    wdiscriminator.train()

    min_batch_size = min(embd0.size(0), embd1.size(0))
    anchor_size = ceil(min_batch_size * anchor_scale)

    for j in range(batch_d_per_iter):
        w0 = wdiscriminator(embd0)
        w1 = wdiscriminator(embd1)
        anchor1 = w1.view(-1).argsort(descending=True)[:anchor_size]
        anchor0 = w0.view(-1).argsort(descending=True)[:anchor_size]
        embd0_anchor = embd0[anchor0, :].clone().detach()
        embd1_anchor = embd1[anchor1, :].clone().detach()

        optimizer_d.zero_grad()

        real_loss = torch.mean(wdiscriminator(embd0_anchor))
        fake_loss = torch.mean(wdiscriminator(embd1_anchor))

        gp = gradient_penalty(wdiscriminator, embd0_anchor, embd1_anchor, device)

        loss_d = -real_loss + fake_loss + lambda_gp * gp
        loss_d.backward()
        optimizer_d.step()

    loss_g = -torch.mean(wdiscriminator(embd1_anchor))
    return loss_g


# GAT

class GAT(nn.Module):
    def __init__(self, hidden_dims):
        super(GAT, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, g, features):
        """
        Args:
            g (dgl.DGLGraph): The input graph.
            features (torch.Tensor): Node features of shape (num_nodes, in_dim).

        Returns:
            tuple: (h2, h4) where h2 is the latent representation and h4 is the reconstructed features.
        """
        h1 = F.elu(self.conv1(g, features))
        h2 = self.conv2(g, h1, attention=False)
        # Transpose weights for conv3 and conv4 to match PyG behavior
        self.conv3.lin.weight.data = self.conv2.lin.weight.transpose(0, 1)
        self.conv4.lin.weight.data = self.conv1.lin.weight.transpose(0, 1)
        h3 = F.elu(self.conv3(g, h2, attention=True, tied_attention=self.conv1.attentions))
        h4 = self.conv4(g, h3, attention=False)
        return h2, h4


class COM_NET(nn.Module):
    def __init__(self, omics1_hidden_dims_pcc, omics2_hidden_dims_pcc, hidden_dims_shared):
        super(COM_NET, self).__init__()

        self.omics1_gat = GAT(omics1_hidden_dims_pcc)
        self.omics2_gat = GAT(omics2_hidden_dims_pcc)
        self.hidden_dims_shared = hidden_dims_shared

        if hidden_dims_shared[-1] != 0:
            self.shared_gat = Encoder_shared(hidden_dims_shared)

    def forward(self, g_omics1, features_omics1, shared_omics1, g_omics2, features_omics2, shared_omics2):

        emb_omics1, de_omics1 = self.omics1_gat(g_omics1, features_omics1)
        emb_omics2, de_omics2 = self.omics2_gat(g_omics2, features_omics2)

        if self.hidden_dims_shared[-1] != 0:
            shared_emb_omics1, shared_emb_omics2 = self.shared_gat(g_omics1, shared_omics1, g_omics2, shared_omics2)
        else:
            shared_emb_omics1, shared_emb_omics2 = None, None

        return emb_omics1, emb_omics2, de_omics1, de_omics2, shared_emb_omics1, shared_emb_omics2


class _GAT(nn.Module):
    def __init__(self, hidden_dims):
        super(_GAT, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims

        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, g, features):
        h1 = F.elu(self.conv1(g, features))
        h2 = self.conv2(g, h1, attention=False)

        return h2


class Encoder_shared(nn.Module):
    def __init__(self, hidden_dims):
        super(Encoder_shared, self).__init__()
        self.encoder = _GAT(hidden_dims)

    def forward(self, g_omics1, features_omics1, g_omics2, features_omics2):
        h1 = self.encoder(g_omics1, features_omics1)
        h2 = self.encoder(g_omics2, features_omics2)

        z1 = normalize_tensor(h1)
        z2 = normalize_tensor(h2)

        return z1, z2

### integration model

# class SharedCrossModalAttention(nn.Module):
#     def __init__(self, dim):
#         super(SharedCrossModalAttention, self).__init__()
#         self.num_heads = 1
#         self.dim = dim
#         self.head_dim = dim
#         self.query_omics1 = nn.Linear(dim, dim)  # 主图查询
#         self.key_omics2 = nn.Linear(dim, dim)  # 辅助图键
#         self.value_omics1 = nn.Linear(dim, dim)  # 主图值
#         self.value_omics2 = nn.Linear(dim, dim)  # 辅助图值
#         self.dropout = nn.Dropout(0)
#         self.out_omics1 = nn.Linear(dim, dim)
#         self.out_omics2 = nn.Linear(dim, dim)
#         self.scale = 1 / (self.head_dim ** 0.5)
#
#     def forward(self, h_omics1, h_omics2, g_cross):
#         batch_size_1 = h_omics1.size(0)
#         batch_size_2 = h_omics2.size(0)
#
#         q_omics1 = self.query_omics1(h_omics1)
#         v_omics1 = self.value_omics1(h_omics1)
#         k_omics2 = self.key_omics2(h_omics2)
#         v_omics2 = self.value_omics2(h_omics2)
#
#         # Squeeze g_cross to [batch_size_1, batch_size_2] if needed
#         g_cross = g_cross.squeeze()
#
#         # Get non-zero indices (edges: row from omics1, col from omics2)
#         row, col = g_cross.nonzero(as_tuple=True)
#
#         # Compute attn_scores per edge (dot product)
#         attn_scores = (q_omics1[row] * k_omics2[col]).sum(dim=-1) * self.scale
#
#         # --- Compute out_h_omics1: softmax per row (omics1) ---
#         # Sort edges by row for grouping
#         sort_idx = torch.argsort(row)
#         row_sorted = row[sort_idx]
#         col_sorted = col[sort_idx]
#         attn_scores_sorted = attn_scores[sort_idx]
#
#         # Get unique rows and their counts
#         unique_rows, counts = torch.unique_consecutive(row_sorted, return_counts=True)
#
#         # Compute softmax weights per group (stable softmax)
#         weights = torch.zeros_like(attn_scores_sorted)
#         start = 0
#         for i, count in enumerate(counts):
#             end = start + count
#             scores_group = attn_scores_sorted[start:end]
#             if count > 0:
#                 max_val = scores_group.max()
#                 exp = torch.exp(scores_group - max_val)
#                 sum_exp = exp.sum()
#                 weights[start:end] = exp / (sum_exp + 1e-10)  # Avoid div by zero
#             start = end
#
#         weights = self.dropout(weights)
#
#         # Aggregate to out_h_omics1 using index_add_
#         out_h_omics1 = torch.zeros(batch_size_1, self.dim, device=h_omics1.device, dtype=h_omics1.dtype)
#         out_h_omics1.index_add_(0, row_sorted, weights.unsqueeze(1) * v_omics2[col_sorted])
#         out_h_omics1 = self.out_omics1(out_h_omics1)
#
#         # --- Compute out_h_omics2: softmax per col (omics2, transposed) ---
#         # Sort edges by col for grouping
#         sort_idx_t = torch.argsort(col)
#         col_sorted_t = col[sort_idx_t]
#         row_sorted_t = row[sort_idx_t]
#         attn_scores_sorted_t = attn_scores[sort_idx_t]
#
#         # Get unique cols and their counts
#         unique_cols, counts_t = torch.unique_consecutive(col_sorted_t, return_counts=True)
#
#         # Compute softmax weights per group (stable softmax)
#         weights_t = torch.zeros_like(attn_scores_sorted_t)
#         start = 0
#         for i, count in enumerate(counts_t):
#             end = start + count
#             scores_group = attn_scores_sorted_t[start:end]
#             if count > 0:
#                 max_val = scores_group.max()
#                 exp = torch.exp(scores_group - max_val)
#                 sum_exp = exp.sum()
#                 weights_t[start:end] = exp / (sum_exp + 1e-10)
#             start = end
#
#         weights_t = self.dropout(weights_t)
#
#         # Aggregate to out_h_omics2 using index_add_
#         out_h_omics2 = torch.zeros(batch_size_2, self.dim, device=h_omics2.device, dtype=h_omics2.dtype)
#         out_h_omics2.index_add_(0, col_sorted_t, weights_t.unsqueeze(1) * v_omics1[row_sorted_t])
#         out_h_omics2 = self.out_omics2(out_h_omics2)
#
#         # Handle empty groups (rows/cols with no edges remain zero, equivalent to original with all -inf)
#         return out_h_omics1, out_h_omics2

class SharedCrossModalAttention(nn.Module):
    def __init__(self, dim):
        super(SharedCrossModalAttention, self).__init__()
        self.num_heads = 1
        self.dim = dim
        self.head_dim = dim
        # assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.query_omics1 = nn.Linear(dim, dim)  # 主图查询
        self.key_omics2 = nn.Linear(dim, dim)  # 辅助图键
        self.value_omics1 = nn.Linear(dim, dim)  # 主图值
        self.value_omics2 = nn.Linear(dim, dim)  # 辅助图值
        self.dropout = nn.Dropout(0)
        self.out_omics1 = nn.Linear(dim, dim)
        self.out_omics2 = nn.Linear(dim, dim)
        self.scale = 1 / (self.head_dim ** 0.5)

    def forward(self, h_omics1, h_omics2, g_cross):
        batch_size_1 = h_omics1.size(0)
        batch_size_2 = h_omics2.size(0)
        seq_len = 1

        # 主图 Q 和 V
        q_omics1 = self.query_omics1(h_omics1).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_omics1 = self.value_omics1(h_omics1).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 辅助图 K 和 V
        k_omics2 = self.key_omics2(h_omics2).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_omics2 = self.value_omics2(h_omics2).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算共享注意力
        attn_scores = torch.matmul(q_omics1, k_omics2.transpose(2, 3)) * self.scale
        attn_scores = attn_scores.masked_fill(g_cross == 0, -torch.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # graph -> aux
        out_h_omics1 = torch.matmul(attn_weights, v_omics2).transpose(1, 2).contiguous().view(batch_size_1, self.dim)
        out_h_omics1 = self.out_omics1(out_h_omics1)

        # aux -> graph（转置复用）
        attn_weights_t = F.softmax(attn_scores.transpose(-2, -1), dim=-1)
        out_h_omics2 = torch.matmul(attn_weights_t, v_omics1).transpose(1, 2).contiguous().view(batch_size_2, self.dim)
        out_h_omics2 = self.out_omics2(out_h_omics2)

        return out_h_omics1, out_h_omics2

class CrossModalGAE(nn.Module):
    def __init__(
            self,
            input_dim_omics1,
            input_dim_omics2,
            hidden_dim,
            latent_dim,
            norm=True
    ):
        super(CrossModalGAE, self).__init__()
        # self.coord_embed = nn.Linear(2, hidden_dim)
        self.norm = norm

        self.gat1_omics1 = GATConv(input_dim_omics1, hidden_dim)
        self.gat1_omics2 = GATConv(input_dim_omics2, hidden_dim)

        self.cross_attn = SharedCrossModalAttention(hidden_dim)

        # self.residual_weight_omics1 = nn.Parameter(torch.ones(num_omics1, 2))
        # self.residual_weight_omics2 = nn.Parameter(torch.ones(num_omics2, 2))

        # self.gat2 = GATConv(hidden_dim, latent_dim)
        # self.gat2_logvar = GATConv(hidden_dim * num_heads, latent_dim, 1, dropout=dropout)

        self.gat2_cross = GATConv(hidden_dim, latent_dim)

        self.gat3_omics1 = GATConv(latent_dim, hidden_dim)
        self.gat3_omics2 = GATConv(latent_dim, hidden_dim)

        self.gat4_omics1 = GATConv(hidden_dim, input_dim_omics1)
        self.gat4_omics2 = GATConv(hidden_dim, input_dim_omics2)

    def encoder(
            self,
            g_omics1,
            g_omics2,
            g_feature_omics1,
            g_feature_omics2,
            x_omics1,
            x_omics2,
            g_cross,
            g_cross_dgl
    ):

        hidden_omics1 = F.elu(0.5*self.gat1_omics1(g_omics1, x_omics1) + 0.5*self.gat1_omics1(g_feature_omics1, x_omics1))
        hidden_omics2 = F.elu(0.5*self.gat1_omics2(g_omics2, x_omics2) + 0.5*self.gat1_omics2(g_feature_omics2, x_omics2))

        h_omics1_attn, h_omics2_attn = self.cross_attn(hidden_omics1, hidden_omics2, g_cross)

        if self.norm:
            hidden_omics1 = normalize_tensor(hidden_omics1)
            hidden_omics2 = normalize_tensor(hidden_omics2)
            h_omics1_attn = normalize_tensor(h_omics1_attn)
            h_omics2_attn = normalize_tensor(h_omics2_attn)

        h_fusion_omics1 = F.elu(0.5*hidden_omics1 + 0.5*h_omics1_attn)
        h_fusion_omics2 = F.elu(0.5*h_omics2_attn + 0.5*hidden_omics2)

        mu_omics = self.gat2_cross(g_cross_dgl, torch.cat((h_fusion_omics1, h_fusion_omics2)))

        mu_omics1 = mu_omics[:h_fusion_omics1.shape[0] , :]
        mu_omics2 = mu_omics[h_fusion_omics1.shape[0]: , :]

        if self.norm:
            mu_omics1 = normalize_tensor(mu_omics1)
            mu_omics2 = normalize_tensor(mu_omics2)

        return mu_omics1, mu_omics2, hidden_omics1, hidden_omics2, h_omics1_attn, h_omics2_attn

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    # def spa_decode(self, z):
    #     adj_logits = torch.matmul(z, z.t())
    #     return torch.sigmoid(adj_logits)

    def exp_decoder_omics1(self, g, z):
        h1 = F.elu(self.gat3_omics1(g, z))
        h2 = self.gat4_omics1(g, h1)
        return h2

    def exp_decoder_omics2(self, g, z):
        h1 = F.elu(self.gat3_omics2(g, z))
        h2 = self.gat4_omics2(g, h1)
        return h2

    def cross_encoder(
            self,
            g_cross_omics1,
            g_cross_omics2,
            g_feature_cross_omics1,
            g_feature_cross_omics2,
            feat_cross_omics1,
            feat_cross_omics2,
            hidden_omics1,
            hidden_omics2,
    ):

        hidden_cross_omics1 = F.elu(0.5*self.gat1_omics1(g_cross_omics1, feat_cross_omics1) + 0.5*self.gat1_omics1(g_feature_cross_omics1, feat_cross_omics1))
        hidden_cross_omics2 = F.elu(0.5*self.gat1_omics2(g_cross_omics2, feat_cross_omics2) + 0.5*self.gat1_omics2(g_feature_cross_omics2, feat_cross_omics2))

        if self.norm:
            hidden_cross_omics1 = normalize_tensor(hidden_cross_omics1)
            hidden_cross_omics2 = normalize_tensor(hidden_cross_omics2)

        h_fusion_cross_omics1 = F.elu(0.5*hidden_omics1 + 0.5*hidden_cross_omics2)
        h_fusion_cross_omics2 = F.elu(0.5*hidden_cross_omics1 + 0.5*hidden_omics2)

        # mu_omics = self.gat2_cross(g_cross_dgl, torch.cat((h_fusion_cross_omics1, h_fusion_cross_omics2)))

        mu_cross_omics1 = self.gat2_cross(g_cross_omics2, h_fusion_cross_omics1)
        mu_cross_omics2 = self.gat2_cross(g_cross_omics1, h_fusion_cross_omics2)
        # logvar_cross_omics1 = self.gat2_logvar(g_cross_omics2, h_fusion_cross_omics1)
        # logvar_cross_omics2 = self.gat2_logvar(g_cross_omics1, h_fusion_cross_omics2)

        return mu_cross_omics1, mu_cross_omics2


    def forward(
            self,
            g_omics1,
            g_omics2,
            g_feature_omics1,
            g_feature_omics2,
            feat_omics1,
            feat_omics2,
            g_cross,
            g_cross_dgl
    ):
        h_omics1, h_omics2, hidden_omics1, hidden_omics2, h_omics1_attn, h_omics2_attn = self.encoder(g_omics1, g_omics2, g_feature_omics1, g_feature_omics2, feat_omics1, feat_omics2, g_cross, g_cross_dgl)

        # h_omics1 = self.reparameterize(mu_omics1, logvar_omics1)
        # h_omics2 = self.reparameterize(mu_omics2, logvar_omics2)

        feat_rec_omics1 = self.exp_decoder_omics1(g_omics1, h_omics1)
        feat_rec_omics2 = self.exp_decoder_omics2(g_omics2, h_omics2)

        feat_cross_omics1 = self.exp_decoder_omics1(g_omics2, h_omics2)
        feat_cross_omics2 = self.exp_decoder_omics2(g_omics1, h_omics1)

        cross_omics1, cross_omics2 = self.cross_encoder(
            g_omics2, g_omics1,
            g_feature_omics2, g_feature_omics1,
            feat_cross_omics1, feat_cross_omics2,
            hidden_omics1, hidden_omics2
        )

        return h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, feat_cross_omics1, feat_cross_omics2, cross_omics1, cross_omics2




