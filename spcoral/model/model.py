# import
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import random
import os
from typing import List, Optional, Tuple, Dict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import COM_NET, train_GAN, WDiscriminator, CrossModalGAE
from .utils import adata_to_dgl, get_adj_nearest, get_kpop_neighbors, cal_morphology, normalize_tensor, build_graph_feature
from ..utils import create_snn_adjacency_matrix, preprogress_adata

from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

__all__ = [
    'regist_model',
    'integrate_model',
    'integrate_model_block',
]


def set_random_seed(seed=2024, strict_repro=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if strict_repro:
        torch.backends.cudnn.benchmark = False
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        print(f"[Strict Repro Mode] Seed={seed}, multi-thread OFF")
    else:
        torch.backends.cudnn.benchmark = True
        print(f"[Fast Mode] Seed={seed}, cudnn.benchmark=True, multi-thread ON")

    import dgl
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

class regist_model:
    """
    A PyTorch-based model for cross-modality spatial omics integration via graph attention networks
    and adversarial alignment.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        First omics AnnData object. Must contain ``obsm['feat']`` (modality-specific features)
        and spatial coordinates (via ``use_obsm``).
    adata_omics2 : anndata.AnnData
        Second omics AnnData object with the same requirements.
    graph_method : str
        Method used to construct spatial graphs (passed to ``adata_to_dgl``; e.g., 'knn' or 'radius').
    k_spatial_omics1 : int, optional
        Number of nearest neighbors for omics1 graph when ``graph_method='knn'``.
    radius_spatial_omics1 : float, optional
        Radius for omics1 graph when ``graph_method='radius'``.
    k_spatial_omics2 : int, optional
        Number of nearest neighbors for omics2 graph.
    radius_spatial_omics2 : float, optional
        Radius for omics2 graph.
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` containing spatial coordinates.
    n_layer : list of int, optional (default: [1, 2, 3, 4, 5])
        Orders of neighborhood aggregation (hop distances) for morphological feature computation.
        The actual layers used are ``[0] + n_layer``.
    alpha : float, optional (default: 0.1)
        Weight balancing reconstruction loss and adversarial (GAN) loss.
    device : torch.device, optional (default: torch.device('cuda:0'))
        Device for training.
    random_seed : int, optional (default: 2024)
        Random seed for reproducibility.
    strict_repro : bool, optional (default: False)
        If True, enforces stricter reproducibility (e.g., deterministic CUDA operations).
    learning_rate : float, optional (default: 0.001)
        Learning rate for the main model optimizer.
    weight_decay : float, optional (default: 0.0001)
        Weight decay for both optimizers.
    epochs : int, optional (default: 100)
        Number of training epochs.
    gradient_clipping : float, optional (default: 5.0)
        Maximum gradient norm for clipping.
    hidden_dim_shared : int, optional (default: 32)
        Hidden dimension in the shared feature branch.
    out_dim_shared : int, optional (default: 8)
        Output dimension of the shared embedding.
    hidden_dim_pcc : int, optional (default: 32)
        Hidden dimension in modality-specific (PCC) branches.
    out_dim_pcc : int, optional (default: 8)
        Output dimension of modality-specific embeddings.
    GAN_batch_d_per_iter : int, optional (default: 5)
        Number of discriminator updates per generator update in the Wasserstein GAN.

    Attributes
    ----------
    adata_omics1, adata_omics2 : anndata.AnnData
        Processed AnnData objects with added embeddings after training.
    model : torch.nn.Module
        The main COM_NET encoder-decoder model (instantiated during ``train``).

    """
    def __init__(
            self,
            adata_omics1: anndata.AnnData,
            adata_omics2: anndata.AnnData,
            # spatial graph param
            graph_method: str,
            k_spatial_omics1: Optional[int] = None,
            radius_spatial_omics1: Optional[float] = None,
            k_spatial_omics2: Optional[int] = None,
            radius_spatial_omics2: Optional[float] = None,
            use_obsm: str = 'spatial',
            # model param
            n_layer: List[int] = [1, 2, 3, 4, 5],
            alpha: float = 0.1,
            device: torch.device = torch.device('cuda:0'),
            random_seed: int = 2024,
            strict_repro: bool = False,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            epochs: int = 100,
            gradient_clipping: float = 5.,
            # gat param shared network
            hidden_dim_shared: int = 32,
            out_dim_shared: int = 8,
            # gat param pcc network
            hidden_dim_pcc: int = 32,
            out_dim_pcc: int = 8,
            # gan param
            GAN_batch_d_per_iter: int = 5,
    ):
        set_random_seed(random_seed, strict_repro=strict_repro)

        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2

        self.graph_omics1 = adata_to_dgl(
            adata_omics1,
            method=graph_method,
            k=k_spatial_omics1,
            radius=radius_spatial_omics1,
            use_obsm=use_obsm
        ).to(device)

        self.graph_omics2 = adata_to_dgl(
            adata_omics2,
            method=graph_method,
            k=k_spatial_omics2,
            radius=radius_spatial_omics2,
            use_obsm=use_obsm
        ).to(device)

        self.feature_omics1 = torch.from_numpy(adata_omics1.obsm['feat']).to(torch.float32).to(device)
        self.feature_omics2 = torch.from_numpy(adata_omics2.obsm['feat']).to(torch.float32).to(device)

        self.share_feature_names = list(np.intersect1d(adata_omics1.var_names, adata_omics2.var_names))

        self.shared_omics1 = torch.from_numpy(adata_omics1[:, self.share_feature_names].X).to(torch.float32).to(device)
        self.shared_omics2 = torch.from_numpy(adata_omics2[:, self.share_feature_names].X).to(torch.float32).to(device)

        self.adj_nearest_omics1 = get_adj_nearest(self.graph_omics1)
        self.adj_nearest_omics2 = get_adj_nearest(self.graph_omics2)

        n_layer = [0] + n_layer

        self.neighbors_omics1 = torch.from_numpy(
            np.array([get_kpop_neighbors(self.adj_nearest_omics1, n_layer[i+1], n_layer[i]) for i in range(len(n_layer) - 1)])
        ).to(torch.float32).to(device)
        self.neighbors_omics2 = torch.from_numpy(
            np.array([get_kpop_neighbors(self.adj_nearest_omics2, n_layer[i+1], n_layer[i]) for i in range(len(n_layer) - 1)])
        ).to(torch.float32).to(device)

        self.hidden_dims_shared = [self.shared_omics1.shape[1], hidden_dim_shared, out_dim_shared]

        self.omics1_hidden_dims_pcc = [self.feature_omics1.shape[1], hidden_dim_pcc, out_dim_pcc]
        self.omics2_hidden_dims_pcc = [self.feature_omics2.shape[1], hidden_dim_pcc, out_dim_pcc]

        self.GAN_batch_d_per_iter = GAN_batch_d_per_iter
        self.alpha = alpha
        self.len_n_layer = len(n_layer) - 1 + out_dim_shared

        self.gradient_clipping = gradient_clipping
        self.random_seed = random_seed
        self.strict_repro=strict_repro
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

    def train(self):
        """
        Train the integration model.

        Returns
        -------
        tuple
            - adata_omics1 : anndata.AnnData
              Updated first omics object with ``obsm['embedding']`` and ``obsm['share_feature']``.
            - adata_omics2 : anndata.AnnData
              Updated second omics object with the same added keys.
            - loss_list : list of [reconstruction_loss, gan_loss] per epoch
              Training loss history.
        """
        set_random_seed(self.random_seed, strict_repro=self.strict_repro)
        global emb_omics1, emb_omics2, share_feature_omics1, share_feature_omics2, add_feature_omics1, add_feature_omics2

        self.model = COM_NET(self.omics1_hidden_dims_pcc, self.omics2_hidden_dims_pcc, self.hidden_dims_shared).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.wdiscriminator = WDiscriminator(self.len_n_layer).to(self.device)
        self.optimizer_wd = torch.optim.Adam(
            self.wdiscriminator.parameters(),
            lr=self.learning_rate * 0.1,
            weight_decay=self.weight_decay
        )

        self.model.train()

        loss_list = []

        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.optimizer.zero_grad()

            emb_omics1, emb_omics2, de_omics1, de_omics2, shared_emb_omics1, shared_emb_omics2 = self.model(
                self.graph_omics1,
                self.feature_omics1,
                self.shared_omics1,
                self.graph_omics2,
                self.feature_omics2,
                self.shared_omics2
            )

            share_feature_omics1 = cal_morphology(self.neighbors_omics1, emb_omics1)
            share_feature_omics2 = cal_morphology(self.neighbors_omics2, emb_omics2)

            share_feature_norm_omics1 = normalize_tensor(share_feature_omics1)
            share_feature_norm_omics2 = normalize_tensor(share_feature_omics2)

            if shared_emb_omics1 != None and shared_emb_omics2 != None:
                add_feature_omics1 = torch.cat([share_feature_norm_omics1, shared_emb_omics1], dim=1)
                add_feature_omics2 = torch.cat([share_feature_norm_omics2, shared_emb_omics2], dim=1)
            else:
                add_feature_omics1 = share_feature_norm_omics1
                add_feature_omics2 = share_feature_norm_omics2

            self.loss_GAN = train_GAN(
                self.wdiscriminator,
                self.optimizer_wd,
                [add_feature_omics1, add_feature_omics2],
                device=self.device,
                batch_d_per_iter=self.GAN_batch_d_per_iter,
                anchor_scale=0.8,
            )

            self.loss_restruction = 0.5 * F.mse_loss(self.feature_omics1, de_omics1) + 0.5 * F.mse_loss(self.feature_omics2, de_omics2)

            # self.loss_shared =

            self.loss = self.loss_restruction + self.alpha * self.loss_GAN

            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()

            loss_list.append([self.loss_restruction, self.alpha * self.loss_GAN])

        self.model.eval()

        self.adata_omics1.obsm['embedding'] = emb_omics1.to('cpu').detach().numpy()
        self.adata_omics2.obsm['embedding'] = emb_omics2.to('cpu').detach().numpy()
        self.adata_omics1.obsm['share_feature'] = add_feature_omics1.to('cpu').detach().numpy()
        self.adata_omics2.obsm['share_feature'] = add_feature_omics2.to('cpu').detach().numpy()

        return self.adata_omics1, self.adata_omics2, loss_list


class integrate_model:
    """
    A graph autoencoder model for cross-modality spatial omics integration.

    This class integrates two spatial omics datasets by learning joint latent embeddings using
    a multi-graph attention-based autoencoder. It leverages:
    - Modality-specific spatial graphs,
    - Feature-based similarity graphs,
    - Cross-modality spatial nearest-neighbor graphs,
    - A unified joint graph combining all connections.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        First omics AnnData object. Must contain ``obsm['feat']`` (modality-specific features)
        and spatial coordinates in ``obsm[use_obsm]``.
    adata_omics2 : anndata.AnnData
        Second omics AnnData object with the same requirements.
    graph_method_single : str
        Method for building individual spatial graphs for each modality
        (e.g., 'knn' or 'radius'; passed to ``adata_to_dgl``).
    k_spatial_omics1 : int, optional
        Number of spatial neighbors for omics1 (used if ``graph_method_single='knn'``).
    radius_spatial_omics1 : float, optional
        Radius for omics1 spatial graph (used if ``graph_method_single='radius'``).
    k_spatial_omics2 : int, optional
        Number of spatial neighbors for omics2.
    radius_spatial_omics2 : float, optional
        Radius for omics2 spatial graph.
    use_obsm : str, optional (default: 'spatial')
        Key in ``.obsm`` containing spatial coordinates.
    g_all_auto : bool, optional (default: True)
        If True, automatically construct the joint graph from individual and cross graphs.
        If False, build it directly from all coordinates using ``k_all_omics``.
    k_feature_omics1 : int, optional (default: 10)
        Number of feature-based nearest neighbors for omics1.
    k_feature_omics2 : int, optional (default: 10)
        Number of feature-based nearest neighbors for omics2.
    k_cross_omics : int, optional (default: 20)
        Number of cross-modality spatial nearest neighbors (bipartite edges).
    k_all_omics : int, optional (default: 25)
        Number of neighbors in joint graph when ``g_all_auto=False``.
    loss_weight : list of float, optional (default: [1, 1, 1, 1, 1])
        Weights for the five loss components:
        [recon_omics1, recon_omics2, cross_omics1, cross_omics2, spatial_graph].
    hidden_dim : int, optional (default: 128)
        Hidden dimension in the GAE encoder/decoder.
    latent_dim : int, optional (default: 64)
        Dimension of the final latent embedding.
    device : torch.device, optional (default: torch.device('cuda:0'))
        Device for training.
    random_seed : int, optional (default: 2020)
        Random seed for reproducibility.
    strict_repro : bool, optional (default: False)
        Enforce strict reproducibility (e.g., deterministic CUDA).
    learning_rate : float, optional (default: 0.001)
        Optimizer learning rate.
    weight_decay : float, optional (default: 0.0001)
        Weight decay for Adam optimizer.
    epochs : int, optional (default: 300)
        Number of training epochs.
    gradient_clipping : float, optional (default: 5.0)
        Gradient clipping norm.
    embedding_key : str, optional (default: 'emb_spcoral')
        Key to store latent embeddings in ``.obsm``.
    rec_key : str, optional (default: 'rec_spcoral')
        Key to store reconstructed modality-specific features.
    cross_key : str, optional (default: 'cross_spcoral')
        Key to store cross-modality predicted features.
    """
    def __init__(
            self,
            adata_omics1: anndata.AnnData,
            adata_omics2: anndata.AnnData,
            graph_method_single: str,
            k_spatial_omics1: Optional[int] = None,
            radius_spatial_omics1: Optional[float] = None,
            k_spatial_omics2: Optional[int] = None,
            radius_spatial_omics2: Optional[float] = None,
            use_obsm: str = 'spatial',
            g_all_auto: bool = True,
            k_feature_omics1: int = 10,
            k_feature_omics2: int = 10,
            k_cross_omics: int = 20,
            k_all_omics: int = 25,
            loss_weight: List[float] = [1, 1, 1, 1, 1],
            hidden_dim: int = 128,
            latent_dim: int = 64,
            device: torch.device = torch.device('cuda:0'),
            random_seed: int = 2020,
            strict_repro: bool = False,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            epochs: int = 300,
            gradient_clipping: float = 5.,
            embedding_key: str = 'emb_spcoral',
            rec_key: str = 'rec_spcoral',
            cross_key: str = 'cross_spcoral',
    ):
        set_random_seed(random_seed, strict_repro=strict_repro)

        self.embedding_key = embedding_key
        self.rec_key = rec_key
        self.cross_key = cross_key

        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2

        self.feature_omics1 = torch.from_numpy(adata_omics1.obsm['feat']).to(torch.float32).to(device)
        self.feature_omics2 = torch.from_numpy(adata_omics2.obsm['feat']).to(torch.float32).to(device)

        self.crds_omics1 = torch.from_numpy(self.adata_omics1.obsm[use_obsm]).to(torch.float32).to(device)
        self.crds_omics2 = torch.from_numpy(self.adata_omics2.obsm[use_obsm]).to(torch.float32).to(device)

        self.graph_omics1 = adata_to_dgl(
            self.adata_omics1,
            method=graph_method_single,
            k=k_spatial_omics1,
            radius=radius_spatial_omics1,
            use_obsm=use_obsm
        ).to(device)

        self.graph_omics2 = adata_to_dgl(
            self.adata_omics2,
            method=graph_method_single,
            k=k_spatial_omics2,
            radius=radius_spatial_omics2,
            use_obsm=use_obsm
        ).to(device)

        self.g_feature_omics1 = build_graph_feature(self.feature_omics1.cpu(), k_feature_omics1).to(device)
        self.g_feature_omics2 = build_graph_feature(self.feature_omics2.cpu(), k_feature_omics2).to(device)

        g_crd_cross = create_snn_adjacency_matrix(
                self.adata_omics1.obsm[use_obsm], self.adata_omics2.obsm[use_obsm], k_cross_omics # 10
            ).astype(np.float32)# .T

        if g_all_auto:
            row_sums = g_crd_cross.sum(axis=1)  # (m,)
            isolated_rows = np.where(row_sums == 0)[0]
            if len(isolated_rows) > 0:
                dists = np.sqrt(((self.adata_omics1.obsm[use_obsm][isolated_rows, np.newaxis] -
                                  self.adata_omics2.obsm[use_obsm]) ** 2).sum(axis=2))  # (isolated_m, n)
                nearest_cols = np.argmin(dists, axis=1)
                g_crd_cross[isolated_rows, nearest_cols] = 1
                col_indices = nearest_cols
                row_indices = isolated_rows
                g_crd_cross.T[col_indices, row_indices] = 1

            col_sums = g_crd_cross.sum(axis=0)  # (n,)
            isolated_cols = np.where(col_sums == 0)[0]
            if len(isolated_cols) > 0:
                dists = np.sqrt(((self.adata_omics2.obsm[use_obsm][isolated_cols, np.newaxis] -
                                  self.adata_omics1.obsm[use_obsm]) ** 2).sum(axis=2))  # (isolated_n, m)
                nearest_rows = np.argmin(dists, axis=1)
                g_crd_cross[nearest_rows, isolated_cols] = 1
                g_crd_cross.T[isolated_cols, nearest_rows] = 1

        self.g_crd_cross = torch.from_numpy(g_crd_cross).to(torch.float32).to(device)
        self.g_crd_cross = self.g_crd_cross.unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)

        if not g_all_auto:
            g_all = create_snn_adjacency_matrix(
                np.concatenate((self.adata_omics1.obsm[use_obsm], self.adata_omics2.obsm[use_obsm]), axis=0),
                np.concatenate((self.adata_omics1.obsm[use_obsm], self.adata_omics2.obsm[use_obsm]), axis=0),
                k_all_omics # 15
                ).astype(np.float32)
        else:
            m, n = g_crd_cross.shape

            adj_omics1 = np.zeros((m, m), dtype=np.float32)
            adj_omics2 = np.zeros((n, n), dtype=np.float32)

            src1, dst1 = self.graph_omics1.edges()
            adj_omics1[src1.cpu().numpy(), dst1.cpu().numpy()] = 1.0

            src2, dst2 = self.graph_omics2.edges()
            adj_omics2[src2.cpu().numpy(), dst2.cpu().numpy()] = 1.0

            g_all = np.zeros((m + n, m + n), dtype=np.float32)
            g_all[:m, :m] = adj_omics1
            g_all[m:, m:] = adj_omics2
            g_all[:m, m:] = g_crd_cross
            g_all[m:, :m] = g_crd_cross.T

            node_sums = g_all.sum(axis=1)
            isolated_nodes = np.where(node_sums == 0)[0]
            if len(isolated_nodes) > 0:
                all_coords = np.concatenate((self.adata_omics1.obsm[use_obsm],
                                             self.adata_omics2.obsm[use_obsm]), axis=0)
                dists = np.sqrt(((all_coords[isolated_nodes, np.newaxis] - all_coords) ** 2).sum(axis=2))
                np.fill_diagonal(dists, np.inf)
                nearest = np.argmin(dists, axis=1)
                g_all[isolated_nodes, nearest] = 1
                g_all[nearest, isolated_nodes] = 1

        src, dst = np.nonzero(g_all)
        src = src.tolist()
        dst = dst.tolist()

        self.g_all_dgl = dgl.graph((src, dst), num_nodes=g_all.shape[0]).to(device)
        self.g_all = torch.from_numpy(g_all).to(torch.float32).to(device)

        self.input_dim_omics1 = self.feature_omics1.shape[1]
        self.input_dim_omics2 = self.feature_omics2.shape[1]
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = 1
        self.dropout = 0.1

        self.loss_weight = loss_weight
        self.gradient_clipping = gradient_clipping
        self.device = device
        self.random_seed = random_seed
        self.strict_repro = strict_repro
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

    def train(self):
        """
        Train the cross-modal integration model.

        Returns
        -------
        tuple
            - adata_omics1 : anndata.AnnData
              Updated first omics object with latent embeddings and reconstructions.
            - adata_omics2 : anndata.AnnData
              Updated second omics object with the same.
            - loss_list : list of [total_loss, recon_loss, cross_loss, spatial_loss] per epoch
              Training loss history.
        """

        import gc

        set_random_seed(self.random_seed, self.strict_repro)

        self.model = CrossModalGAE(
            self.input_dim_omics1,
            self.input_dim_omics2,
            self.hidden_dim,
            self.latent_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()

        loss_list = []
        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.optimizer.zero_grad()

            h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, _, _, cross_omics1, cross_omics2 = self.model(
                self.graph_omics1,
                self.graph_omics2,
                self.g_feature_omics1,
                self.g_feature_omics2,
                self.feature_omics1,
                self.feature_omics2,
                self.g_crd_cross,
                self.g_all_dgl
            )

            loss_recon = self.loss_weight[0]*F.mse_loss(self.feature_omics1, feat_rec_omics1) + self.loss_weight[1]*F.mse_loss(self.feature_omics2, feat_rec_omics2)
            loss_cross = self.loss_weight[2]*F.mse_loss(h_omics1, cross_omics1) + self.loss_weight[3]*F.mse_loss(h_omics2, cross_omics2)

            loss_spatial = self.loss_weight[4] * F.mse_loss(torch.sigmoid(torch.cat((h_omics1, h_omics2)) @ torch.cat((h_omics1, h_omics2)).T), self.g_all)
            loss = loss_recon + loss_cross + loss_spatial # +  self.loss_GAN

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()

            del h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, cross_omics1, cross_omics2

            loss_list.append([loss.item(), loss_recon.item(), loss_cross.item(), loss_spatial.item()])

            del loss_recon, loss_cross, loss_spatial, loss

            gc.collect()
            torch.cuda.empty_cache()

        self.model.eval()

        with torch.no_grad():
            h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, feat_cross_omics2to1, feat_cross_omics1to2, _, _ = self.model(
                self.graph_omics1,
                self.graph_omics2,
                self.g_feature_omics1,
                self.g_feature_omics2,
                self.feature_omics1,
                self.feature_omics2,
                self.g_crd_cross,
                self.g_all_dgl
            )


            self.adata_omics1.obsm[self.embedding_key] = h_omics1.to('cpu').detach().numpy()
            self.adata_omics2.obsm[self.embedding_key] = h_omics2.to('cpu').detach().numpy()

            self.adata_omics1.obsm[self.rec_key] = feat_rec_omics1.to('cpu').detach().numpy()
            self.adata_omics2.obsm[self.rec_key] = feat_rec_omics2.to('cpu').detach().numpy()

            self.adata_omics1.obsm[self.cross_key] = feat_cross_omics1to2.to('cpu').detach().numpy()
            self.adata_omics2.obsm[self.cross_key] = feat_cross_omics2to1.to('cpu').detach().numpy()

        return self.adata_omics1, self.adata_omics2, loss_list


class integrate_model_block:
    """
        Block-wise cross-modal spatial omics integration model using graph autoencoders.

        This class enables scalable integration of large multi-modal spatial datasets by:
        1. Dividing the overlapping tissue region into grid blocks (from ``clipping_patch``).
        2. Preprocessing each block in parallel to build multiple graphs and tensors.
        3. Training a single shared CrossModalGAE model across all blocks with memory-efficient per-block updates.
        4. Supporting optional edge consistency loss between adjacent blocks for smoother global alignment.

        Parameters
        ----------
        clip_results : dict
            Output dictionary from ``clipping_patch`` containing:
            - 'x_clip', 'y_clip', 'x_num', 'y_num', 'x_retain', 'y_retain'
            - 'adata_omics1_clip_dict', 'adata_omics2_clip_dict' (block AnnData objects)
            - 'feature_omics1', 'feature_omics2' (input feature dimensions)
        is_norm : bool, optional (default: False)
            Whether to apply normalization in the model (passed to CrossModalGAE).
        hidden_dim : int, optional (default: 128)
            Hidden dimension in the GAE encoder/decoder layers.
        latent_dim : int, optional (default: 64)
            Dimension of the final joint latent embedding.
        device : torch.device, optional (default: torch.device('cuda:0'))
            Device used for training.
        random_seed : int, optional (default: 2020)
            Random seed for reproducibility.
        strict_repro : bool, optional (default: False)
            If True, enforces strict deterministic behavior (e.g., CUDA determinism).
        learning_rate : float, optional (default: 0.001)
            Optimizer learning rate.
        weight_decay : float, optional (default: 0.0001)
            Weight decay for Adam optimizer.
        loss_weight : list of float, optional
            Custom weights for loss components. If None, defaults to:
            - [1, 1, 0.5, 0.5, 1] when ``edge_loss=False``
            - [1, 1, 0.5, 0.5, 1, 1] when ``edge_loss=True`` (last weight for overlap loss)
        epochs : int, optional (default: 300)
            Number of training epochs.
        gradient_clipping : float, optional (default: 5.0)
            Maximum gradient norm for clipping.
        edge_loss : bool, optional (default: False)
            If True, adds consistency loss between overlapping edges of adjacent blocks.

        Notes
        -----
        - Designed to handle very large datasets that cannot fit into GPU memory as a whole.
        - Requires external helper functions: ``process_block``, ``preprogress_adata``, ``adata_to_dgl``,
          ``build_graph_feature``, ``create_snn_adjacency_matrix``.
        """

    def __init__(
            self,
            clip_results: Dict,
            is_norm: bool = False,
            hidden_dim: int = 128,
            latent_dim: int = 64,
            device: torch.device = torch.device('cuda:0'),
            random_seed: int = 2020,
            strict_repro: bool = False,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            loss_weight: Optional[List[float]] = None,
            epochs: int = 300,
            gradient_clipping: float = 5.0,
            edge_loss: bool = False,
    ):
        set_random_seed(random_seed, strict_repro=strict_repro)

        self.norm = is_norm
        self.x_clip = clip_results['x_clip']
        self.y_clip = clip_results['y_clip']
        self.x_num = clip_results['x_num']
        self.y_num = clip_results['y_num']
        self.x_retain = clip_results['x_retain']
        self.y_retain = clip_results['y_retain']
        self.adata_omics1_clip_dict = clip_results['adata_omics1_clip_dict']
        self.adata_omics2_clip_dict = clip_results['adata_omics2_clip_dict']
        self.input_dim_omics1 = clip_results['feature_omics1']
        self.input_dim_omics2 = clip_results['feature_omics2']

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = 1
        self.dropout = 0.1

        self.loss_weight = loss_weight
        self.gradient_clipping = gradient_clipping
        self.device = device
        self.random_seed = random_seed
        self.strict_repro = strict_repro
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.edge_loss = edge_loss

        if self.loss_weight is None:
            if not self.edge_loss:
                self.loss_weight = [1, 1, 0.5, 0.5, 1]
            else:
                self.loss_weight = [1, 1, 0.5, 0.5, 1, 1]

    def preprocess(
            self,
            graph_method_single: str,
            k_spatial_omics1: Optional[int] = None,
            radius_spatial_omics1: Optional[float] = None,
            k_spatial_omics2: Optional[int] = None,
            radius_spatial_omics2: Optional[float] = None,
            use_obsm: str = 'spatial',
            g_all_auto: bool = True,
            k_feature_omics1: int = 10,
            k_feature_omics2: int = 10,
            k_cross_omics: int = 20,
            k_all_omics: int = 25,
            num_processes: Optional[int] = None,
    ):
        """
        Preprocess data blocks in parallel, with optional user-specified number of processes.

        Parameters:
        -----------
        graph_method_single : str
            Method for constructing single-omics graph (e.g., 'knn').
        k_spatial_omics1 : int, optional
            Number of neighbors for omics1 spatial graph.
        radius_spatial_omics1 : float, optional
            Radius for omics1 spatial graph.
        k_spatial_omics2 : int, optional
            Number of neighbors for omics2 spatial graph.
        radius_spatial_omics2 : float, optional
            Radius for omics2 spatial graph.
        use_obsm : str, optional
            Key in adata.obsm for spatial coordinates (default: 'spatial').
        k_feature_omics1 : int, optional
            Number of neighbors for omics1 feature graph (default: 10).
        k_feature_omics2 : int, optional
            Number of neighbors for omics2 feature graph (default: 10).
        num_processes : int, optional
            Number of processes for parallel processing. If None, uses min(cpu_count(), task_count).
        """
        self.feature_omics1_dict = {}
        self.feature_omics2_dict = {}
        self.crd_omics1_dict = {}
        self.crd_omics2_dict = {}
        self.g_crd_cross_dict = {}
        self.g_all_dict = {}
        self.g_all_dgl_dict = {}
        self.g_spa_omics1_dict = {}
        self.g_spa_omics2_dict = {}
        self.g_feature_omics1_dict = {}
        self.g_feature_omics2_dict = {}
        self.loss_used_obs_omics1 = {}
        self.loss_used_obs_omics2 = {}
        if self.edge_loss:
            self.mask_up_omics1 = {}
            self.mask_up_omics2 = {}
            self.mask_down_omics1 = {}
            self.mask_down_omics2 = {}
            self.mask_right_omics1 = {}
            self.mask_right_omics2 = {}
            self.mask_left_omics1 = {}
            self.mask_left_omics2 = {}
            self.mask_barcode_omics1 = {}
            self.mask_barcode_omics2 = {}

        # Prepare arguments for parallel processing
        tasks = [(col, row, self.adata_omics1_clip_dict, self.adata_omics2_clip_dict)
                 for col in range(self.x_num) for row in range(self.y_num)]

        # Determine number of processes
        if num_processes is None:
            num_processes = min(cpu_count(), len(tasks))
        else:
            num_processes = min(num_processes, len(tasks), cpu_count())

        # Use multiprocessing Pool
        pool = Pool(processes=num_processes)

        # Create partial function to pass additional arguments
        process_block_partial = partial(
            process_block,
            graph_method_single=graph_method_single,
            k_spatial_omics1=k_spatial_omics1,
            radius_spatial_omics1=radius_spatial_omics1,
            k_spatial_omics2=k_spatial_omics2,
            radius_spatial_omics2=radius_spatial_omics2,
            use_obsm=use_obsm,
            k_feature_omics1=k_feature_omics1,
            k_feature_omics2=k_feature_omics2,
            k_cross_omics=k_cross_omics,
            k_all_comics=k_all_omics,
            g_all_auto=g_all_auto,
        )

        # Process blocks in parallel with progress bar
        for result in tqdm(pool.imap_unordered(process_block_partial, tasks),
                           total=len(tasks), desc=f"Preprocessing blocks with {num_processes} processes"):
            if result is not None:
                key, result_dict = result
                self.adata_omics1_clip_dict[key] = result_dict['adata_omics1']
                self.adata_omics2_clip_dict[key] = result_dict['adata_omics2']
                self.feature_omics1_dict[key] = result_dict['feature_omics1']
                self.feature_omics2_dict[key] = result_dict['feature_omics2']
                self.crd_omics1_dict[key] = result_dict['crds_omics1']
                self.crd_omics2_dict[key] = result_dict['crds_omics2']
                self.g_crd_cross_dict[key] = result_dict['g_crd_cross']
                self.g_all_dict[key] = result_dict['g_all']
                self.g_all_dgl_dict[key] = result_dict['g_all_dgl']
                self.g_spa_omics1_dict[key] = result_dict['g_spa_omics1']
                self.g_spa_omics2_dict[key] = result_dict['g_spa_omics2']
                self.g_feature_omics1_dict[key] = result_dict['g_feature_omics1']
                self.g_feature_omics2_dict[key] = result_dict['g_feature_omics2']
                self.loss_used_obs_omics1[key] = result_dict['loss_used_obs_omics1']
                self.loss_used_obs_omics2[key] = result_dict['loss_used_obs_omics2']
                if self.edge_loss:
                    self.mask_up_omics1[key] = result_dict['mask_up_omics1']
                    self.mask_up_omics2[key] = result_dict['mask_up_omics2']
                    self.mask_down_omics1[key] = result_dict['mask_down_omics1']
                    self.mask_down_omics2[key] = result_dict['mask_down_omics2']
                    self.mask_right_omics1[key] = result_dict['mask_right_omics1']
                    self.mask_right_omics2[key] = result_dict['mask_right_omics2']
                    self.mask_left_omics1[key] = result_dict['mask_left_omics1']
                    self.mask_left_omics2[key] = result_dict['mask_left_omics2']
                    self.mask_barcode_omics1[key] = {
                        'up': result_dict['adata_omics1'][result_dict['mask_up_omics1'], :].obs_names.tolist(),
                        'down': result_dict['adata_omics1'][result_dict['mask_down_omics1'], :].obs_names.tolist(),
                        'right': result_dict['adata_omics1'][result_dict['mask_right_omics1'], :].obs_names.tolist(),
                        'left': result_dict['adata_omics1'][result_dict['mask_left_omics1'], :].obs_names.tolist()
                    }
                    self.mask_barcode_omics2[key] = {
                        'up': result_dict['adata_omics2'][result_dict['mask_up_omics2'], :].obs_names.tolist(),
                        'down': result_dict['adata_omics2'][result_dict['mask_down_omics2'], :].obs_names.tolist(),
                        'right': result_dict['adata_omics2'][result_dict['mask_right_omics2'], :].obs_names.tolist(),
                        'left': result_dict['adata_omics2'][result_dict['mask_left_omics2'], :].obs_names.tolist()
                    }

        pool.close()
        pool.join()

    def train(self):
        """
        Train the shared CrossModalGAE model across all preprocessed spatial blocks.

        Returns
        -------
        list of list of float
            Training loss history with one entry per epoch. Each entry contains:
            - Total loss
            - Reconstruction loss (omics1 + omics2)
            - Cross-prediction loss
            - Spatial graph reconstruction loss
            - Overlap consistency loss (only if ``edge_loss=True``)
        """
        set_random_seed(self.random_seed, strict_repro=self.strict_repro)

        self.model = CrossModalGAE(
            self.input_dim_omics1,
            self.input_dim_omics2,
            self.hidden_dim,
            self.latent_dim,
            self.norm
        ).to(self.device, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.model.train()

        loss_list = []
        total_blocks = sum(1 for col in range(self.x_num) for row in range(self.y_num)
                           if str(col) + '_' + str(row) in self.adata_omics1_clip_dict
                           and str(col) + '_' + str(row) in self.adata_omics2_clip_dict)

        for epoch in tqdm(range(self.epochs), desc="Training"):
            self.optimizer.zero_grad()  # Clear gradients at the start of the epoch
            epoch_loss_recon = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            epoch_loss_cross = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            epoch_loss_spatial = torch.tensor(0.0, dtype=torch.float32, device=self.device)

            if self.edge_loss:
                epoch_loss_overlap = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                overlap_dict_up_omics1 = {}
                overlap_dict_up_omics2 = {}
                overlap_dict_right_omics1 = {}
                overlap_dict_right_omics2 = {}

            # Process each block individually
            for col in range(self.x_num):
                for row in range(self.y_num):
                    key = str(col) + '_' + str(row)
                    key_left = str(col - 1) + '_' + str(row)
                    key_down = str(col) + '_' + str(row - 1)

                    # Check if the key exists in the clip dictionaries
                    if key not in self.adata_omics1_clip_dict or key not in self.adata_omics2_clip_dict:
                        continue

                    # Load data for the current block
                    feature_omics1 = self.feature_omics1_dict[key].to(self.device, dtype=torch.float32)
                    feature_omics2 = self.feature_omics2_dict[key].to(self.device, dtype=torch.float32)
                    g_crd_cross = self.g_crd_cross_dict[key].to(self.device, dtype=torch.float32)
                    g_all = self.g_all_dict[key].to(self.device, dtype=torch.float32)
                    g_all_dgl = self.g_all_dgl_dict[key].to(self.device)
                    g_spa_omics1 = self.g_spa_omics1_dict[key].to(self.device)
                    g_spa_omics2 = self.g_spa_omics2_dict[key].to(self.device)
                    g_feature_omics1 = self.g_feature_omics1_dict[key].to(self.device)
                    g_feature_omics2 = self.g_feature_omics2_dict[key].to(self.device)

                    if self.edge_loss:
                        mask_up_omics1 = torch.tensor(self.mask_up_omics1[key], device=self.device, dtype=torch.bool)
                        mask_up_omics2 = torch.tensor(self.mask_up_omics2[key], device=self.device, dtype=torch.bool)
                        mask_down_omics1 = torch.tensor(self.mask_down_omics1[key], device=self.device,
                                                        dtype=torch.bool)
                        mask_down_omics2 = torch.tensor(self.mask_down_omics2[key], device=self.device,
                                                        dtype=torch.bool)
                        mask_right_omics1 = torch.tensor(self.mask_right_omics1[key], device=self.device,
                                                         dtype=torch.bool)
                        mask_right_omics2 = torch.tensor(self.mask_right_omics2[key], device=self.device,
                                                         dtype=torch.bool)
                        mask_left_omics1 = torch.tensor(self.mask_left_omics1[key], device=self.device,
                                                        dtype=torch.bool)
                        mask_left_omics2 = torch.tensor(self.mask_left_omics2[key], device=self.device,
                                                        dtype=torch.bool)

                    h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, feat_cross_omics1, feat_cross_omics2, cross_omics1, cross_omics2 = self.model(
                        g_spa_omics1,
                        g_spa_omics2,
                        g_feature_omics1,
                        g_feature_omics2,
                        feature_omics1,
                        feature_omics2,
                        g_crd_cross,
                        g_all_dgl,
                    )

                    del feat_cross_omics1, feat_cross_omics2

                    if self.edge_loss:
                        overlap_dict_up_omics1[key] = h_omics1[mask_up_omics1]
                        overlap_dict_up_omics2[key] = h_omics2[mask_up_omics2]
                        overlap_dict_right_omics1[key] = h_omics1[mask_right_omics1]
                        overlap_dict_right_omics2[key] = h_omics2[mask_right_omics2]

                    # Compute loss for the block
                    h_cat = torch.cat((h_omics1, h_omics2))
                    h_cat_masked = h_cat
                    g_all_masked = g_all

                    loss_recon = self.loss_weight[0]*F.mse_loss(feat_rec_omics1, feature_omics1) + \
                                 self.loss_weight[1]*F.mse_loss(feat_rec_omics2, feature_omics2)
                    loss_cross = self.loss_weight[2]*F.mse_loss(h_omics1, cross_omics1) + \
                                 self.loss_weight[3]*F.mse_loss(h_omics2, cross_omics2)

                    del feat_rec_omics1, feat_rec_omics2, cross_omics1, cross_omics2

                    loss_spatial = self.loss_weight[4]*F.mse_loss(torch.sigmoid(h_cat_masked @ h_cat_masked.T), g_all_masked)

                    if self.edge_loss:
                        if key_left in overlap_dict_right_omics1:
                            barcode_neigh_right_omics1 = self.mask_barcode_omics1[key_left]['right']
                            barcode_own_left_omics1 = self.mask_barcode_omics1[key]['left']
                            barcode_neigh_omics1 = np.isin(barcode_neigh_right_omics1, list(
                                set(barcode_neigh_right_omics1) & set(barcode_own_left_omics1)))
                            barcode_own_omics1 = np.isin(barcode_own_left_omics1, list(
                                set(barcode_neigh_right_omics1) & set(barcode_own_left_omics1)))
                            barcode_neigh_omics1 = torch.tensor(barcode_neigh_omics1, device=self.device,
                                                                dtype=torch.bool)
                            barcode_own_omics1 = torch.tensor(barcode_own_omics1, device=self.device, dtype=torch.bool)

                            barcode_neigh_right_omics2 = self.mask_barcode_omics2[key_left]['right']
                            barcode_own_left_omics2 = self.mask_barcode_omics2[key]['left']
                            barcode_neigh_omics2 = np.isin(barcode_neigh_right_omics2, list(
                                set(barcode_neigh_right_omics2) & set(barcode_own_left_omics2)))
                            barcode_own_omics2 = np.isin(barcode_own_left_omics2, list(
                                set(barcode_neigh_right_omics2) & set(barcode_own_left_omics2)))
                            barcode_neigh_omics2 = torch.tensor(barcode_neigh_omics2, device=self.device,
                                                                dtype=torch.bool)
                            barcode_own_omics2 = torch.tensor(barcode_own_omics2, device=self.device, dtype=torch.bool)

                            left_own_omics1 = h_omics1[mask_left_omics1][barcode_own_omics1]
                            left_own_omics2 = h_omics2[mask_left_omics2][barcode_own_omics2]
                            right_neigh_omics1 = overlap_dict_right_omics1[key_left][barcode_neigh_omics1]
                            right_neigh_omics2 = overlap_dict_right_omics2[key_left][barcode_neigh_omics2]

                            loss_overlap1 = F.mse_loss(left_own_omics1, right_neigh_omics1) + \
                                            F.mse_loss(left_own_omics2, right_neigh_omics2)

                            del overlap_dict_right_omics1[key_left]
                            del overlap_dict_right_omics2[key_left]
                            del barcode_neigh_omics1, barcode_own_omics1, barcode_neigh_omics2, barcode_own_omics2
                            del left_own_omics1, left_own_omics2, right_neigh_omics1, right_neigh_omics2
                        else:
                            loss_overlap1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)

                        if key_down in overlap_dict_up_omics1:
                            barcode_neigh_up_omics1 = self.mask_barcode_omics1[key_down]['up']
                            barcode_own_down_omics1 = self.mask_barcode_omics1[key]['down']
                            barcode_neigh_omics1 = np.isin(barcode_neigh_up_omics1, list(
                                set(barcode_neigh_up_omics1) & set(barcode_own_down_omics1)))
                            barcode_own_omics1 = np.isin(barcode_own_down_omics1, list(
                                set(barcode_neigh_up_omics1) & set(barcode_own_down_omics1)))
                            barcode_neigh_omics1 = torch.tensor(barcode_neigh_omics1, device=self.device,
                                                                dtype=torch.bool)
                            barcode_own_omics1 = torch.tensor(barcode_own_omics1, device=self.device, dtype=torch.bool)

                            barcode_neigh_up_omics2 = self.mask_barcode_omics2[key_down]['up']
                            barcode_own_down_omics2 = self.mask_barcode_omics2[key]['down']
                            barcode_neigh_omics2 = np.isin(barcode_neigh_up_omics2, list(
                                set(barcode_neigh_up_omics2) & set(barcode_own_down_omics2)))
                            barcode_own_omics2 = np.isin(barcode_own_down_omics2, list(
                                set(barcode_neigh_up_omics2) & set(barcode_own_down_omics2)))
                            barcode_neigh_omics2 = torch.tensor(barcode_neigh_omics2, device=self.device,
                                                                dtype=torch.bool)
                            barcode_own_omics2 = torch.tensor(barcode_own_omics2, device=self.device, dtype=torch.bool)

                            down_own_omics1 = h_omics1[mask_down_omics1][barcode_own_omics1]
                            down_own_omics2 = h_omics2[mask_down_omics2][barcode_own_omics2]
                            up_neigh_omics1 = overlap_dict_up_omics1[key_down][barcode_neigh_omics1]
                            up_neigh_omics2 = overlap_dict_up_omics2[key_down][barcode_neigh_omics2]

                            loss_overlap2 = F.mse_loss(down_own_omics1, up_neigh_omics1) + \
                                            F.mse_loss(down_own_omics2, up_neigh_omics2)

                            del overlap_dict_up_omics1[key_down]
                            del overlap_dict_up_omics2[key_down]
                            del barcode_neigh_omics1, barcode_own_omics1, barcode_neigh_omics2, barcode_own_omics2
                            del down_own_omics1, down_own_omics2, up_neigh_omics1, up_neigh_omics2
                        else:
                            loss_overlap2 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                        loss_overlap = self.loss_weight[5] * (loss_overlap1 + loss_overlap2)

                    # Accumulate loss for the block
                    if self.edge_loss:
                        block_loss = loss_recon + loss_cross + loss_spatial + loss_overlap
                    else:
                        block_loss = loss_recon + loss_cross + loss_spatial

                    # Perform backward pass immediately to release intermediate tensors
                    block_loss.backward()

                    # Accumulate detached losses for logging
                    epoch_loss_recon += loss_recon.detach()
                    epoch_loss_cross += loss_cross.detach()
                    epoch_loss_spatial += loss_spatial.detach()
                    if self.edge_loss:
                        epoch_loss_overlap += loss_overlap.detach()

                    # Clear intermediate tensors
                    del feature_omics1, feature_omics2, g_crd_cross, g_all, g_all_dgl
                    del g_spa_omics1, g_spa_omics2, g_feature_omics1, g_feature_omics2
                    del h_omics1, h_omics2
                    del h_cat, h_cat_masked, g_all_masked, block_loss, loss_recon, loss_cross, loss_spatial
                    if self.edge_loss:
                        del loss_overlap, loss_overlap1, loss_overlap2
                        del mask_up_omics1, mask_up_omics2
                        del mask_down_omics1, mask_down_omics2
                        del mask_right_omics1, mask_right_omics2
                        del mask_left_omics1, mask_left_omics2
                    torch.cuda.empty_cache()

            # Check if any blocks were processed
            if total_blocks == 0:
                print(f"Epoch {epoch + 1}: No valid blocks processed, skipping update.")
                continue

            # Perform optimization step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()

            if self.edge_loss:
                loss_list.append([
                    (epoch_loss_recon + epoch_loss_cross + epoch_loss_spatial + epoch_loss_overlap).item(),
                    epoch_loss_recon.item(),
                    epoch_loss_cross.item(),
                    epoch_loss_spatial.item(),
                    epoch_loss_overlap.item(),
                ])
            else:
                loss_list.append([
                    (epoch_loss_recon + epoch_loss_cross + epoch_loss_spatial).item(),
                    epoch_loss_recon.item(),
                    epoch_loss_cross.item(),
                    epoch_loss_spatial.item(),
                ])

        self.model.eval()

        h_omics1_dict = {}
        h_omics2_dict = {}

        rec_omics1_dict = {}
        rec_omics2_dict = {}

        cross_omics1_dict = {}
        cross_omics2_dict = {}
        with torch.no_grad():
            with tqdm(total=total_blocks, desc="Inference blocks") as block_pbar:
                for col in range(self.x_num):
                    for row in range(self.y_num):
                        key = str(col) + '_' + str(row)

                        # Check if the key exists in the clip dictionaries
                        if key not in self.adata_omics1_clip_dict or key not in self.adata_omics2_clip_dict:
                            continue

                        feature_omics1 = self.feature_omics1_dict[key].to(self.device, dtype=torch.float32)
                        feature_omics2 = self.feature_omics2_dict[key].to(self.device, dtype=torch.float32)
                        g_crd_cross = self.g_crd_cross_dict[key].to(self.device, dtype=torch.float32)
                        g_all_dgl = self.g_all_dgl_dict[key].to(self.device)
                        g_spa_omics1 = self.g_spa_omics1_dict[key].to(self.device)
                        g_spa_omics2 = self.g_spa_omics2_dict[key].to(self.device)
                        g_feature_omics1 = self.g_feature_omics1_dict[key].to(self.device)
                        g_feature_omics2 = self.g_feature_omics2_dict[key].to(self.device)

                        h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, feat_cross_omics2to1, feat_cross_omics1to2, _, _ = self.model(
                            g_spa_omics1,
                            g_spa_omics2,
                            g_feature_omics1,
                            g_feature_omics2,
                            feature_omics1,
                            feature_omics2,
                            g_crd_cross,
                            g_all_dgl
                        )

                        h_omics1_dict[key] = h_omics1.cpu()
                        h_omics2_dict[key] = h_omics2.cpu()

                        rec_omics1_dict[key] = feat_rec_omics1.cpu()
                        rec_omics2_dict[key] = feat_rec_omics2.cpu()

                        cross_omics1_dict[key] = feat_cross_omics1to2.cpu()
                        cross_omics2_dict[key] = feat_cross_omics2to1.cpu()

                        # Clear intermediate tensors
                        del feature_omics1, feature_omics2, g_crd_cross, g_all_dgl
                        del g_spa_omics1, g_spa_omics2, g_feature_omics1, g_feature_omics2
                        del h_omics1, h_omics2, feat_rec_omics1, feat_rec_omics2, feat_cross_omics2to1, feat_cross_omics1to2
                        torch.cuda.empty_cache()

                        block_pbar.update(1)

        self.h_omics1_dict = h_omics1_dict
        self.h_omics2_dict = h_omics2_dict
        self.rec_omics1_dict = rec_omics1_dict
        self.rec_omics2_dict = rec_omics2_dict
        self.cross_omics1_dict = cross_omics1_dict
        self.cross_omics2_dict = cross_omics2_dict
        self.loss_list = loss_list

    def map_results_to_adata(
            self,
            embedding_key: str = 'emb_spcoral',
            rec_key: str = 'rec_spcoral',
            cross_key: str = 'cross_spcoral',
    ) -> Tuple[anndata.AnnData, anndata.AnnData]:
        """
        Map block-wise model outputs back to full-resolution integrated AnnData objects.

        Parameters
        ----------
        embedding_key : str, optional (default: 'emb_spcoral')
            Key under which the joint latent embedding will be stored in ``.obsm``.
        rec_key : str, optional (default: 'rec_spcoral')
            Key for reconstructed modality-specific features in ``.obsm``.
        cross_key : str, optional (default: 'cross_spcoral')
            Key for cross-modality predicted features in ``.obsm``:
            - For omics1: predicted omics2 features from omics1 embedding
            - For omics2: predicted omics1 features from omics2 embedding

        Returns
        -------
        tuple of (anndata.AnnData, anndata.AnnData)
            - Full integrated AnnData for omics1 with added ``.obsm`` layers.
            - Full integrated AnnData for omics2 with added ``.obsm`` layers.
        """
        adata_omics1_list = []
        adata_omics2_list = []

        for key in tqdm(self.h_omics1_dict.keys(), desc="Mapping blocks"):
            # Get clipped adata and embeddings for the block
            adata_omics1_clip = self.adata_omics1_clip_dict[key]
            adata_omics2_clip = self.adata_omics2_clip_dict[key]

            loss_used_obs_omics1 = self.loss_used_obs_omics1[key]
            loss_used_obs_omics2 = self.loss_used_obs_omics2[key]

            adata_omics1_clip.obsm[embedding_key] = self.h_omics1_dict[key].numpy()
            adata_omics2_clip.obsm[embedding_key] = self.h_omics2_dict[key].numpy()
            adata_omics1_clip.obsm[rec_key] = self.rec_omics1_dict[key].numpy()
            adata_omics2_clip.obsm[rec_key] = self.rec_omics2_dict[key].numpy()
            adata_omics1_clip.obsm[cross_key] = self.cross_omics1_dict[key].numpy()
            adata_omics2_clip.obsm[cross_key] = self.cross_omics2_dict[key].numpy()

            adata_omics1_list.append(adata_omics1_clip[loss_used_obs_omics1, :].copy())
            adata_omics2_list.append(adata_omics2_clip[loss_used_obs_omics2, :].copy())

        adata_omics1 = sc.concat(adata_omics1_list, index_unique=None)
        adata_omics2 = sc.concat(adata_omics2_list, index_unique=None)

        return adata_omics1, adata_omics2


def process_block(
        args,
        graph_method_single,
        k_spatial_omics1,
        radius_spatial_omics1,
        k_spatial_omics2,
        radius_spatial_omics2,
        use_obsm,
        k_feature_omics1,
        k_feature_omics2,
        k_cross_omics,
        k_all_comics,
        g_all_auto,
    ):
    col, row, adata_omics1_clip_dict, adata_omics2_clip_dict = args
    key = str(col) + '_' + str(row)

    # Check if the key exists in the clip dictionaries
    if key not in adata_omics1_clip_dict or key not in adata_omics2_clip_dict:
        return None

    adata_omics1 = adata_omics1_clip_dict[key]
    adata_omics2 = adata_omics2_clip_dict[key]

    adata_omics1, adata_omics2 = preprogress_adata(adata_omics1, adata_omics2, k=k_cross_omics, use_obsm=use_obsm)

    feature_omics1 = torch.from_numpy(adata_omics1.obsm['feat']).to(torch.float32)
    feature_omics2 = torch.from_numpy(adata_omics2.obsm['feat']).to(torch.float32)

    crds_omics1 = torch.from_numpy(adata_omics1.obsm[use_obsm]).to(torch.float32)
    crds_omics2 = torch.from_numpy(adata_omics2.obsm[use_obsm]).to(torch.float32)

    graph_omics1 = adata_to_dgl(
        adata_omics1,
        method=graph_method_single,
        k=k_spatial_omics1,
        radius=radius_spatial_omics1,
        use_obsm=use_obsm
    )

    graph_omics2 = adata_to_dgl(
        adata_omics2,
        method=graph_method_single,
        k=k_spatial_omics2,
        radius=radius_spatial_omics2,
        use_obsm=use_obsm
    )

    g_feature_omics1 = build_graph_feature(feature_omics1, k_feature_omics1)
    g_feature_omics2 = build_graph_feature(feature_omics2, k_feature_omics2)

    g_crd_cross = create_snn_adjacency_matrix(
        adata_omics1.obsm[use_obsm], adata_omics2.obsm[use_obsm], k_cross_omics
    ).astype(np.float32)

    if g_all_auto:
        row_sums = g_crd_cross.sum(axis=1)  # (m,)
        isolated_rows = np.where(row_sums == 0)[0]
        if len(isolated_rows) > 0:
            dists = np.sqrt(((adata_omics1.obsm[use_obsm][isolated_rows, np.newaxis] -
                              adata_omics2.obsm[use_obsm]) ** 2).sum(axis=2))  # (isolated_m, n)
            nearest_cols = np.argmin(dists, axis=1)
            g_crd_cross[isolated_rows, nearest_cols] = 1
            col_indices = nearest_cols
            row_indices = isolated_rows
            g_crd_cross.T[col_indices, row_indices] = 1

        col_sums = g_crd_cross.sum(axis=0)  # (n,)
        isolated_cols = np.where(col_sums == 0)[0]
        if len(isolated_cols) > 0:
            dists = np.sqrt(((adata_omics2.obsm[use_obsm][isolated_cols, np.newaxis] -
                              adata_omics1.obsm[use_obsm]) ** 2).sum(axis=2))  # (isolated_n, m)
            nearest_rows = np.argmin(dists, axis=1)
            g_crd_cross[nearest_rows, isolated_cols] = 1
            g_crd_cross.T[isolated_cols, nearest_rows] = 1

    g_crd_cross = torch.from_numpy(g_crd_cross).to(torch.float32)
    g_crd_cross = g_crd_cross.unsqueeze(0).unsqueeze(0).to(torch.float32)

    if not g_all_auto:
        g_all = create_snn_adjacency_matrix(
            np.concatenate((adata_omics1.obsm[use_obsm], adata_omics2.obsm[use_obsm]), axis=0),
            np.concatenate((adata_omics1.obsm[use_obsm], adata_omics2.obsm[use_obsm]), axis=0),
            k_all_comics
        ).astype(np.float32)
    else:
        _, _, m, n = g_crd_cross.shape

        adj_omics1 = np.zeros((m, m), dtype=np.float32)
        adj_omics2 = np.zeros((n, n), dtype=np.float32)

        src1, dst1 = graph_omics1.edges()
        adj_omics1[src1.cpu().numpy(), dst1.cpu().numpy()] = 1.0

        src2, dst2 = graph_omics2.edges()
        adj_omics2[src2.cpu().numpy(), dst2.cpu().numpy()] = 1.0

        g_all = np.zeros((m + n, m + n), dtype=np.float32)
        g_all[:m, :m] = adj_omics1
        g_all[m:, m:] = adj_omics2
        g_all[:m, m:] = g_crd_cross
        g_all[m:, :m] = g_crd_cross.T

        node_sums = g_all.sum(axis=1)
        isolated_nodes = np.where(node_sums == 0)[0]
        if len(isolated_nodes) > 0:
            all_coords = np.concatenate((adata_omics1.obsm[use_obsm],
                                         adata_omics2.obsm[use_obsm]), axis=0)
            dists = np.sqrt(((all_coords[isolated_nodes, np.newaxis] - all_coords) ** 2).sum(axis=2))
            np.fill_diagonal(dists, np.inf)
            nearest = np.argmin(dists, axis=1)
            g_all[isolated_nodes, nearest] = 1
            g_all[nearest, isolated_nodes] = 1

    src, dst = np.nonzero(g_all)
    src = src.tolist()
    dst = dst.tolist()

    g_all_dgl = dgl.graph((src, dst), num_nodes=g_all.shape[0])
    g_all = torch.from_numpy(g_all).to(torch.float32)

    results_dict = {
        'adata_omics1': adata_omics1,
        'adata_omics2': adata_omics2,
        'feature_omics1': feature_omics1,
        'feature_omics2': feature_omics2,
        'crds_omics1': crds_omics1,
        'crds_omics2': crds_omics2,
        'g_crd_cross': g_crd_cross,
        'g_all': g_all,
        'g_all_dgl': g_all_dgl,
        'g_spa_omics1': graph_omics1,
        'g_spa_omics2': graph_omics2,
        'g_feature_omics1': g_feature_omics1,
        'g_feature_omics2': g_feature_omics2,
        'loss_used_obs_omics1': adata_omics1.obs['retain'].tolist(),
        'loss_used_obs_omics2': adata_omics2.obs['retain'].tolist(),
        'mask_up_omics1': adata_omics1.obs['mask_up'].tolist(),
        'mask_up_omics2': adata_omics2.obs['mask_up'].tolist(),
        'mask_down_omics1': adata_omics1.obs['mask_down'].tolist(),
        'mask_down_omics2': adata_omics2.obs['mask_down'].tolist(),
        'mask_right_omics1': adata_omics1.obs['mask_right'].tolist(),
        'mask_right_omics2': adata_omics2.obs['mask_right'].tolist(),
        'mask_left_omics1': adata_omics1.obs['mask_left'].tolist(),
        'mask_left_omics2': adata_omics2.obs['mask_left'].tolist(),
    }

    return (key, results_dict)

