# import
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
import torch
import dgl
import torch.nn as nn

# subfunction in dgl model

from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.spatial import Delaunay

def build_knn_graph(coords, k, n_obs):

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coords)
    adj = np.zeros((n_obs, n_obs))
    distances, indices = nbrs.kneighbors(coords)
    for i in range(n_obs):
        for j in range(k):
            adj[i, indices[i, j]] = 1
            adj[indices[i, j], i] = 1
    u, v = np.nonzero(adj)
    g = dgl.graph((u, v), num_nodes=n_obs)
    return g


def build_radius_graph(coords, radius, n_obs):

    adj = radius_neighbors_graph(coords, radius, mode='connectivity', include_self=False)
    u, v = adj.nonzero()
    g = dgl.graph((u, v), num_nodes=n_obs)
    return g


def build_delaunay_graph(coords, n_obs):

    tri = Delaunay(coords)
    adj = np.zeros((n_obs, n_obs))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                adj[simplex[i], simplex[j]] = 1
                adj[simplex[j], simplex[i]] = 1  # 无向图
    u, v = np.nonzero(adj)
    g = dgl.graph((u, v), num_nodes=n_obs)
    return g


def adata_to_dgl(adata, method='knn', k=10, radius=50, use_obsm='spatial'):

    if method == 'knn':
        g = build_knn_graph(adata.obsm[use_obsm], k=k, n_obs=adata.n_obs)
    elif method == 'radius':
        g = build_radius_graph(adata.obsm[use_obsm], radius=radius, n_obs=adata.n_obs)
    elif method == 'delaunay':
        g = build_delaunay_graph(adata.obsm[use_obsm], adata.n_obs)
    else:
        raise ValueError("Method must be one of 'knn', 'radius', or 'delaunay'")

    g.ndata['spatial'] = torch.tensor(adata.obsm[use_obsm], dtype=torch.float)

    return g

def multi_model_adj(adata_omics1, adata_omics2, method='knn', k=10, radius=50, use_obsm='spatial'):
    coords1 = adata_omics1.obsm[use_obsm]
    coords2 = adata_omics2.obsm[use_obsm]
    n_obs_1 = adata_omics1.n_obs
    n_obs_2 = adata_omics2.n_obs

    if method == 'knn':
        nbrs_1to2 = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(coords2)
        adj_1to2 = nbrs_1to2.kneighbors_graph(coords1, mode='connectivity')
        nbrs_2to1 = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(coords1)
        adj_2to1 = nbrs_2to1.kneighbors_graph(coords2, mode='connectivity')
        # omics1 -> omics2
        u_1to2, v_1to2 = adj_1to2.nonzero()
        v_1to2 = v_1to2 + n_obs_1
        # omics2 -> omics1
        u_2to1, v_2to1 = adj_2to1.nonzero()
        u_2to1 = u_2to1 + n_obs_1
        u = np.concatenate([u_1to2, v_2to1])
        v = np.concatenate([v_1to2, u_2to1])
        g = dgl.graph((u, v), num_nodes=n_obs_1 + n_obs_2)
    elif method == 'radius':
        coords = np.concatenate((coords1, coords2))
        g = build_radius_graph(coords, radius=radius, n_obs=n_obs_1 + n_obs_2)
    else:
        raise ValueError("Method must be one of 'knn' or 'radius'")

    adj_matrix = get_adj_nearest(g)
    adj_matrix = adj_matrix[0:n_obs_1, n_obs_1:n_obs_1 + n_obs_2]

    return adj_matrix

def get_adj_nearest(g):
    from scipy.sparse.csgraph import dijkstra

    adj = g.adj().to_dense().numpy()
    adj_nearest = dijkstra(adj)

    return np.array(adj_nearest)

def get_kpop_neighbors(adj_nearest, k_up, k_down):

    adj_kpop = adj_nearest.copy()

    adj_kpop[adj_nearest <= k_up] = 1.
    adj_kpop[adj_nearest <= k_down] = 0.
    adj_kpop[adj_nearest > k_up] = 0.
    np.fill_diagonal(adj_kpop,0.)

    return adj_kpop.astype(np.float32)

# cal morphology
def cal_morphology(adj, emb_features):

    neigh_features = torch.matmul(adj, emb_features) # (l, n, d)
    emb_features = emb_features.unsqueeze(0) # (1, n, d)

    emb_features_mean = emb_features.mean(dim=-1, keepdim=True)
    neigh_features_mean = neigh_features.mean(dim=-1, keepdim=True)

    emb_features_centered = emb_features - emb_features_mean  # (1, n, d)
    neigh_features_centered = neigh_features - neigh_features_mean  # (l, n, d)

    cov = (emb_features_centered * neigh_features_centered).sum(dim=-1)  # (l, n)
    emb_features_std = torch.sqrt((emb_features_centered ** 2).sum(dim=-1))  # (1, n)
    neigh_features_std = torch.sqrt((neigh_features_centered ** 2).sum(dim=-1))  # (l, n)

    corr = cov / (emb_features_std * neigh_features_std + 1e-8)

    return corr.T


def normalize_tensor(x):

    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)

    return (x - mean) / (std + 1e-8)


# FGW OT

from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import math
from sklearn import mixture
from scipy import stats
from scipy.optimize import minimize_scalar


def compute_cell_distances(X, spatial_coords, a=1):

    distances_spatial = pdist(spatial_coords, metric='euclidean')
    distances_X = pdist(X, metric='euclidean')

    distance_matrix_spatial = squareform(distances_spatial)
    # distance_matrix_spatial = distance_matrix_spatial**2
    distance_matrix_X = squareform(distances_X)

    distance_matrix_spatial = distance_matrix_spatial / np.max(distance_matrix_spatial)
    distance_matrix_X = distance_matrix_X / np.max(distance_matrix_X)

    return a * distance_matrix_spatial + (1 - a) * distance_matrix_X


def compute_euclidean_distance_with_normalization(expr_matrix1, expr_matrix2):

    if len(expr_matrix1.shape) == 1:
        expr_matrix1 = expr_matrix1.reshape(-1, 1)
    if len(expr_matrix2.shape) == 1:
        expr_matrix2 = expr_matrix2.reshape(-1, 1)

    scaler = StandardScaler()
    expr_matrix1_scaled = scaler.fit_transform(expr_matrix1)
    expr_matrix2_scaled = scaler.fit_transform(expr_matrix2)

    distance_matrix = cdist(expr_matrix1_scaled, expr_matrix2_scaled, metric='euclidean')

    return distance_matrix


def compute_affine_transform(source_points, target_points):
    if len(source_points) < 3 or len(target_points) < 3:
        raise ValueError("The pairs of source and target points must >= 3")

    A = []
    B = []

    for (x, y), (u, v) in zip(source_points, target_points):
        A.append([x, y, 1, 0, 0, 0])
        A.append([0, 0, 0, x, y, 1])
        B.append(u)
        B.append(v)

    A = np.array(A)
    B = np.array(B)

    # A * X = B
    X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

    affine_matrix = np.array([
        [X[0], X[1], X[2]],
        [X[3], X[4], X[5]],
        [0, 0, 1]
    ])

    return affine_matrix


def compute_rigid_transform(source_points, target_points):
    """
    Computes the rigid transformation matrix (rotation + translation) from source to target points using SVD.
    This assumes no reflection (proper rotation).

    Parameters:
    - source_points: Source coordinates (numpy array of shape (N, 2))
    - target_points: Target coordinates (numpy array of shape (N, 2))

    Returns:
    - rigid_matrix: 3x3 rigid transformation matrix
    """
    if len(source_points) < 2 or len(target_points) < 2:
        raise ValueError("At least two pairs of points are needed to calculate the rigid transformation matrix.")

    centroid_src = np.mean(source_points, axis=0)
    centroid_dst = np.mean(target_points, axis=0)

    src_centered = source_points - centroid_src
    dst_centered = target_points - centroid_dst

    H = src_centered.T @ dst_centered  # 2x2 matrix

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_dst - R @ centroid_src

    rigid_matrix = np.eye(3)
    rigid_matrix[:2, :2] = R
    rigid_matrix[:2, 2] = t

    return rigid_matrix


def apply_transform(T_matrix, source_points):
    if T_matrix.shape != (3, 3):
        raise ValueError("The shape of T_matrix must be (3, 3)")

    homogeneous_source_points = np.hstack([
        np.array(source_points),
        np.ones((len(source_points), 1))
    ])

    transformed_points = np.dot(homogeneous_source_points, T_matrix.T)

    transformed_points = transformed_points[:, :2] / transformed_points[:, 2][:, np.newaxis]

    return transformed_points


def calculate_transformation_matrix(crds_omics1, crds_omics2, index_omics1, index_omics2, n_iter, registration_method):
    if registration_method not in ['affine', 'rigid']:
        raise ValueError("registration_type must be 'affine' or 'rigid'")

    if n_iter == 0:

        src_crds = crds_omics1[index_omics1]
        dst_crds = crds_omics2[index_omics2]

        if registration_method == 'affine':
            T_matrix = compute_affine_transform(src_crds, dst_crds)
        elif registration_method == 'rigid':
            T_matrix = compute_rigid_transform(src_crds, dst_crds)

        return T_matrix, index_omics1, index_omics2


    for i in range(n_iter):
        dis = np.array([np.linalg.norm(crds_omics1[index_omics1[i]] - crds_omics2[index_omics2[i]]) for i in
                        range(len(index_omics1))])

        gmm2 = mixture.GaussianMixture(
            n_components=2, covariance_type="full", random_state=None
        ).fit(dis.reshape(-1, 1))
        cutoff = minimize_scalar(
            fun=stats.gaussian_kde(dis), bounds=sorted(gmm2.means_), method="bounded"
        ).x[0]

        min_std = math.sqrt(
            [gmm2.covariances_[0][0][0], gmm2.covariances_[1][0][0]][np.argmin([gmm2.means_[0][0], gmm2.means_[1][0]])])
        min_cut = np.min([gmm2.means_[0][0], gmm2.means_[1][0]]) - 2 * min_std
        max_cut = np.max([gmm2.means_[0][0], gmm2.means_[1][0]])

        index_pairs = np.intersect1d(np.where(dis > min_cut)[0], np.where(dis < max_cut)[0])
        index_omics1, index_omics2 = index_omics1[index_pairs], index_omics2[index_pairs]
        src_crds = crds_omics1[index_omics1]
        dst_crds = crds_omics2[index_omics2]

        if registration_method == 'affine':
            T_matrix = compute_affine_transform(src_crds, dst_crds)
        elif registration_method == 'rigid':
            T_matrix = compute_rigid_transform(src_crds, dst_crds)

        if i == 0:
            before_T = T_matrix
            continue
        if np.linalg.norm(T_matrix - before_T, ord=2) < 1:
            break
        else:
            before_T = T_matrix

    return T_matrix, index_omics1, index_omics2


def FGW_OT(
        feature_omics1,
        feature_omics2,
        share_feature_omics1,
        share_feature_omics2,
        spatial_omics1,
        spatial_omics2,
        a=1,
        b=0.8,
        random_state=None,
):
    from ot.gromov import semirelaxed_fused_gromov_wasserstein

    M = compute_euclidean_distance_with_normalization(share_feature_omics1, share_feature_omics2)
    C2 = compute_cell_distances(feature_omics1, spatial_omics1, a)
    C3 = compute_cell_distances(feature_omics2, spatial_omics2, a)

    OT_23, log_23 = semirelaxed_fused_gromov_wasserstein(
        M, C2, C3, symmetric=True, alpha=b, log=True, G0=None)

    sim_matrix = OT_23.T / np.sum(OT_23, axis=1)
    sim_matrix = sim_matrix.T

    return sim_matrix


def find_anchor(
        spatial_omics1,
        spatial_omics2,
        sim_matrix,
        n_iter=10,
        method='affine'
):

    pairs_list = []

    for i in range(sim_matrix.shape[0]):
        ind = np.argmax(sim_matrix[i, :])
        pairs_list.append([i, ind])

    pairs_list = np.array(pairs_list)

    T, omics1_index, omics2_index = calculate_transformation_matrix(
        spatial_omics1,
        spatial_omics2,
        pairs_list.T[0],
        pairs_list.T[1],
        n_iter=n_iter,
        registration_method=method
    )

    print(f'The number of anchors is {len(omics1_index)}')

    return T, omics1_index, omics2_index


def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.number_of_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = x.clone()
    feat = drop_feature(feat, feat_drop_rate)

    ng = dgl.graph([], device=graph.device)
    ng.add_nodes(n_node)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    ng.add_edges(nsrc, ndst)

    return ng, feat

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    # x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()
    mask_rates = torch.ones(E, device=graph.device) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def correlation(x, y):
    n_samples_x, n_features = x.shape
    n_samples_y = y.shape[0]

    mean_x = torch.mean(x, dim=0, keepdim=True)
    mean_y = torch.mean(y, dim=0, keepdim=True)
    x_centered = x - mean_x
    y_centered = y - mean_y

    n_samples = min(n_samples_x, n_samples_y)
    cov = torch.matmul(x_centered.t(), y_centered) / (n_samples - 1)

    std_x = torch.std(x, dim=0, unbiased=True)
    std_y = torch.std(y, dim=0, unbiased=True)
    std_matrix = std_x.view(-1, 1) @ std_y.view(1, -1)

    corr = cov / (std_matrix + 1e-8)

    corr = torch.clamp(corr, -1.0, 1.0)

    return corr

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def loss_rep_single_model(c, lambd):
    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()

    loss = on_diag + lambd * off_diag

    return loss

def loss_rep_cross_model(c, comm_dim, lambd):
    c_c = c[:comm_dim, :comm_dim]
    c_u = c[comm_dim:, comm_dim:]

    on_diag_c = (torch.diagonal(c_c) - 1).pow(2).mean()
    off_diag_c = off_diagonal(c_c).pow(2).mean()

    # on_diag_u = torch.diagonal(c_u).pow(2).mean()
    # off_diag_u = off_diagonal(c_u).pow(2).mean()

    loss_c = on_diag_c + lambd * off_diag_c
    # loss_u = on_diag_u + lambd * off_diag_u

    return loss_c # + loss_u

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def build_graph_feature(feature, k):
    num_nodes = len(feature)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(feature)
    distances, indices = nbrs.kneighbors(feature)

    src = []
    dst = []
    for i in range(len(feature)):
        for j in indices[i][1:]:
            src.append(i)
            dst.append(j)

    fg = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
    fg = dgl.add_self_loop(fg)

    return fg


