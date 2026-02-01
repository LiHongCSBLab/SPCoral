import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Literal, Union, Tuple
import anndata
import os
from sklearn.cluster import KMeans

__all__=[
    'cluster'
]

def _louvain(
        adata: anndata.AnnData,
        n_neighbors: int = 30,
        resolution: float = 0.5,
        emb_label='emb_spcoral',
        key_added: str = 'domain',
        **kwargs
):
    sc.pp.neighbors(adata, use_rep=emb_label, n_neighbors=n_neighbors)
    # louvain
    sc.tl.louvain(adata, flavor="vtraag", resolution=resolution, key_added=key_added, **kwargs)
    return adata

def _mclust(
        adata: anndata.AnnData,
        num_cluster: int,
        key_added: str = 'domain',
        modelNames='EEE',
        emb_label='emb_spcoral',
        random_seed: int = 2020,
        **kwargs
):
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    np.random.seed(random_seed)
    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[emb_label]), num_cluster, modelNames, **kwargs)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('str')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata


def _kmeans(
        adata: anndata.AnnData,
        num_cluster: int,
        key_added: str = 'domain',
        emb_label: str = 'emb_spcoral',
        random_seed: int = 2020,
        **kwargs
):
    np.random.seed(random_seed)

    kmeans = KMeans(
        n_clusters=num_cluster,
        random_state=random_seed,
        n_init='auto',
        **kwargs
    )
    labels = kmeans.fit_predict(adata.obsm[emb_label])

    adata.obs[key_added] = labels
    adata.obs[key_added] = adata.obs[key_added].astype('str')
    adata.obs[key_added] = adata.obs[key_added].astype('category')

    return adata


def cluster(
        adata_omics1: anndata.AnnData,
        adata_omics2: anndata.AnnData,
        emb_label: str = 'emb_spcoral',
        cluster_method: Literal['m_clust', 'louvain', 'kmeans'] = 'louvain',
        cluster_number: int = 10,
        cluster_key: str = 'domain',
        random_seed: int = 2020,
        resolution_louvain: float = 0.5,
        n_neighbors_louvain: int = 30,
        **kwargs
)-> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    Perform joint clustering on the integrated embeddings of two omics datasets.

    Concatenates the latent embeddings from both modalities, runs clustering on the combined space,
    and assigns the resulting cluster labels back to each individual AnnData object.

    Parameters
    ----------
    adata_omics1, adata_omics2 : anndata.AnnData
        Integrated AnnData objects containing joint embeddings in ``obsm[emb_label]``.
    emb_label : str, optional (default: 'emb_spcoral')
        Key in ``.obsm`` where the shared latent embedding is stored.
    cluster_method : {'mclust', 'louvain', 'kmeans'}, optional (default: 'mclust')
        Clustering algorithm to use on the joint embedding space.
    cluster_number : int, optional
        Number of clusters (required for 'mclust' and 'kmeans'; ignored for 'louvain').
    cluster_key : str, optional (default: 'domain')
        Key under which cluster labels will be stored in ``.obs``.
    random_seed : int, optional (default: 2020)
        Random seed for reproducibility.
    resolution_louvain : float, optional (default: 0.5)
        Resolution parameter for Louvain clustering.
    n_neighbors_louvain : int, optional (default: 30)
        Number of neighbors for KNN graph in Louvain.
    **kwargs
        Additional arguments passed to the selected clustering function.

    Returns
    -------
    tuple of (anndata.AnnData, anndata.AnnData)
        The input AnnData objects with cluster labels added to ``obs[cluster_key]``.

    Raises
    ------
    ValueError
        If ``cluster_method`` is invalid or ``cluster_number`` is missing for methods requiring it.
    """

    num_omics1 = adata_omics1.shape[0]
    num_omics2 = adata_omics2.shape[0]

    data = np.concatenate((adata_omics1.obsm[emb_label], adata_omics2.obsm[emb_label]), axis=0)
    adata_con = sc.AnnData(data)
    adata_con.obs['omics'] = ['omics1' for i in range(num_omics1)] + ['omics2' for i in range(num_omics2)]
    adata_con.obsm[emb_label] = data

    if cluster_method == 'louvain':
        adata_con = _louvain(
            adata_con,
            n_neighbors=n_neighbors_louvain,
            resolution=resolution_louvain,
            emb_label=emb_label,
            key_added=cluster_key,
            **kwargs
        )
    elif cluster_method == 'mclust':
        adata_con = _mclust(
            adata_con,
            num_cluster=cluster_number,
            key_added=cluster_key,
            emb_label=emb_label,
            random_seed=random_seed,
            **kwargs
        )
    elif cluster_method == 'kmeans':
        adata_con = _kmeans(
            adata_con,
            num_cluster=cluster_number,
            key_added=cluster_key,
            emb_label=emb_label,
            random_seed=random_seed,
            **kwargs
        )
    else:
        raise ValueError('cluster_method must be one of "louvain" or "mclust"')

    adata_omics1.obs[cluster_key] = adata_con.obs[cluster_key][:num_omics1].tolist()
    adata_omics2.obs[cluster_key] = adata_con.obs[cluster_key][num_omics1:].tolist()

    return adata_omics1, adata_omics2

if __name__ == '__main__':
    pass