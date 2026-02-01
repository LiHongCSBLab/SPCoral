import numpy as np
import pandas as pd
import anndata as ad
from libpysal.weights import DistanceBand, KNN
from esda.moran import Moran
from tqdm import tqdm
from typing import Optional, Union, Literal, List, Tuple
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial

__all__ = [
    'calculate_global_moran',
]

def _calculate_single_gene_moran(
        gene_expression: np.ndarray,
        w: any,
        permutations: int = 0,
        gene_idx: Optional[int] = None
) -> Tuple[float, float]:

    try:
        if hasattr(gene_expression, 'toarray'):
            gene_expression = gene_expression.toarray().flatten()
        elif hasattr(gene_expression, 'A1'):
            gene_expression = gene_expression.A1

        # moran index
        moran = Moran(gene_expression, w, permutations=permutations)
        moran_i = moran.I
        moran_pval = moran.p_sim if permutations > 0 else moran.p_norm

        return (moran_i, moran_pval)

    except Exception as e:
        if gene_idx is not None:
            print(f"calculate {gene_idx} error: {e}")
        return (np.nan, np.nan)


def calculate_global_moran(
        adata: ad.AnnData,
        spatial_coords_key: str = 'spatial',
        weight_method: Literal['knn', 'distance'] = 'knn',
        k: int = 6,
        distance_threshold: Optional[float] = None,
        use_layer: Optional[str] = None,
        permutations: int = 0,
        multiple_testing_correction: Optional[Literal['fdr_bh', 'bonferroni']] = 'fdr_bh',
        alpha: float = 0.05,
        I_cut: float = None,
        n_jobs: int = -1,
        batch_size: Optional[int] = None,
        verbose: bool = True
) -> ad.AnnData:
    """
    Compute global Moran's I for each gene and return a new AnnData object containing only genes
    with significant spatial autocorrelation (p-value < alpha).

    Supports parallel computation for faster processing on large datasets.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object containing gene expression data and spatial coordinates.
    spatial_coords_key : str, optional (default: 'spatial')
        Key in ``.obsm`` where spatial coordinates are stored.
    weight_method : {'knn', 'distance'}, optional (default: 'knn')
        Method to construct the spatial weights matrix:
        - 'knn': K-nearest neighbors.
        - 'distance': Distance threshold (inverse distance weighting within threshold).
    k : int, optional (default: 6)
        Number of nearest neighbors when ``weight_method='knn'``.
    distance_threshold : float, optional
        Distance threshold when ``weight_method='distance'``. If None, automatically set to
        1.5 × median pairwise distance.
    use_layer : str, optional
        Key in ``.layers`` to use for expression data. If None, uses ``.X``.
    permutations : int, optional (default: 0)
        Number of permutations for permutation test. If 0, uses normal approximation for p-values;
        if >0, uses permutation test.
    multiple_testing_correction : {'fdr_bh', 'bonferroni'}, optional (default: 'fdr_bh')
        Multiple testing correction method. If None, no correction is applied.
    alpha : float, optional (default: 0.05)
        Significance level threshold.
    I_cut : float, optional
        Optional lower threshold for Moran's I value. Only genes with I > ``I_cut`` are considered
        significant (in addition to p-value < alpha).
    n_jobs : int, optional (default: -1)
        Number of parallel jobs. -1 uses all available CPU cores; 1 disables parallelism.
    batch_size : int, optional
        Number of genes processed per worker batch. If None, automatically determined.
    verbose : bool, optional (default: True)
        Whether to display progress and information messages.

    Returns
    -------
    AnnData
        New AnnData object containing only genes with significant spatial autocorrelation.
        The ``.var`` dataframe includes columns:
        - 'moran_i': Moran's I statistic.
        - 'moran_pval': raw p-value.
        - 'moran_pval_<method>' (if correction applied): corrected p-value.

    """

    if spatial_coords_key not in adata.obsm:
        raise ValueError(
            f"Spatial coordinates key '{spatial_coords_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    coords = adata.obsm[spatial_coords_key]

    if weight_method == 'knn':
        w = KNN(coords, k=k)
        if verbose:
            print(f"Building spatial weights matrix using K-nearest neighbors (k={k})")
    elif weight_method == 'distance':
        if distance_threshold is None:
            from scipy.spatial.distance import pdist
            distances = pdist(coords)
            distance_threshold = np.median(distances) * 1.5
            if verbose:
                print(f"Automatically computed distance threshold: {distance_threshold:.2f}")
        w = DistanceBand(coords, threshold=distance_threshold, binary=False)
        if verbose:
            print(f"Building spatial weights matrix using distance threshold: {distance_threshold}")
    else:
        raise ValueError("weight_method must be 'knn' or 'distance'")

    w.transform = 'r'  # row-standardize weights

    if use_layer is not None:
        if use_layer not in adata.layers:
            raise ValueError(
                f"Specified layer '{use_layer}' not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        data_matrix = adata.layers[use_layer]
        if verbose:
            print(f"Using expression data from layer: {use_layer}")
    else:
        data_matrix = adata.X
        if verbose:
            print("Using default expression matrix (.X)")

    n_genes = data_matrix.shape[1]

    if verbose:
        print(f"Starting computation of global Moran's I for {n_genes} genes...")
        print(f"Using {mp.cpu_count() if n_jobs == -1 else n_jobs} parallel processes")

    gene_data = [data_matrix[:, i] for i in range(n_genes)]

    moran_func = partial(
        _calculate_single_gene_moran,
        w=w,
        permutations=permutations
    )

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if batch_size is None:
        batch_size = max(1, n_genes // (n_jobs * 10))
        if verbose:
            print(f"Automatically set batch size: {batch_size}")

    if n_jobs == 1:
        if verbose:
            results = []
            for i in tqdm(range(n_genes), desc="Computing Moran's I"):
                result = _calculate_single_gene_moran(gene_data[i], w, permutations, i)
                results.append(result)
        else:
            results = [moran_func(gene_expr, gene_idx=i) for i, gene_expr in enumerate(gene_data)]
    else:
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=10 if verbose else 0)(
            delayed(moran_func)(gene_expr, gene_idx=i)
            for i, gene_expr in enumerate(gene_data)
        )

    moran_i, moran_pval = zip(*results)
    moran_i = np.array(moran_i)
    moran_pval = np.array(moran_pval)

    adata.var['moran_i'] = moran_i
    adata.var['moran_pval'] = moran_pval

    if multiple_testing_correction:
        valid_indices = ~np.isnan(moran_pval)
        pvals_to_correct = moran_pval[valid_indices]

        if len(pvals_to_correct) > 0:
            reject, pvals_corrected, _, _ = multipletests(
                pvals_to_correct,
                alpha=alpha,
                method=multiple_testing_correction
            )

            pvals_corrected_full = np.full_like(moran_pval, np.nan, dtype=float)
            pvals_corrected_full[valid_indices] = pvals_corrected

            correction_name = f'moran_pval_{multiple_testing_correction}'
            adata.var[correction_name] = pvals_corrected_full

            if verbose:
                n_corrected_sig = np.sum(pvals_corrected < alpha)
                print(
                    f"After multiple testing correction ({multiple_testing_correction}), "
                    f"number of significant genes: {n_corrected_sig}"
                )

    if multiple_testing_correction:
        pval_col = f'moran_pval_{multiple_testing_correction}'
    else:
        pval_col = 'moran_pval'

    if I_cut is not None:
        significant_mask = (
                (adata.var[pval_col] < alpha) &
                (~np.isnan(adata.var[pval_col])) &
                (adata.var['moran_i'] > I_cut)
        )
    else:
        significant_mask = (
                (adata.var[pval_col] < alpha) &
                (~np.isnan(adata.var[pval_col]))
        )

    significant_genes = adata.var_names[significant_mask]

    if verbose:
        n_sig = len(significant_genes)
        print(f"Found {n_sig} genes with significant spatial autocorrelation (p < {alpha})")
        if n_sig > 0:
            # Get top 10 genes with highest Moran's I among significant ones
            sig_indices = np.where(significant_mask)[0]
            top_indices = sig_indices[np.argsort(-moran_i[sig_indices])[:min(10, n_sig)]]
            top_genes = adata.var_names[top_indices]
            print("Top genes by Moran's I:")
            for i, gene in enumerate(top_genes):
                idx = np.where(adata.var_names == gene)[0][0]
                print(
                    f"  {i + 1}. {gene}: I={moran_i[idx]:.3f}, "
                    f"p={'{:.3e}'.format(adata.var.loc[gene, pval_col])}"
                )

    adata_sig = adata[:, significant_genes].copy()

    moran_cols = ['moran_i', 'moran_pval']
    if multiple_testing_correction:
        moran_cols.append(f'moran_pval_{multiple_testing_correction}')

    adata_sig.var[moran_cols] = adata.var.loc[significant_genes, moran_cols]

    return adata_sig