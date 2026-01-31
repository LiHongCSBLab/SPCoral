import numpy as np
import pandas as pd
import scanpy as sc
import geopandas as gpd
from libpysal.weights import Queen, KNN
from esda.moran import Moran_BV, Moran_Local_BV
from typing import Optional, Literal, Union, List
from shapely.geometry import Point
import scipy
from .mapping import mapping_slides
import anndata
from scipy.stats import pearsonr


__all__ = [
    'bivariate_moran',
    'bivariate_local_moran',
    'pearson'
]

def pearson(
        adata_omics1: anndata.AnnData,
        adata_omics2: anndata.AnnData,
        features_omics1: Union[List, str],
        features_omics2: Union[List, str],
        used_projection: Literal['omics1', 'omics2'] = 'omics2',
        used_rec: bool = True,
        rec_key: str = 'rec_spcoral',
        cross_key: str = 'cross_spcoral',
):
    if isinstance(features_omics1, str):
        features_omics1 = [features_omics1]
    if isinstance(features_omics2, str):
        features_omics2 = [features_omics2]

    if used_projection == 'omics2':
        adata_omics1_used = anndata.AnnData(adata_omics1.obsm[rec_key] if used_rec else adata_omics1.X, var=adata_omics1.var, obsm=adata_omics1.obsm)
        adata_omics2_used = anndata.AnnData(adata_omics1.obsm[cross_key], var=adata_omics2.var, obsm=adata_omics1.obsm)
    elif used_projection == 'omics1':
        adata_omics1_used = anndata.AnnData(adata_omics2.obsm[cross_key], var=adata_omics1.var, obsm=adata_omics2.obsm)
        adata_omics2_used = anndata.AnnData(adata_omics2.obsm[rec_key] if used_rec else adata_omics2.X, var=adata_omics2.var, obsm=adata_omics2.obsm)
    else:
        raise ValueError('used_projection must be one of "omics1" or "omics2"')

    res_dict = {}
    res_dict['features_omics1'] = features_omics1
    res_dict['features_omics2'] = features_omics2

    for i in features_omics1:
        for j in features_omics2:
            key = i + '_' + j
            x = adata_omics1_used[:, i].X.flatten()
            y = adata_omics2_used[:, j].X.flatten()
            corr, p_value = pearsonr(x, y)
            res_dict[key] = {
                'P': p_value,
                'corr': corr
            }

    return res_dict


def bivariate_moran(
        adata_omics1: anndata.AnnData,
        adata_omics2: anndata.AnnData,
        features_omics1: Union[List, str],
        features_omics2: Union[List, str],
        used_projection: Literal['omics1', 'omics2'] = 'omics2',
        used_rec: bool = True,
        rec_key: str = 'rec_spcoral',
        cross_key: str = 'cross_spcoral',
        n_neighbors: int = 10,
        use_obsm='spatial'
):
    if isinstance(features_omics1, str):
        features_omics1 = [features_omics1]
    if isinstance(features_omics2, str):
        features_omics2 = [features_omics2]

    if used_projection == 'omics2':
        adata_omics1_used = anndata.AnnData(adata_omics1.obsm[rec_key] if used_rec else adata_omics1.X, var=adata_omics1.var, obsm=adata_omics1.obsm)
        adata_omics2_used = anndata.AnnData(adata_omics1.obsm[cross_key], var=adata_omics2.var, obsm=adata_omics1.obsm)
    elif used_projection == 'omics1':
        adata_omics1_used = anndata.AnnData(adata_omics2.obsm[cross_key], var=adata_omics1.var, obsm=adata_omics2.obsm)
        adata_omics2_used = anndata.AnnData(adata_omics2.obsm[rec_key] if used_rec else adata_omics2.X, var=adata_omics2.var, obsm=adata_omics2.obsm)
    else:
        raise ValueError('used_projection must be one of "omics1" or "omics2"')

    spatial_coords = adata_omics1_used.obsm[use_obsm]
    geometry = [Point(xy) for xy in spatial_coords]

    res_dict = {}
    res_dict['features_omics1'] = features_omics1
    res_dict['features_omics2'] = features_omics2

    for feature1 in features_omics1:
        for feature2 in features_omics2:
            key = feature1 + '_' + feature2

            expr1 = adata_omics1_used[:, feature1].X.flatten()
            expr2 = adata_omics2_used[:, feature2].X.flatten()

            gdf = gpd.GeoDataFrame({
                'f1': expr1,
                'f2': expr2,
                'geometry': geometry
            })

            w = KNN.from_dataframe(gdf, k=n_neighbors)
            w.transform = 'r'
            moran_bv = Moran_BV(gdf['f1'], gdf['f2'], w)
            res_dict[key] = {
                'P': moran_bv.p_sim,
                'I': moran_bv.I
            }

    return res_dict


def bivariate_local_moran(
        adata_omics1: anndata.AnnData,
        adata_omics2: anndata.AnnData,
        features_omics1: Union[List, str],
        features_omics2: Union[List, str],
        used_projection: Literal['omics1', 'omics2'] = 'omics2',
        used_rec: bool = True,
        rec_key: str = 'rec_spcoral',
        cross_key: str = 'cross_spcoral',
        n_neighbors: int = 10,
        use_obsm='spatial'
):
    if isinstance(features_omics1, str):
        features_omics1 = [features_omics1]
    if isinstance(features_omics2, str):
        features_omics2 = [features_omics2]

    if used_projection == 'omics2':
        adata_omics1_used = anndata.AnnData(adata_omics1.obsm[rec_key] if used_rec else adata_omics1.X, var=adata_omics1.var, obsm=adata_omics1.obsm)
        adata_omics2_used = anndata.AnnData(adata_omics1.obsm[cross_key], var=adata_omics2.var, obsm=adata_omics1.obsm)
    elif used_projection == 'omics1':
        adata_omics1_used = anndata.AnnData(adata_omics2.obsm[cross_key], var=adata_omics1.var, obsm=adata_omics2.obsm)
        adata_omics2_used = anndata.AnnData(adata_omics2.obsm[rec_key] if used_rec else adata_omics2.X, var=adata_omics2.var, obsm=adata_omics2.obsm)
    else:
        raise ValueError('used_projection must be one of "omics1" or "omics2"')

    spatial_coords = adata_omics1_used.obsm[use_obsm]
    geometry = [Point(xy) for xy in spatial_coords]

    res_dict = {}
    res_dict['features_omics1'] = features_omics1
    res_dict['features_omics2'] = features_omics2

    for feature1 in features_omics1:
        for feature2 in features_omics2:
            key = feature1 + '_' + feature2

            expr1 = adata_omics1_used[:, feature1].X.flatten()
            expr2 = adata_omics2_used[:, feature2].X.flatten()

            gdf = gpd.GeoDataFrame({
                'gene1': expr1,
                'gene2': expr2,
                'geometry': geometry
            })

            w = KNN.from_dataframe(gdf, k=n_neighbors)
            w.transform = 'r'
            moran_local_bv = Moran_Local_BV(gdf['gene1'], gdf['gene2'], w)
            res_dict[key] = {
                'P': moran_local_bv.p_sim,
                'I': moran_local_bv.Is
            }

    return res_dict


if __name__ == '__main__':
    pass







