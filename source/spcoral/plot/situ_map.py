import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Union, Any, Tuple, Sequence, Mapping, Dict, List
from os import PathLike
from ..utils import extract_spatial_region
from .color import _get_color, color_list_2
from ._3d import match_3D_multi
import matplotlib
import anndata as ad
import os

__all__ = [
    'show_marker',
    'show_cross_align',
    'show_cross_align_3D'
]

def show_marker(
    adata: ad.AnnData,
    marker: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: Optional[float] = None,
    use_obsm: str = 'spatial',
    x_region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    y_region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    palette: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = True,
    save: Optional[Union[str, os.PathLike]] = None,
    **kwargs: Dict[str, Any]
) -> Optional[matplotlib.axes.Axes]:
    """
    Plot spatial embedding for a given marker in an AnnData object.

    Parameters:
    -----------
    adata : anndata.AnnData
        The AnnData object containing the data.
    marker : str
        The gene or metadata column to color the embedding by.
    vmin : Optional[float], default None
        Minimum value for color scaling (continuous variables).
    vmax : Optional[float], default None
        Maximum value for color scaling (continuous variables).
    size : Optional[float], default None
        Size of the points in the scatter plot.
    use_obsm : str, default 'spatial'
        Key in adata.obsm containing the spatial coordinates.
    x_region : Optional[Tuple[Union[int, float], Union[int, float]]], default None
        Tuple of (min, max) for x-axis spatial region.
    y_region : Optional[Tuple[Union[int, float], Union[int, float]]], default None
        Tuple of (min, max) for y-axis spatial region.
    palette : Optional[str], default None
        Color palette for categorical variables (Matplotlib/Seaborn palette name).
    cmap : Optional[str], default None
        Colormap for continuous variables (Matplotlib colormap name).
    figsize : Optional[Tuple[float, float]], default None
        Figure size as (width, height).
    dpi : Optional[float], default None
        Figure resolution in dots per inch.
    title : Optional[str], default None
        Title of the plot.
    ax : Optional[matplotlib.axes.Axes], default None
        Matplotlib Axes object to plot on.
    show : bool, default True
        Whether to display the plot.
    save : Optional[Union[str, os.PathLike]], default None
        Path to save the plot.
    **kwargs : Dict[str, Any]
        Additional arguments passed to scanpy.pl.embedding.

    Returns:
    --------
    Optional[matplotlib.axes.Axes]
        The Matplotlib Axes object if show=False, else None.
    """
    if x_region is not None or y_region is not None:
        if x_region is None:
            x_region = (np.min(adata.obsm[use_obsm].T[0]), np.max(adata.obsm[use_obsm].T[0]))
        if y_region is None:
            y_region = (np.min(adata.obsm[use_obsm].T[1]), np.max(adata.obsm[use_obsm].T[1]))

        adata = extract_spatial_region(
            adata=adata,
            minx=x_region[0],
            miny=y_region[0],
            maxx=x_region[1],
            maxy=y_region[1],
            retainx=0,
            retainy=0,
            used_obsm=use_obsm
        )

    if ax is None:
        if figsize is None:
            alpha = 7 * (np.max(adata.obsm[use_obsm].T[1]) - np.min(adata.obsm[use_obsm].T[1])) / \
                    (np.max(adata.obsm[use_obsm].T[0]) - np.min(adata.obsm[use_obsm].T[0]))
            figsize = (7, alpha)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if size is None:
        size = 120000 / adata.shape[0]

    if marker in adata.var_names.tolist():
        exp = list(np.array(adata[:, marker].X).T[0])
        color_config = _get_color(exp)
        if cmap is None:
            cmap = color_config['cmap']
    elif marker in adata.obs.columns.tolist():
        exp = list(adata.obs[marker].tolist())
        color_config = _get_color(exp)
        if palette is None:
            palette = color_config['palette']
    else:
        raise ValueError("Input marker not found in var_names and obs.columns")

    if cmap is not None:
        ax = sc.pl.embedding(
            adata,
            basis=use_obsm,
            color=marker,
            size=size,
            ax=ax,
            cmap=cmap,
            show=False,
            title=title,
            vmax=vmax,
            vmin=vmin,
            **kwargs
        )
    else:
        ax = sc.pl.embedding(
            adata,
            basis=use_obsm,
            color=marker,
            size=size,
            ax=ax,
            palette=palette,
            show=False,
            title=title,
            vmax=vmax,
            vmin=vmin,
            **kwargs
        )

    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()
    else:
        return ax


def show_cross_align(
    adata_omics1: ad.AnnData,
    adata_omics2: ad.AnnData,
    omics1_use_obsm: str,
    omics2_use_obsm: str,
    omics1_name: str = 'omics1',
    omics2_name: str = 'omics2',
    size_omics1: Optional[float] = None,
    size_omics2: Optional[float] = None,
    alpha_omics1: Optional[float] = None,
    alpha_omics2: Optional[float] = None,
    palette: Optional[Union[str, List[str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    show: bool = True,
    save: Optional[Union[str, os.PathLike]] = None,
    **kwargs: Dict[str, Any]
) -> Optional[matplotlib.axes.Axes]:
    """
    Plot spatial alignment of two omics datasets.

    Parameters:
    -----------
    adata_omics1 : anndata.AnnData
        AnnData object for the first omics dataset.
    adata_omics2 : anndata.AnnData
        AnnData object for the second omics dataset.
    omics1_use_obsm : str
        Key in adata_omics1.obsm containing the spatial coordinates.
    omics2_use_obsm : str
        Key in adata_omics2.obsm containing the spatial coordinates.
    omics1_name : str, default 'omics1'
        Name for the first omics dataset.
    omics2_name : str, default 'omics2'
        Name for the second omics dataset.
    size : Optional[float], default None
        Size of the points in the scatter plot.
    palette : Optional[Union[str, List[str]]], default None
        Color palette for categorical variables (Matplotlib/Seaborn palette name or list of colors).
    figsize : Optional[Tuple[float, float]], default None
        Figure size as (width, height).
    dpi : Optional[float], default None
        Figure resolution in dots per inch.
    title : Optional[str], default None
        Title of the plot. Defaults to 'spatial alignment' if None.
    ax : Optional[matplotlib.axes.Axes], default None
        Matplotlib Axes object to plot on.
    show : bool, default True
        Whether to display the plot.
    save : Optional[Union[str, os.PathLike]], default None
        Path to save the plot.
    **kwargs : Dict[str, Any]
        Additional arguments passed to scanpy.pl.embedding.

    Returns:
    --------
    Optional[matplotlib.axes.Axes]
        The Matplotlib Axes object if show=False, else None.
    """
    adata_omics1 = adata_omics1.copy()
    adata_omics2 = adata_omics2.copy()

    adata_omics1.obsm['spatial'] = adata_omics1.obsm[omics1_use_obsm]
    adata_omics2.obsm['spatial'] = adata_omics2.obsm[omics2_use_obsm]
    adata_omics1.obs['sample'] = omics1_name
    adata_omics2.obs['sample'] = omics2_name

    adata_con = adata_omics1.concatenate(adata_omics2)

    if title is None:
        title = 'spatial alignment'

    if palette is None:
        palette = color_list_2

    if ax is None:
        if figsize is None:
            alpha = 7 * (np.max(adata_con.obsm['spatial'].T[1]) - np.min(adata_con.obsm['spatial'].T[1])) / \
                    (np.max(adata_con.obsm['spatial'].T[0]) - np.min(adata_con.obsm['spatial'].T[0]))
            figsize = (7, alpha)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if size_omics1 is None:
        size_omics1 = 120000 / adata_omics1.shape[0]
    if size_omics2 is None:
        size_omics2 = 120000 / adata_omics2.shape[0]

    if alpha_omics1 is None:
        alpha_omics1 = 0.8
    if alpha_omics2 is None:
        alpha_omics2 = 0.8

    size = [size_omics1 for i in range(adata_omics1.shape[0])] + [size_omics2 for i in range(adata_omics2.shape[0])]
    alpha = [alpha_omics1 for i in range(adata_omics1.shape[0])] + [alpha_omics2 for i in range(adata_omics2.shape[0])]

    ax = sc.pl.embedding(
        adata_con,
        basis='spatial',
        color='sample',
        size=size,
        alpha=alpha,
        ax=ax,
        palette=palette,
        show=False,
        title=title,
        **kwargs
    )

    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()
    else:
        return ax


def show_cross_align_3D(
        adata_omics1: ad.AnnData,
        adata_omics2: ad.AnnData,
        registering_parameters: Dict,
        subsample_size: Optional[int] = 300,
        color_based: Optional[str] = None,
        omics1_use_obsm: str = 'spatial',
        omics2_use_obsm: str = 'spatial',
        size_omics1: float = 1.5,
        size_omics2: float = 1.5,
        conf_cutoff: Optional[float] = 0,
        line_width: Optional[float] = 0.3,
        line_color: Optional[str] = "grey",
        line_alpha: Optional[float] = 0.7,
        hide_axis: Optional[bool] = True,
        show_error: Optional[bool] = True,
        show_celltype: Optional[bool] = False,
        color: Union[List, Dict, None] = None,
        cmap: Optional[str] = "Reds",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        title: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        show: bool = True,
        save: Optional[Union[str, os.PathLike]] = None,
        **kwargs: Dict[str, Any]
) -> Optional[matplotlib.axes.Axes]:
    """
    Visualize the 3D alignment of two omics datasets with spatial coordinates and cell matching.

    This function creates a 3D scatter plot to display the spatial alignment of two datasets (e.g., omics data)
    with optional lines connecting matched cells. It supports coloring by metadata (e.g., cell types), scaling
    coordinates, and visualizing matching errors.

    Parameters
    ----------
    adata_omics1 : ad.AnnData
        AnnData object for the first omics dataset (reference dataset).
    adata_omics2 : ad.AnnData
        AnnData object for the second omics dataset (target dataset).
    registering_parameters : Dict
        Dictionary containing matching parameters, where keys 1 and 2 correspond to indices of matched cells.
    subsample_size : int, optional
        Number of cell pairs to subsample for visualization (default: 300).
    color_based : str, optional
        Column name in adata_omics*.obs to color points by (e.g., 'celltype'). If None, no coloring is applied.
    omics1_use_obsm : str, optional
        Key in adata_omics1.obsm for spatial coordinates (default: 'spatial').
    omics2_use_obsm : str, optional
        Key in adata_omics2.obsm for spatial coordinates (default: 'spatial').
    size_omics1 : float, optional
        Size of scatter points for the first dataset (default: 1.5).
    size_omics2 : float, optional
        Size of scatter points for the second dataset (default: 1.5).
    conf_cutoff : float, optional
        Confidence threshold for displaying matched pairs (default: 0).
    line_width : float, optional
        Width of lines connecting matched cells (default: 0.3).
    line_color : str, optional
        Color of lines connecting matched cells (default: 'grey').
    line_alpha : float, optional
        Transparency of lines connecting matched cells (default: 0.7).
    hide_axis : bool, optional
        If True, hide plot axes (default: True).
    show_error : bool, optional
        If True, color lines differently for mismatched cell types (requires meta) (default: True).
    show_celltype : bool, optional
        If True, color lines by cell type for matched pairs (requires meta) (default: False).
    cmap : str, optional
        Colormap for visualizing expression or metadata (default: 'Reds').
    figsize : Tuple[float, float], optional
        Figure size as (width, height). If None, defaults to (7, 7).
    dpi : float, optional
        Resolution of the figure in dots per inch. If None, uses default.
    title : str, optional
        Title of the plot. If None, no title is set.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes are created.
    show : bool, optional
        If True, display the plot (default: True).
    save : str or os.PathLike, optional
        File path to save the plot. If None, the plot is not saved.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to `match_3D_multi`.

    Returns
    -------
    matplotlib.axes.Axes, optional
        The axes object containing the plot, returned only if `show` is False.

    Notes
    -----
    - The function assumes `adata_omics1` and `adata_omics2` have spatial coordinates in `.obsm`.
    - The `registering_parameters` dictionary should contain matching indices under keys 1 and 2.
    - If `color_based` is provided, it must exist in both datasets' `.obs`.
    """
    adata1_df = pd.DataFrame({'index': range(adata_omics1.shape[0]),
                              'x': adata_omics1.obsm[omics1_use_obsm][:, 0],
                              'y': adata_omics1.obsm[omics1_use_obsm][:, 1]}
                             )
    adata2_df = pd.DataFrame({'index': range(adata_omics2.shape[0]),
                              'x': adata_omics2.obsm[omics2_use_obsm][:, 0],
                              'y': adata_omics2.obsm[omics2_use_obsm][:, 1]}
                             )

    if color_based is not None:
        adata1_df[color_based] = adata_omics1.obs[color_based].tolist()
        adata2_df[color_based] = adata_omics2.obs[color_based].tolist()

    points_size = [size_omics1, size_omics2]

    matching = np.array((registering_parameters[1], registering_parameters[2]))

    if ax is None:
        if figsize is None:
            figsize = (7, 7)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    multi_align = match_3D_multi(adata1_df, adata2_df, matching, meta=color_based,
                                 scale_coordinate=True, subsample_size=subsample_size,
                                 **kwargs
                                 )

    ax = multi_align.draw_3D(
        point_size=points_size,
        hide_axis=hide_axis,
        ax=ax,
        conf_cutoff=conf_cutoff,
        line_width=line_width,
        line_color=line_color,
        line_alpha=line_alpha,
        show_error=show_error,
        show_celltype=show_celltype,
        color=color,
        cmap=cmap,
    )

    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax





