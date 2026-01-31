import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Union, Any, Tuple, Sequence, Mapping, Dict, List, Literal
import matplotlib
import os
from .color import _get_color, color_list_2
from ._sankey import sankey
from ._heatmap import _heatmap_with_dendrogram_and_bar

__all__ = [
    'show_cross_anchor_heatmap',
    'show_cross_anchor_Sankey',
    'show_integrate_umap'
]


def show_cross_anchor_heatmap(
        adata_omics1: ad.AnnData,
        adata_omics2: ad.AnnData,
        registering_parameters: Dict,
        celltype_label: Optional[str] = None,
        heatmap_cmap: str = "viridis",
        bar_palette: Optional[Union[List, Dict]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        title: Optional[str] = None,
        norm_axis: Optional[Literal[0, 1]] = None,
        show: bool = True,
        save: Optional[Union[str, os.PathLike]] = None,
        omics1_bar_kwargs: Optional[Dict] = None,
        omics2_bar_kwargs: Optional[Dict] = None,
        **kwargs: Any,
):
    """
    Visualize the cross-modality anchor correspondence as a heatmap with optional cell-type side bars.

    Parameters
    ----------
    adata_omics1 : anndata.AnnData
        First (source) omics AnnData object containing anchor spots.
    adata_omics2 : anndata.AnnData
        Second (target) omics AnnData object containing corresponding anchor spots.
    registering_parameters : dict
        Dictionary or list from registration output, expected to contain:
        - Index 1: array of anchor indices in ``adata_omics1``
        - Index 2: array of corresponding anchor indices in ``adata_omics2``
        (typically ``registering_parameters = [T, omics1_index, omics2_index]``)
    celltype_label : str, optional
        Column name in ``.obs`` containing cell-type or cluster annotations.
        If None, raw counts without labels are used (indices as categories).
    heatmap_cmap : str, optional (default: 'viridis')
        Colormap for the central correspondence heatmap.
    bar_palette : list or dict, optional
        Custom color palette for cell-type side bars.
        If list, colors are assigned sequentially; if dict, maps cell types to colors.
        If None, automatically generated using ``_get_color``.
    figsize : tuple of (float, float), optional
        Figure size passed to matplotlib.
    dpi : float, optional
        Figure resolution.
    title : str, optional
        Figure title.
    norm_axis : {0, 1}, optional
        Axis along which to normalize heatmap values to proportions:
        - 0: row-wise (proportion of omics2 types per omics1 type)
        - 1: column-wise
        - None: raw counts
    show : bool, optional (default: True)
        Whether to display the figure immediately.
    save : str or PathLike, optional
        Path to save the figure. If provided, figure is saved (format inferred from extension).
    omics1_bar_kwargs, omics2_bar_kwargs : dict, optional
        Additional keyword arguments passed to the horizontal/vertical side bar plots.
    **kwargs : Any
        Additional arguments passed to the underlying ``_heatmap_with_dendrogram_and_bar`` function.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, axes) or None
        Figure and axes if ``show=False``; otherwise None (figure is displayed).

    """

    anchor_omics1 = adata_omics1.obs[celltype_label][registering_parameters[1]].tolist()
    anchor_omics2 = adata_omics2.obs[celltype_label][registering_parameters[2]].tolist()

    anchor_set_omics1 = list(set(anchor_omics1))
    anchor_set_omics2 = list(set(anchor_omics2))

    color_list = list(set(anchor_omics1) | set(anchor_omics2))

    if bar_palette is None:
        bar_palette = _get_color(color_list)['palette']

    if isinstance(bar_palette, list):
        bar_palette = dict(zip(bar_palette[0: len(color_list)], color_list))

    data = pd.DataFrame(np.zeros((len(anchor_set_omics1), len(anchor_set_omics2))),index=anchor_set_omics1, columns=anchor_set_omics2)

    data_count = pd.DataFrame({
        'omics1': anchor_omics1,
        'omics2': anchor_omics2,
    })

    dict_omics1 = {clu: i for i, clu in enumerate(anchor_set_omics1)}
    dict_omics2 = {clu: i for i, clu in enumerate(anchor_set_omics2)}

    fig, axes = _heatmap_with_dendrogram_and_bar(
        data,
        data_count=data_count,
        x_label='omics1',
        y_label='omics2',
        x_map=dict_omics1,
        y_map=dict_omics2,
        x_dendrogram=False,
        y_dendrogram=False,
        norm_axis=norm_axis,
        method='proportion',
        x_bar=True,
        y_bar=True,
        palette=bar_palette,
        cmap=heatmap_cmap,
        title=title,
        figsize=figsize,
        dpi=dpi,
        xbar_kwags=omics1_bar_kwargs,
        ybar_kwags=omics2_bar_kwargs,
        **kwargs
    )

    if title is not None:
        fig.title(title)
    if save is not None:
        fig.savefig(save)
    if show:
        fig.show()
    else:
        return fig, axes


def show_cross_anchor_Sankey(
        adata_omics1: ad.AnnData,
        adata_omics2: ad.AnnData,
        registering_parameters: List,
        celltype_label: Optional[str] = None,
        # omics1_name: str = 'omics1',
        # omics2_name: str = 'omics2',
        fontsize: int = 12,
        color_map: Optional[Union[List, Dict]] = None,
        aspect: float = 10,
        leftLabels: Optional[List] = None,
        rightLabels: Optional[List] = None,
        rightColor: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        title: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        show: bool = True,
        save: Optional[Union[str, os.PathLike]] = None,
):
    """
    Generate a Sankey diagram to visualize the relationship between two omics datasets based on cell type labels.

    Parameters:
    -----------
    adata_omics1 : anndata.AnnData
        AnnData object containing the first omics dataset.
    adata_omics2 : anndata.AnnData
        AnnData object containing the second omics dataset.
    registering_parameters : Dict
        Dictionary containing indices or keys to access anchor points for both omics datasets.
    celltype_label : Optional[str], default None
        Column name in adata.obs containing cell type labels.
    fontsize : int, default 14
        Font size for labels in the Sankey diagram.
    color_map : Optional[Union[List, Dict]], default None
        Color mapping for the Sankey diagram. If a list, maps colors to labels; if a dict, maps labels to colors.
        If None, a default color palette is generated.
    aspect : float, default 10
        Aspect ratio of the Sankey diagram (vertical extent / horizontal extent).
    leftLabels : Optional[List], default None
        Ordered list of labels for the left side of the Sankey diagram. If None, unique labels are used.
    rightLabels : Optional[List], default None
        Ordered list of labels for the right side of the Sankey diagram. If None, unique labels are used.
    rightColor : bool, default False
        If True, color the strips based on right-side labels; otherwise, use left-side labels.
    figsize : Optional[Tuple[float, float]], default None
        Figure size (width, height) for the plot. If None, defaults to (7, 7).
    dpi : Optional[float], default None
        Dots per inch for the figure. If None, uses matplotlib's default.
    title : Optional[str], default None
        Title for the plot. If None, no title is displayed.
    ax : Optional[matplotlib.axes.Axes], default None
        Matplotlib axes object to draw the plot on. If None, a new figure and axes are created.
    show : bool, default True
        If True, display the plot. If False, return the axes object.
    save : Optional[Union[str, os.PathLike]], default None
        File path to save the plot. If None, the plot is not saved.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to the sankey function.

    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the Sankey diagram, returned only if show=False.
    """
    anchor_omics1 = adata_omics1.obs[celltype_label][registering_parameters[1]].tolist()
    anchor_omics2 = adata_omics2.obs[celltype_label][registering_parameters[2]].tolist()
    color_list = list(set(anchor_omics1) | set(anchor_omics2))

    if color_map is None:
        color_map = _get_color(color_list)['palette']

    if isinstance(color_map, list):
        color_map = dict(zip(color_list, color_map[0: len(color_list)]))

    df = {
        'omics1': anchor_omics1,
        'omics2': anchor_omics2,
    }

    if ax is None:
        if figsize is None:
            figsize = (7, 7)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    ax = sankey(
        df['omics1'], df['omics2'], ax, aspect=aspect, colorDict=color_map,
        leftLabels=leftLabels, rightLabels=rightLabels, rightColor=rightColor,
        fontsize=fontsize,
    )

    if title is not None:
        plt.title(title)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax


def show_integrate_umap(
        adata_omics1: ad.AnnData,
        adata_omics2: ad.AnnData,
        model: Literal['omics', 'celltype'] = None,
        celltype_label: Optional[str] = None,
        omics1_name: str = 'omics1',
        omics2_name: str = 'omics2',
        size: Optional[float] = None,
        emb_label: str = 'emb_spcoral',
        palette: Optional[Union[str, List[str]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[float] = None,
        title: Optional[str] = None,
        ax: Optional[matplotlib.axes.Axes] = None,
        show: bool = True,
        save: Optional[Union[str, os.PathLike]] = None,
        **kwargs: Dict[str, Any]
):
    num_omics1 = adata_omics1.shape[0]
    num_omics2 = adata_omics2.shape[0]

    data = np.concatenate((adata_omics1.obsm[emb_label], adata_omics2.obsm[emb_label]), axis=0)
    adata_con = sc.AnnData(data)
    adata_con.obs['omics'] = [omics1_name for i in range(num_omics1)] + [omics2_name for i in range(num_omics2)]

    if model == 'omics':
        color = 'omics'
    elif model == 'celltype':
        color = celltype_label
        if celltype_label is None:
            raise ValueError('celltype_label must be specified when model is \'celltype\'')
    else:
        raise ValueError('model must be one of \'omics\', \'celltype\'')

    sc.tl.umap(adata_con)

    if model == 'omics' and palette is None:
        palette = color_list_2
    if model == 'celltype' and palette is None:
        palette = _get_color(adata_con.obs[celltype_label].tolist)

    if ax is None:
        if figsize is None:
            figsize = (7, 7)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    ax = sc.pl.umap(
        adata_con,
        color=color,
        # Setting a smaller point size to get prevent overlap
        size=size,
        ax=ax,
        title=title,
        palette=palette,
        show=False,
        **kwargs
    )

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
    else:
        return ax

