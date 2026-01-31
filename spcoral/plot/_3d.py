### The codes in this file is copy from https://github.com/gao-lab/SLAT/tree/main
### References: https://doi.org/10.1038/s41467-023-43105-5

import random
from typing import List, Optional, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
from anndata import AnnData
import matplotlib

from .color import _get_color

class match_3D_multi:
    r"""
    Plot the mapping result between 2 datasets

    Parameters
    ----------
    dataset_A
        pandas dataframe which contain ['index','x','y'], reference dataset
    dataset_B
        pandas dataframe which contain ['index','x','y'], target dataset
    matching
        matching results
    meta
        dataframe colname of meta, such as celltype
    expr
        dataframe colname of gene expr
    subsample_size
        subsample size of matches
    reliability
        match score (cosine similarity score)
    scale_coordinate
        if scale coordinate via (:math:`data - np.min(data)) / (np.max(data) - np.min(data))`)
    rotate
        how to rotate the slides (force scale_coordinate), such as ['x','y'], means dataset0 rotate on x axes
        and dataset1 rotate on y axes
    change_xy
        exchange x and y on dataset_B
    subset
        index of query cells to be plotted

    Note
    ----------
    dataset_A and dataset_B can in different length

    """

    def __init__(
        self,
        dataset_A: pd.DataFrame,
        dataset_B: pd.DataFrame,
        matching: np.ndarray,
        meta: Optional[str] = None,
        expr: Optional[str] = None,
        subsample_size: Optional[int] = 300,
        reliability: Optional[np.ndarray] = None,
        scale_coordinate: Optional[bool] = True,
        rotate: Optional[List[str]] = None,
        exchange_xy: Optional[bool] = False,
        subset: Optional[List[int]] = None,
    ) -> None:
        self.dataset_A = dataset_A.copy()
        self.dataset_B = dataset_B.copy()
        self.meta = meta
        self.matching = matching
        self.conf = reliability
        self.subset = subset  # index of query cells to be plotted
        scale_coordinate = True if rotate is not None else scale_coordinate

        assert all(item in dataset_A.columns.values for item in ["index", "x", "y"])
        assert all(item in dataset_B.columns.values for item in ["index", "x", "y"])

        if meta is not None:
            set1 = list(set(self.dataset_A[meta].tolist()))
            set2 = list(set(self.dataset_B[meta].tolist()))
            self.celltypes = set1 + [x for x in set2 if x not in set1]
            self.celltypes.sort()  # make sure celltypes are in the same order
            overlap = set(set2).intersection(set1)
            print(
                f"dataset1: {len(set1)} cell types; dataset2: {len(set2)} cell types; \n\
                    Total :{len(self.celltypes)} celltypes; Overlap: {len(overlap)} cell types \n\
                    Not overlap :[{[y for y in (set1+set2) if y not in overlap]}]"
            )
        self.expr = expr if expr else False

        if scale_coordinate:
            for i, dataset in enumerate([self.dataset_A, self.dataset_B]):
                for axis in ["x", "y"]:
                    dataset[axis] = (dataset[axis] - np.min(dataset[axis])) / (
                        np.max(dataset[axis]) - np.min(dataset[axis])
                    )
                    if rotate is None:
                        pass
                    elif axis in rotate[i]:
                        dataset[axis] = 1 - dataset[axis]
        if exchange_xy:
            self.dataset_B[["x", "y"]] = self.dataset_B[["y", "x"]]

        if subset is not None:
            matching = matching[:, subset]
        if matching.shape[1] > subsample_size and subsample_size > 0:
            self.matching = matching[
                :, np.random.choice(matching.shape[1], subsample_size, replace=False)
            ]
        else:
            subsample_size = matching.shape[1]
            self.matching = matching
        print(f"Subsampled {subsample_size} pairs from {matching.shape[1]}")

        self.datasets = [self.dataset_A, self.dataset_B]

    def draw_3D(
            self,
            conf_cutoff: Optional[float] = 0,
            point_size: Optional[List[float]] = [0.1, 0.1],
            line_width: Optional[float] = 0.3,
            line_color: Optional[str] = "grey",
            line_alpha: Optional[float] = 0.7,
            hide_axis: Optional[bool] = False,
            show_error: Optional[bool] = True,
            show_celltype: Optional[bool] = False,
            color: Union[List, Dict, None] = None,
            cmap: Optional[str] = "Reds",
            ax: Optional[plt.Axes] = None,
    ) -> Optional[matplotlib.axes.Axes]:
        r"""
        Draw 3D picture of two datasets

        Parameters
        ----------
        size
            plt figure size
        conf_cutoff
            confidence cutoff of mapping to be plotted
        point_size
            point size of every dataset
        line_width
            pair line width
        line_color
            pair line color
        line_alpha
            pair line alpha
        hide_axis
            if hide axis
        show_error
            if show error celltype mapping with different color
        cmap
            color map when vis expr
        save
            save file path
        """
        self.conf_cutoff = conf_cutoff
        show_error = show_error if self.meta else False
        # color by meta
        if self.meta:
            if color is None:
                print(_get_color(self.celltypes))
                color = _get_color(self.celltypes)['palette']
            if isinstance(color, list):
                c_map = {}
                for i, celltype in enumerate(self.celltypes):
                    c_map[celltype] = color[i]
            else:
                c_map = color
            if self.expr:
                c_map = cmap
                # expr_concat = pd.concat(self.datasets)[self.expr].to_numpy()
                # norm = plt.Normalize(expr_concat.min(), expr_concat.max())
            for i, dataset in enumerate(self.datasets):
                if self.expr:
                    norm = plt.Normalize(
                        dataset[self.expr].to_numpy().min(),
                        dataset[self.expr].to_numpy().max(),
                    )
                for cell_type in self.celltypes:
                    slice = dataset[dataset[self.meta] == cell_type]
                    xs = slice["x"]
                    ys = slice["y"]
                    zs = i
                    if self.expr:
                        ax.scatter(
                            xs,
                            ys,
                            zs,
                            s=point_size[i],
                            c=slice[self.expr],
                            cmap=c_map,
                            norm=norm,
                        )
                    else:
                        ax.scatter(xs, ys, zs, s=point_size[i], c=c_map[cell_type])
        # plot points without meta
        else:
            for i, dataset in enumerate(self.datasets):
                xs = dataset["x"]
                ys = dataset["y"]
                zs = i
                ax.scatter(xs, ys, zs, s=point_size[i])
        # plot line
        self.c_map = c_map
        self.draw_lines(ax, show_error, show_celltype, line_color, line_width, line_alpha)
        if hide_axis:
            plt.axis("off")
        return ax


    def draw_lines(
        self, ax, show_error, show_celltype, line_color, line_width=0.3, line_alpha=0.7
    ) -> None:
        r"""
        Draw lines between paired cells in two datasets
        """
        for i in range(self.matching.shape[1]):
            if self.conf is not None and self.conf[i] < self.conf_cutoff:
                continue
            pair = self.matching[:, i]
            default_color = line_color
            if self.meta is not None:
                celltype1 = (
                    self.dataset_A.loc[self.dataset_A["index"] == pair[0], self.meta]
                    .astype(str)
                    .values[0]
                )
                celltype2 = (
                    self.dataset_B.loc[self.dataset_B["index"] == pair[1], self.meta]
                    .astype(str)
                    .values[0]
                )
                if show_error:
                    if celltype1 == celltype2:
                        color = "#ade8f4"  # blue
                    else:
                        color = "#ffafcc"  # red
                if show_celltype:
                    if celltype1 == celltype2:
                        color = self.c_map[celltype1]
                    else:
                        color = "#696969"  # celltype1 error match color
            point0 = np.append(self.dataset_A[self.dataset_A["index"] == pair[0]][["x", "y"]], 0)
            point1 = np.append(self.dataset_B[self.dataset_B["index"] == pair[1]][["x", "y"]], 1)

            coord = np.row_stack((point0, point1))
            color = color if show_error or show_celltype else default_color
            ax.plot(
                coord[:, 0],
                coord[:, 1],
                coord[:, 2],
                color=color,
                linestyle="dashed",
                linewidth=line_width,
                alpha=line_alpha,
            )
