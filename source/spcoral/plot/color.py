# define color list in SPCoral
from typing import Sequence, Union
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

# continuous color
colors = ["#780000","#c1121f","#fdf0d5","#003049","#669bbc"]
n_bins = 256
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# 2 color
color_list_2 = ['#ff595e', '#1982c4']

# 5 color
color_list_5 = ["#f9c80e","#f86624","#ea3546","#662e9b","#43bccd"]

# 10 color
color_list_10 = ["#d00000","#ffba08","#cbff8c","#8fe388","#1b998b","#3185fc","#5d2e8c","#46237a","#ff7b9c","#ff9b85"]

# 20 color
color_list_20 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
                 "#17becf", "#9edae5", "#ffbb78", "#d5e8d4", "#fcae91", "#d7c1de", "#cdac93", "#ebceeb", "#a6a6a6",
                 "#000000", "#545454"]

# 50 color
color_list_50 = [
    '#5050FFFF', '#CE3D32FF', '#749B58FF', '#F0E685FF', '#466983FF', '#BA6338FF', '#5DB1DDFF',
    '#802268FF', '#6BD76BFF', '#D595A7FF', '#924822FF', '#837B8DFF', '#C75127FF', '#D58F5CFF',
    '#7A65A5FF', '#E4AF69FF', '#3B1B53FF', '#CDDEB7FF', '#612A79FF', '#AE1F63FF', '#E7C76FFF',
    '#5A655EFF', '#CC9900FF', '#99CC00FF', '#A9A9A9FF', '#CC9900FF', '#99CC00FF', '#00D68FFF',
    '#14FFB1FF', '#00CC99FF', '#0099CCFF', '#0A47FFFF', '#4775FFFF', '#FFC20AFF', '#FFD147FF',
    '#990033FF', '#991A00FF', '#996600FF', '#809900FF', '#339900FF', '#00991AFF', '#009966FF',
    '#008099FF', '#003399FF', '#1A0099FF', '#660099FF', '#990080FF', '#D60047FF', '#FF1463FF',
    '#00D68FFF'
]


def _get_color(input_list):
    if not input_list or not isinstance(input_list, list):
        raise ValueError("Input must be a non-empty list")

    first_element = input_list[0]

    if isinstance(first_element, (int, float)):
        return {'color_type': 'continuous', 'cmap': custom_cmap}

    if isinstance(first_element, str):
        n_categories = len(set(input_list))
        if n_categories <= 2:
            return {'color_type': 'categorical', 'palette': color_list_2}
        elif n_categories <= 5:
            return {'color_type': 'categorical', 'palette': color_list_5}
        elif n_categories <= 10:
            return {'color_type': 'categorical', 'palette': color_list_10}
        elif n_categories <= 20:
            return {'color_type': 'categorical', 'palette': color_list_20}
        elif n_categories <= 50:
            return {'color_type': 'categorical', 'palette': color_list_50}
        else:
            palette = color_list_50 * (n_categories // len(color_list_50) + 1)
            return {'color_type': 'categorical', 'palette': palette[:n_categories]}

    raise ValueError("Input list elements must be int, float, or str")


def _get_palette(categorical, sort_order: bool = True, palette: Union[Sequence, ListedColormap] = None) -> dict:
    are_all_str = all(map(lambda x: isinstance(x, str), categorical))
    if not are_all_str:
        categorical = str(categorical)

    if sort_order:
        categorical = sorted(categorical)

    if palette is None:
        if len(categorical) <= 5:
            palette = color_list_5
        elif len(categorical) <= 10:
            palette = color_list_10
        elif len(categorical) <= 20:
            palette = color_list_20
        else:
            palette = color_list_50

    if isinstance(palette, ListedColormap):
        palette = palette.colors

    palette = palette[0: len(categorical)]
    palette = dict(zip(categorical, palette))
    return palette