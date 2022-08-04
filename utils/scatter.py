from typing import Optional, Union, List

import numpy as np
from plotly.graph_objs import Scatter3d, Figure


def pcs_to_plotly(pcs: Union[np.ndarray, List[np.ndarray]],
                  colormaps: Optional[Union[str, List[str], List[float], List[List[float]]]] = None,
                  colors: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                  names: Optional[Union[str, List[str]]] = None):
    """
    Takes a list of point clouds represented as numpy arrays and return a plotly figure
    representing them.
    :param pcs: list of point clouds to render
    :param colormaps: it can be a list describing the rgb values or the name of a colorscale supported by plotly
    :param colors: list of colors associated to each point.
                    If None the colors are associated with the 3d points position.
    :return: A plotly figure containing the input point clouds.
    """
    if isinstance(pcs, np.ndarray):
        pcs = [pcs]

    if not isinstance(colormaps, list) or isinstance(colormaps[0], int):
        colormaps = [colormaps] * len(pcs)

    if not isinstance(colors, list):
        colors = [colors] * len(pcs)

    if not isinstance(names, list):
        names = [names] * len(pcs)

    scatters = []
    for pc, c_map, color, name in zip(pcs, colormaps, colors, names):
        x, y, z = pc.T

        if color is None:
            color = x + y + z
        if c_map is None:
            c_map = 'Viridis'
        if isinstance(c_map, list):
            min_map = f'rgb({",".join([str(150) if x == 0 else str(x) for x in c_map])})'
            max_map = f'rgb({",".join([str(n) for n in c_map])})'
            c_map = [[0, min_map], [1, max_map]]

        scatters.append(Scatter3d({'x': x, 'y': y, 'z': z},
                                  mode='markers',
                                  name=name,
                                  marker=dict(
                                      size=3,
                                      color=color,
                                      colorscale=c_map,
                                      opacity=0.8
                                  )
                                  )
                        )

    fig = Figure(data=scatters, layout={'template': "plotly_dark", 'scene': dict(
                    xaxis=dict(nticks=10, range=[-0.5, 0.5], ),
                    yaxis=dict(nticks=10, range=[-0.5, 0.5], ),
                    zaxis=dict(nticks=10, range=[-0.5, 0.5], ),
                    aspectmode='cube')})

    return fig