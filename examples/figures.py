from nsimplex import Polygon
from nsimplex.plot import plot_polygons, plot_polygons_list
from typing import List, Tuple
from numpy import array
import matplotlib.pyplot as pp



# constant pulled from https://webapps.frm2.tum.de/neutroncalc/
V2_TO_MEV = array([[1, 0], [0, 5.22703717 / 1000 / 1000]])


def to_meV(ts: List[Polygon]):
    return [x.square().transform(V2_TO_MEV) for x in [t.invert() for t in ts]]


def figure_source_sample(source: List[Polygon], sample: List[Polygon], ylabel=None):
    fig, axs = pp.subplots(1, 2, sharex='col', sharey='all')

    h_source = plot_polygons(axs[0], source, vertices={'marker': None, 'linestyle': '-', 'edgecolor': 'gray'})
    h_sample = plot_polygons(axs[1], sample, vertices={'marker': None, 'linestyle': '-', 'edgecolor': 'gray'})
    pp.setp(axs[0], xlabel='Emission time / s', ylabel=ylabel)
    pp.setp(axs[1], xlabel='Sample arrival time / s')

    return fig, axs, (h_source, h_sample)


def figure_source_sample_detector(source: List[Polygon], sample: List[Polygon],
                                  detector: List[List[Polygon]],
                                  plottype=None, repeat=None):
    fig, axs = pp.subplots(1, 3, sharex='col', sharey='all')
    li, lf = '', ''
    if plottype is not None:
        if 'energy' in plottype.lower():
            li = r'$E_i$ / meV'
            lf = f'$E_f$ / meV'
        elif 'velocity' in plottype.lower():
            li = r'$v_i$ / m s$^{-1}$'
            li = r'$v_f$ / m s$^{-1}$'
    vertices = {'marker': None, 'linestyle': '-', 'edgecolor': 'gray'}

    ax0, ax1, ax2 = axs[0], axs[1], axs[2]
    if repeat is None:
        h_source = plot_polygons(ax0, source, vertices=vertices)
        h_sample = plot_polygons(ax1, sample, vertices=vertices)
        h_detector = plot_polygons_list(ax2, detector, vertices=vertices)
    else:
        h_source = plot_polygons_list(ax0, [source, [s + (repeat, 0) for s in source]], vertices=vertices)
        h_sample = plot_polygons_list(ax1, [sample, [s + (repeat, 0) for s in sample]], vertices=vertices)
        h_detector = plot_polygons_list(ax2, detector + [[s + (repeat, 0) for s in d] for d in detector], vertices=vertices)

    pp.setp(ax0, xlabel='Emission time / s', ylabel=li)
    pp.setp(ax1, xlabel='Sample arrival time / s')
    pp.setp(ax2, xlabel='Detector arrival time /s', ylabel=lf)

    return fig, axs, (h_source, h_sample, h_detector)


def figure_source_sample_after_detector(source: List[Polygon], sample: List[Polygon],
                                  after_sample: List[Polygon],
                                  detector: List[List[Polygon]],
                                  plottype=None, repeat=None):
    fig, axs = pp.subplots(1, 3, sharex='col', sharey='all')
    li, lf = '', ''
    if plottype is not None:
        if 'energy' in plottype.lower():
            li = r'$E_i$ / meV'
            lf = f'$E_f$ / meV'
        elif 'velocity' in plottype.lower():
            li = r'$v_i$ / m s$^{-1}$'
            li = r'$v_f$ / m s$^{-1}$'
    vertices = {'marker': None, 'linestyle': '-', 'edgecolor': 'gray'}

    ax0, ax1, ax2 = axs[0], axs[1], axs[2]
    if repeat is None:
        h_source = plot_polygons(ax0, source, vertices=vertices)
        h_sample = plot_polygons(ax1, sample, vertices=vertices)
        plot_polygons(ax1, after_sample, vertices=vertices)
        h_detector = plot_polygons_list(ax2, detector, vertices=vertices)
    else:
        h_source = plot_polygons_list(ax0, [source, [s + (repeat, 0) for s in source]], vertices=vertices)
        h_sample = plot_polygons_list(ax1, [sample, [s + (repeat, 0) for s in sample]], vertices=vertices)
        plot_polygons_list(ax1, [after_sample, [s + (repeat, 0) for s in after_sample]], vertices=vertices)
        h_detector = plot_polygons_list(ax2, detector + [[s + (repeat, 0) for s in d] for d in detector], vertices=vertices)

    pp.setp(ax0, xlabel='Emission time / s', ylabel=li)
    pp.setp(ax1, xlabel='Sample arrival time / s')
    pp.setp(ax2, xlabel='Detector arrival time /s', ylabel=lf)

    return fig, axs, (h_source, h_sample, h_detector)