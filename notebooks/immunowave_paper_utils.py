#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:51:05 2024

@author: brandon
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


"""functions"""


def style_axes(ax, fontsize=24):
    plt.minorticks_off()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    # plt.tight_layout()

    return ax


"""params"""
colors = {"wave": np.array([32, 187, 178]) / 255, "cell": np.array([86, 37, 108]) / 255}

linewidth = 4
fontsize = 24
rc_params = {"font.family": "Arial", "axes.linewidth": linewidth, "font.size": fontsize}
