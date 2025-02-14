import os
import pandas as pd
import numpy as np
import math
import seaborn as sns
from graphing_functions import *
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

def iqr_removal(dataset):
    new = dataset.copy()
    for col in dataset.columns:
        if col == "Source" or col == "RoiType" or col == 'Group' or col == "Date":
            continue
        coldata = dataset[col]
        mean, std, median = np.mean(coldata), np.std(coldata), np.median(coldata)
        plow, phigh = np.percentile(coldata, (25, 75))
        iqr = phigh - plow
        new[col].where(new[col].between(median - 1.5 * iqr, median + 1.5 * iqr), inplace=True)
    return new

def format_p(pval):
    if pval > 0.05:
        return 'N.S.'
    elif pval <= 0.05 and pval > 0.01:
        return '*'
    else:
        return math.floor(-math.log(pval, 10)) * '*'
    
def process_title(string, keyword):
    if string == "Area":
        return "Mean Area of " + keyword + " " + r'${({\mu}m^2)}$'#r'$\bf{({\mu}m^2)}$'
    elif string == "Circ.":
        return "Circularity"
    elif string == "Mean" or string == "Min" or string == "Max":
        return f"{string} Grey Value"
    else:
        return string
    
def plot_sem(ax, xval, width, dataset, metric, group):
    this_data = dataset[dataset['Group'] == group][metric]
    meanval = this_data.mean()
    sem = stats.sem(this_data, axis = 0, ddof = 0)
    subwidth = width * 0.5
    ax.plot([xval - width, xval + width], [meanval, meanval], c = 'k', linewidth = 0.6, zorder = 3)
    ax.plot([xval, xval], [meanval - sem, meanval + sem], c = 'k', linewidth = 0.35, zorder = 3)
    ax.plot([xval - subwidth, xval + subwidth], [meanval + sem, meanval + sem], c = 'k', linewidth = 0.35, zorder = 3)
    ax.plot([xval - subwidth, xval + subwidth], [meanval - sem, meanval - sem], c = 'k', linewidth = 0.35, zorder = 3)