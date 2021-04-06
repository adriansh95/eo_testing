import numpy as np
from scipy.stats import binned_statistic

class profile:
    def __init__(self, xarr, yarr, xerr, yerr):
        self.xarr = xarr
        self.yarr = yarr
        self.xerr = xerr
        self.yerr = yerr
        
    def __sub__(self, other):
        sx = self.xarr
        ox = other.xarr
        iself = np.in1d(sx, ox)
        iother = np.in1d(ox, sx)

        sub_xarr = sx[iself]
        sub_yarr = self.yarr[iself] - other.yarr[iother]
        sub_xerr = self.xerr[iself]
        sub_yerr = self.yerr[iself] + other.yerr[iother]

        sub_prof = profile(sub_xarr, sub_yarr, sub_xerr, sub_yerr)

        return sub_prof
    
    def unpack(self):
        return self.xarr, self.yarr, self.xerr, self.yerr

def make_profile(x, y, xlims=(0, 200001), bin_size=500):
    xmin = xlims[0] 
    xmax = xlims[1] 
    bins = np.arange(xmin, xmax, bin_size)

    means, bin_edges, binnumber = binned_statistic(x, y, statistic='mean', bins=bins)
    stds, bin_edges, binnumber = binned_statistic(x, y, statistic='std', bins=bins)
    counts, bin_edges, binnumber = binned_statistic(x, y, statistic='count', bins=bins)

    valid = np.array(counts>20)
    bin_mids = bin_edges[1:] - (bin_size / 2)
    yarr = means[valid]
    xarr = bin_mids[valid]
    yerr = stds[valid]/np.sqrt(counts[valid])
    xerr = np.full(len(xarr), 0.5*bin_size)

    prof = profile(xarr, yarr, xerr, yerr)
    return prof 

def xtalk_profile(x, r, xlims=(0, 200001), bin_size=500):
    xmin = xlims[0] 
    xmax = xlims[1] 
    bins = np.arange(xmin, xmax, bin_size)
    y = x * r

    y_means, bin_edges, binnumber = binned_statistic(x, y, statistic='mean', bins=bins)
    r_stds, bin_edges, binnumber = binned_statistic(x, r, statistic='std', bins=bins)
    x_means, bin_edges, binnumber = binned_statistic(x, x, statistic='mean', bins=bins)
    counts, bin_edges, binnumber = binned_statistic(x, y, statistic='count', bins=bins)

    r_means = y_means/x_means
    valid = np.array(counts>20)
    bin_mids = bin_edges[1:] - (bin_size / 2)
    yarr = r_means[valid]
    xarr = bin_mids[valid]
    yerr = r_stds[valid]/np.sqrt(counts[valid])
    xerr = np.full(len(xarr), 0.5*bin_size)

    prof = profile(xarr, yarr, xerr, yerr)
    return prof 
