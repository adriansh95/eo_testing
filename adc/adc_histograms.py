import numpy as np
import matplotlib.pyplot as plt
import lsst.afw.display as afwDisplay
import os.path
#import tracemalloc
#import linecache
import gc

from collections import defaultdict
from lsst.daf.butler import Butler

# helper function for memory issue
#def display_top(snapshot, key_type='lineno', limit=10):
#    snapshot = snapshot.filter_traces((
#        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"), 
#        tracemalloc.Filter(False, "<unknown>"),
#    ))
#    top_stats = snapshot.statistics(key_type)
#
#    print("Top %s lines" % limit)
#    for index, stat in enumerate(top_stats[:limit], 1):
#        frame = stat.traceback[0]
#        print("#%s: %s:%s: %.1f KiB"
#              % (index, frame.filename, frame.lineno, stat.size / 1024))
#        line = linecache.getline(frame.filename, frame.lineno).strip()
#        if line:
#            print('    %s' % line)
#
#        other = top_stats[limit:]
#        if other:
#            size = sum(stat.size for stat in other)
#            print("%s other: %.1f KiB" % (len(other), size / 1024))
#        total = sum(stat.size for stat in top_stats)
#        print("Total allocated size: %.1f KiB" % (total / 1024))

mod = 32
maxp = 8
powers = np.arange(maxp)
bins = np.arange(mod + 1)
saveto = 'analysis/plots/adc/'
imgtype = 'flat'
mode = 'pairs'

if mode == 'runs':
    subdir = 'runs'
    ids = [3020111800043, 3020112000115]
    expTimes = ['369.208', '369.208']  
    runs = ["'12761'", "'12781'"]
    colors = ['tab:orange', 'tab:blue']
    markers = ['o', 'v']
elif mode == 'pairs':
    subdir = 'pairs'
    ids = [3020112000117, 3020112000118]  
    expTimes = ['539.912', '539.912']  
    colors = ['tab:orange', 'tab:blue']
    markers = ['o', 'v']
    runs = ["'12781'"]
elif mode =='times':
    subdir = 'times'
    ids = [3020112000006, 3020112000097, 3020112000118]
    expTimes = ['13.159', '188.296', '539.912']
    runs = ["'12781'"]
    colors = ['tab:orange', 'tab:blue', 'tab:green']
    markers = ['o', 'v', 'x']

runstr = ', '.join(runs)

where = f"""
instrument = 'LSSTCam'
and exposure.observation_type='{imgtype}'
and exposure.science_program in ({runstr})
"""

if mode =='times':
    where += f'and exposure.id in ({ids[0]}, {ids[1]}, {ids[2]})'
else:
    where += f'and exposure.id in ({ids[0]}, {ids[1]})'


repo = '/sdf/group/lsst/camera/IandT/repo_gen3/bot/butler.yaml'
collections = 'LSSTCam/raw/all'
b = Butler(repo, collections=collections)
reg = b.registry
dtype = 'raw'

# I've had to break the list of detectors up into two because of memory leak
detectors = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
             53, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
             84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
             97, 98, 117, 118, 119, 120, 121, 122, 123, 124, 125,
             126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 
             136, 137, 138, 139, 140, 141, 142, 143]

#detectors = [97, 98, 117, 118, 119, 120, 121, 122, 123, 124, 125,
#             126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 
#             136, 137, 138, 139, 140, 141, 142, 143]
 
# loop over detectors, make two figures for each detector
for detector in detectors:
    #fig0 for histograms, fig1 for bit probabilities
    fig0, axs0 = plt.subplots(8, 2, figsize=(28, 20))
    fig1, axs1 = plt.subplots(8, 2, figsize=(28, 20), sharex=True)

    # do default formatting of ax1 ylims, this will be overwritten if
    # a bit probility deviates from 0.5 by more than 0.05
    for ax1 in axs1.ravel():
        ax1.set_ylim(0.45, 0.55)

    # two dictionaries of dictionaries of arrays to hold bit frequency info, keyed by exposure id, amp
    freqs = defaultdict(lambda: defaultdict(lambda: np.zeros(maxp)))
    freq_errs = defaultdict(lambda: defaultdict(lambda: np.zeros(maxp)))

    # dictionary to hold amp medians, keyed by id, amp
    meds = defaultdict(lambda: defaultdict(lambda: 0))

    # max deviation from 0.5, used to override ylims if necessary, one per amp
    max_diff = defaultdict(lambda: 0)

    # loop over relevant exposures
    #for iid, expTime, color, marker, run in zip(ids, expTimes, colors, markers, runs): ## use this if comparing runs
    for iid, expTime, color, marker in zip(ids, expTimes, colors, markers): ## use this otherwise
        dataId = {'exposure': iid, 'detector': detector}
        datarefs = list(reg.queryDatasets(datasetType=dtype, collections=collections, where=where, dataId=dataId))
        exp = b.get(datarefs[0]) # datarefs should only be one element long
        det = exp.getDetector()
        detName = det.getName()
        detType = det.getPhysicalType()

        # loop over amps once to make histograms and collect bit freq info
        for ampi, ax0 in zip(range(0, 16), axs0.ravel()):
            amp = det[ampi]
            ampName = amp.getName()

            # getting the image array
            trimmed_im = exp.getMaskedImage()[amp.getRawDataBBox()].getImage()
            im_arr = trimmed_im.getArray().flatten().astype(int)

            # computing and storing the median of the amp
            med = np.median(im_arr)
            meds[iid][ampName] = med
            npix = len(im_arr)

            # compute and store the bit freq info
            for power in powers:
                bit = 2**power
                bit_on = (im_arr & bit) // bit
                n_on = np.sum(bit_on)
                freqs[iid][ampName][power] = n_on / npix
                freq_errs[iid][ampName][power] = np.sqrt(n_on) / npix # not sure this is the right error

                del bit_on # deleting anything that might consume a lot of memory

            # Making the histograms
            im_mod = im_arr % mod
            #ax0.hist(im_mod, bins=bins, label=f'{run}, {expTime} s, med={med}', fill=False, edgecolor=color, linewidth=3) ## comparing runs
            ax0.hist(im_mod, bins=bins, label=f'{iid}, {expTime} s, med={med}', fill=False, edgecolor=color, linewidth=3) ## pairs/times

            ax0.set_title(f'{ampName}', fontsize=14)

            # storing the max deviation from 0.5 to override ylims in bit freq plots
            # keeps the largest deviation for each amp accross all exposures
            max_freq_diff = np.max(np.abs(freqs[iid][ampName] - 0.5))
            if max_freq_diff > max_diff[ampName]:
                max_diff[ampName] = max_freq_diff

            del amp, trimmed_im, im_arr, im_mod
        del exp, det

    # override ylims if necessary
    for (amp, diff), ax1 in zip(max_diff.items(), axs1.ravel()):
        if diff > 0.05:
            ax1.set_ylim(0.5 - (diff + 0.02), 0.5 + (diff + 0.02))
 
    # loop over freq dict info to plot bit prob
    for (iid, freq_dict), expTime, marker, color in zip(freqs.items(), expTimes, markers, colors):
        # get the errors and medians dicts
        err_dict = freq_errs[iid]
        med_dict = meds[iid]

        # loop over amps again to plot bit freq info did this 
        # after looping over amps the first time so I can override ylims
        for (amp, freq), ax1 in zip(freq_dict.items(), axs1.ravel()):
            #get errs and medians
            freq_err = err_dict[amp]
            med = med_dict[amp]
            # plot
            ax1.errorbar(powers, freq, yerr=freq_err, label=f'{iid}, {expTime} s, med={med}',
                         linestyle='None', c=color, marker=marker) 
            ax1.set_xticks(powers)
            ax1.grid(b=True)
            ax1.set_title(f'{amp}', fontsize=14)

    # set labels
    for ax1 in axs1[:, 0]:
        ax1.set_ylabel('Bit Probability', fontsize=18)
    for ax1, ax0 in zip(axs1[-1], axs0[-1]):
        ax0.set_xlabel('Signal mod 32', fontsize=18)
        ax1.set_xlabel('Bit power', fontsize=18)

    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()

    # title and legend
    fig0name = f'{runstr} {detName} Signal mod {mod} ({detType})' 
    fig1name = f'{runstr} {detName} Bit Probabilities ({detType})'
    fig0.legend(handles0, labels0, loc='upper right', fontsize=18)
    fig1.legend(handles1, labels1, loc='upper right', fontsize=18)
    fig0.suptitle(fig0name, fontsize=24)
    fig1.suptitle(fig1name, fontsize=24)

    #fig.tight_layout()
    # save the figures
    basename0 = f'{detName}_signal_{mode}.png' 
    basename1 = f'{detName}_bit_probability_{mode}.png' 
    fname0 = os.path.join(saveto, subdir, basename0)
    fname1 = os.path.join(saveto, subdir, basename1)
    fig0.savefig(fname0)
    fig1.savefig(fname1)
    print(f'Wrote {fname0}')
    print(f'Wrote {fname1}')
    # close the figures
    plt.close('all')
    gc.collect()
