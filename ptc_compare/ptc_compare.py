import astropy.io.fits as pyfits
import numpy as np
import pandas as pd
import os
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lsst.daf.persistence import Butler

default_csv_loc = 'no_oscan/analysis/csvs'
default_pkl_loc = 'analysis/test/no_oscan/calibrations/ptc'
default_butler_repo = '/lsstdata/offline/teststand/BOT/gen2repo'  
default_writeto = 'analysis/ptc_comparison/no_oscan'
amp_names = ['C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 
                'C07', 'C06', 'C05', 'C04', 'C03', 'C02', 'C01', 'C00']
labels = ['mean', 'var'] 

def get_det_num(butler, dataId):
    detectors = butler.queryMetadata('raw', 'detector', dataId=dataId)
    return detectors[0]

def get_eotest_mean_var(eotest_file):
    amp_means = {}
    amp_vars = {}
    hdulist = pyfits.open(eotest_file)
    for iamp, amp in enumerate(amp_names, start=1):
        name = f'AMP{iamp:02d}'
        mean = hdulist[1].data[name+'_mean']
        var = hdulist[1].data[name+'_var']
        amp_means[amp] = mean
        amp_vars[amp] = var

    return amp_means, amp_vars

def get_dm_mean_var(dm_file):
    amp_means = {}
    amp_vars = {}
    with open(dm_file, 'rb') as f:
        ptc_dataset = pickle.load(f)
        for amp_name, amp_mean in ptc_dataset.rawMeans.items():
            amp_means[amp_name] = amp_mean
            amp_vars[amp_name] = ptc_dataset.rawVars[amp_name]

    return amp_means, amp_vars

def relative_diff(eotest_vals, dm_vals):
    rel_diffs = {}
    for amp, eotest_val in eotest_vals.items():
        rel_diffs[amp] = (dm_vals[amp] - eotest_val) / eotest_val

    return rel_diffs

def plot_eotest_dm_vals(eotest_file, dm_file, writeto):
    eotest_means, eotest_vars = get_eotest_mean_var(eotest_file)
    dm_means, dm_vars = get_dm_mean_var(dm_file)
    means = [eotest_means, dm_means]
    variances = [eotest_vars, dm_vars]
    x = np.linspace(0, 2e5, 1001)

    for label, dict_list in zip(labels, [means, variances]):
        fig, axs = plt.subplots(4, 4, sharex=True, figsize=(27, 18))
        axs = axs.ravel()
        eotest_dict = dict_list[0]
        dm_dict = dict_list[1]
        
        for ax, (amp, eotest_vals) in zip(axs, eotest_dict.items()):
            size = 1.0
            pad = 0.10
            divider = make_axes_locatable(ax)
            ax2 = divider.append_axes("bottom", size=size, pad=pad, sharex=ax)

            slope, intercept, r, p, std_err = stats.linregress(eotest_vals, dm_dict[amp])
            y = slope*x + intercept
            ax.plot(x, y, 'r', label=f'r^2 = {r**2:.3f}')
            ax.legend(loc='lower right')
            ax.plot(eotest_vals, dm_dict[amp], 'x')
            ax.set_title(f'{amp}')
            ax.set_ylabel(f'DM {label}')

            resids = dm_dict[amp] - (slope*eotest_vals + intercept)
            ax2.set_xlabel(f'Eotest {label}')
            ax2.plot(eotest_vals, resids, 'x')
            ax2.axhline(y=0, color='lightgrey', linestyle='--')
            
        fig_name = os.path.join(writeto, f'det000_eotest_dm_{label}.png')
        fig.tight_layout()
        plt.savefig(fig_name)
        plt.close()

def plot_mean_var_diff(eotest_file, dm_file, writeto):
    eotest_means, eotest_vars = get_eotest_mean_var(eotest_file)
    dm_means, dm_vars = get_dm_mean_var(dm_file)
    mean_rel_diffs = relative_diff(eotest_means, dm_means)
    var_rel_diffs = relative_diff(eotest_vars, dm_vars)

    for label, diffs in zip(labels, [mean_rel_diffs, var_rel_diffs]):
        fig, axs = plt.subplots(4, 4, sharex=True, figsize=(27, 18))
        axs = axs.ravel()
        
        for ax, (amp, diff) in zip(axs, diffs.items()):
            ax.plot(eotest_means[amp], diff, 'o')
            ax.set_title(f'{amp} {label}')
            ax.set_ylabel('(DM-eotest)/eotest')
            ax.axhline(y=0, color='lightgrey', linestyle='--')
            ax.set_xlabel('Eotest Mean Signal (ADU)')
            ax.set_xlim([0, 200000])

        fig_name = os.path.join(writeto, f'det000_{label}_relative_difference.png')
        fig.tight_layout()
        plt.savefig(fig_name)
        plt.close()

def main(csv_loc, pkl_loc, butler_repo, writeto):
    slot_names = []
    for i in range(3):
        for j in range(3):
            slot_names.append(f'S{i}{j}')

    csv_files = glob.glob(f'{csv_loc}/*ptc.csv')
    pkl_files = glob.glob(f'{pkl_loc}/ptc*.pkl')
    b = Butler(butler_repo)

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)

        raft = filename.split('_')[0]
        run = filename.split('_')[1]

        fig, axs = plt.subplots(3, 3, sharey=True, figsize=(27, 18))
        axs = axs.ravel()

        for ax, (islot, slot) in zip(axs, enumerate(slot_names)):
            dataId = dict(run=run, raftName=raft, detectorName=slot, imageType='FLAT')
            expIds = b.queryMetadata('raw', 'expId', dataId=dataId)
            dataId['expId'] = expIds[0]

            det_num = get_det_num(b, dataId)

            try:
                pkl_file = glob.glob(f'{pkl_loc}/ptc*det{det_num:03d}.pkl')[0]
            except IndexError:
                print(f'No .pkl file for detector {det_num}. Skipping.')
                continue

            amp_names = []
            dm_gains = np.zeros(16)
            dm_gain_errors = np.zeros(16)

            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

                for idx, (amp_name, gain) in enumerate(data.gain.items()):
                    amp_names.append(amp_name)
                    dm_gains[idx] = gain
                    dm_gain_errors[idx] = data.gainErr[amp_name]

            eou_gains = df.gain[df.slot == islot]
            eou_gain_errors = df.gain_error[df.slot == islot]
            amp_nums = df.amp[df.slot == islot]

            ax.errorbar(amp_nums, eou_gains, yerr=eou_gain_errors, linestyle='None', fmt = 'o', label = 'eo_utils')
            ax.errorbar(amp_nums, dm_gains, yerr=dm_gain_errors, linestyle='None', fmt = 'o', label = 'DM')
            ax.set_title(slot)
            ax.set_ylabel('Gain')
            ax.set_ylim([0.75, 1.2])
            ax.set_xlabel('Amp')
            ax.set_xticks(amp_nums)
            ax.set_xticklabels(amp_names)

            size = 1.0
            pad = 0.10
            diff = dm_gains - eou_gains
            diff_err = dm_gain_errors + eou_gain_errors
            divider = make_axes_locatable(ax)
            ax2 = divider.append_axes("bottom", size=size, pad=pad, sharex=ax)
            ax2.errorbar(amp_nums, diff, yerr=diff_err, linestyle='None', color='r', fmt='o')
            ax2.set_ylim([0.1, 0.4])
 
        fig_name = os.path.join(writeto, f'{raft}_ptc_compare.png')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, 'upper right', fontsize='x-large')
        fig.tight_layout()
        plt.savefig(fig_name)
        plt.close()

#main(default_csv_loc, default_pkl_loc, default_butler_repo, default_writeto)
plot_eotest_dm_vals('analysis/ptc_comparison/simulated_pedestal/eotest_results/no_oscan/det000_ptc.fits', 
                'analysis/ptc_comparison/simulated_pedestal/rerun/no_oscan/calibrations/ptc/ptcDataset-det000.pkl',
                'analysis/ptc_comparison/plots/no_oscan')

plot_mean_var_diff('analysis/ptc_comparison/simulated_pedestal/eotest_results/no_oscan/det000_ptc.fits', 
                'analysis/ptc_comparison/simulated_pedestal/rerun/med_per_row/calibrations/ptc/ptcDataset-det000.pkl',
                'analysis/ptc_comparison/plots/no_oscan')
