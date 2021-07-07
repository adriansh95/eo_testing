import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import matplotlib.ticker as mtick
import pickle as pkl
import glob
import os
import itertools
import math

from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import linregress
from pathlib import Path
from utils.defaults import all_amps, amp_neighbors
from utils.BOT_9raft_detmap import detectors_map
from utils.plotting_utils import plot_fp, map_detector_to_fp,\
    plot_raft, map_detector_to_raft
from utils.profile import profile, make_profile


class xtalkDataset:
    def __init__(self, run, detector):
        self.run = run
        self.detector = detector
        self.summary = defaultdict(lambda: defaultdict(dict))
        self.profiles = defaultdict(lambda: defaultdict(profile)) 

class xtalk_nonlinearityConfig():
    """ Config class for xtalk_nonlinearityTask """
    def __init__(self, runs, xtalk_data_loc, write_to,
                 make_datasets, plot_profiles,
                 plot_focalplane, norm_cutoff=45000,
                 profile_xlims=(30000, 200001), bin_size=500):

        self.runs = runs
        self.xtalk_data_loc = xtalk_data_loc
        self.write_to = write_to
        self.make_datasets = make_datasets
        self.plot_profiles = plot_profiles
        self.plot_focalplane = plot_focalplane
        self.norm_cutoff = norm_cutoff
        self.profile_xlims = profile_xlims
        self.bin_size = bin_size

class xtalk_nonlinearityTask():
    def __init__(self, config):
        self.config = config
        self.amp_pairs = list(itertools.combinations(all_amps, r=2))
        self.bad_pairs = defaultdict(lambda: defaultdict(list))

    def find_rat_flux_data(self, run):
        fname = f'ratios_fluxes_{run}*.pkl'
        globstring = os.path.join(self.config.xtalk_data_loc, fname)
        filepaths = glob.glob(globstring)
        return filepaths

    def find_dataset_files(self, run):
        fname = f'{run}*xtalk_dataset.pkl'
        globstring = os.path.join(self.config.write_to, 'datasets', fname)
        filepaths = glob.glob(globstring)
        return filepaths

    def run(self):
        if self.config.make_datasets:
            self.make_datasets()
        if self.config.plot_profiles:
            self.plot_profiles()
        if self.config.plot_focalplane:
            self.plot_focalplane()
            self.plot_histograms()

    def make_datasets(self):
        for run in self.config.runs:
            pkl_files = self.find_rat_flux_data(run)
            
            for filename in pkl_files:
                stem = os.path.basename(filename).split('.')[0]
                detector = stem.split('_')[3][3:] 

                # Initialize a dataset to hold detector summary information
                dataset = xtalkDataset(run, detector)

                with open(filename, 'rb') as pkl_file:
                    d = pkl.load(pkl_file)
                    ratios = d['ratios'] #keyed by target, source
                    fluxes = d['fluxes'] #keyed by target, source

                    for amp_i, amp_j in self.amp_pairs:
                        
                        if len(fluxes[amp_i][amp_j]) == 0 or len(fluxes[amp_j][amp_i]) == 0:
                            print(f'Bad amp pair: {amp_i}, {amp_j}')
                            bad_pair = True
                            self.bad_pairs[run][f'det{detector}'].append((amp_i, amp_j))
                            continue

                        for amp1, amp2 in [(amp_i, amp_j), (amp_j, amp_i)]:

                            # Get the xtalk data for amp pair
                            amp_fluxes = np.array(fluxes[amp2][amp1])

                            amp_ratios = np.array(ratios[amp2][amp1])
                            secondary_signals = amp_ratios * amp_fluxes

                            # Fit line, subtract intercept, recompute ratios
                            signals_linr = linregress(amp_fluxes, secondary_signals)
                            signals_intercept = signals_linr.intercept
                            amp_ratios_corrected = (secondary_signals - signals_intercept) / amp_fluxes

                            # Calculate the average ratio, std, slope, norm
                            av = np.mean(amp_ratios_corrected)
                            std = np.std(amp_ratios_corrected)
                            norm = self.calculate_norm(amp_ratios_corrected, amp_fluxes)
                            amp_ratios_linr = linregress(amp_fluxes, amp_ratios_corrected)
                            amp_ratios_slope = amp_ratios_linr.slope

                            # Make profile
                            profile = make_profile(amp_fluxes, amp_ratios_corrected,
                                                   xlims=self.config.profile_xlims,
                                                   bin_size=self.config.bin_size)

                            # Compile summary statistics for amp pair in detector level dataset
                            summary = dict(norm=norm, xtalk_coeff=av, xtalk_err=std, nonlinearity=amp_ratios_slope)

                            dataset.summary[amp2][amp1] = summary
                            dataset.profiles[amp2][amp1] = profile

                        summ_ji = dataset.summary[amp_j][amp_i]  
                        summ_ij = dataset.summary[amp_i][amp_j] 
                        diff = np.abs(summ_ji['xtalk_coeff'] - summ_ij['xtalk_coeff'])
                        diff_err = np.abs(summ_ji['xtalk_err'] - summ_ij['xtalk_err'])

                        for amp1, amp2 in [(amp_i, amp_j), (amp_j, amp_i)]:
                            dataset.summary[amp2][amp1]['diff'] = diff
                            dataset.summary[amp2][amp1]['diff_err'] = diff_err

                self.save_dataset(dataset)         

    def calculate_norm(self, ratios, fluxes):
        norm_cutoff = self.config.norm_cutoff
        return np.mean(ratios[fluxes < norm_cutoff])

    def save_dataset(self, dataset):
        run = dataset.run
        detector = dataset.detector
        
        # turn default dicts into regular dicts for pickling
        summary_temp = dict(dataset.summary)
        profiles_temp = dict(dataset.profiles)
        for k, v in summary_temp.items():
            vtemp = dict(v)
            pvtemp = dict(profiles_temp[k])

            summary_temp[k] = vtemp
            profiles_temp[k] = pvtemp

        dataset.summary = summary_temp
        dataset.profiles = profiles_temp

        fname = f'{run}_det{detector}_xtalk_dataset.pkl'
        pathname = os.path.join(self.config.write_to, 'datasets')
        full_name = os.path.join(pathname, fname)
        Path(pathname).mkdir(parents=True, exist_ok=True)

        with open(full_name, 'wb') as f:
            pkl.dump(dataset, f)
        print(f'Wrote {full_name}')

    def plot_focalplane(self):
        amps = all_amps
        runs_str = ''
        for run in self.config.runs:
            runs_str += f'{run}_'

        summary_keys = ['xtalk_coeff', 'nonlinearity', 'diff']

        plot_titles = [f'{runs_str}Xtalk Coefficient', f'{runs_str}Nonlinearity (First Order)',
                       f'{runs_str}Amp Pair Asymmetry']

        figure_names = [f'{runs_str}fp_coeff.png', f'{runs_str}fp_nonlinearity.png',
                        f'{runs_str}fp_asymmetry.png']

        colorbounds = [[-1e-2, -1e-3, -1e-4, -1e-5, -5e-6, 5e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                       np.concatenate((np.arange(-8e-11, 0, 1e-11), np.arange(1e-11, 9e-11, 1e-11))),
                       np.arange(0, 10e-5, 1e-5)]
                       
        cmaps = ["seismic", "seismic", "Reds"]

        fdict = defaultdict(dict)

        for key, colorbound, cmap in zip(summary_keys, colorbounds, cmaps):
            fig, axs = plot_fp()
            fdict[key]['fig'] = fig
            fdict[key]['axs'] = axs
            fdict[key]['bounds'] = colorbound
            fdict[key]['cmap'] = cmap

        for run in self.config.runs:
            dataset_files = self.find_dataset_files(run)

            for dataset_file in dataset_files:
                for key in summary_keys:
                    axs = fdict[key]['axs']
                    bounds = fdict[key]['bounds']
                    cmap = fdict[key]['cmap']
                    vals = np.full((16, 16), np.nan)
                    
                    with open(dataset_file, 'rb') as f:
                        ds = pkl.load(f)
                        summary = ds.summary
                        detector = ds.detector
                    
                        for target, sdict in summary.items():
                            i = amps.index(target)
                            for source, vdict in sdict.items():
                                j = amps.index(source)
                                try:
                                    val = vdict[key]
                                    vals[i, j] = val
                                except KeyError:
                                    continue
            
                    ax_idx = map_detector_to_fp(detector)
                    ax = axs[ax_idx]

                    isNan = np.isnan(vals)
            
                    ax.imshow(isNan, cmap='binary')
                    fdict[key]['im'] = ax.imshow(vals, cmap=cmap,
                                                 norm=colors.BoundaryNorm(boundaries=bounds,
                                                                          ncolors=256)) ## different norm for each plot
            
        for key, title, fname in zip(summary_keys, plot_titles, figure_names):
            fig = fdict[key]['fig']
            bounds = fdict[key]['bounds']
            im = fdict[key]['im']
            fig.subplots_adjust(right=0.9, top=0.9)
            cbar_ax = fig.add_axes([0.93, 0.2, 0.025, 0.6])
            fig.colorbar(im, cax=cbar_ax, format='%.2e', ticks=bounds)
            fig.suptitle(title, fontsize='xx-large')
            pathname = os.path.join(self.config.write_to, 'plots/focal_plane')
            Path(pathname).mkdir(parents=True, exist_ok=True)
            full_name = os.path.join(pathname, fname)
            fig.savefig(full_name)
            print(f'Wrote {full_name}')
            plt.close()

    def plot_histograms(self):
        amps = amp_neighbors

        summary_keys = ['xtalk_coeff', 'nonlinearity', 'diff']

        bin_scales = [10**4, 2e10, 10**4]
        #bin_scales = [10**4, 2e9, 10**4] ##

        xlims = [[-1e-4, 4e-4], [-4e-11, 1.2e-10], [0, 2e-4]]
        #xlims = [[0, 8e-4], [-5e-9, 0], [0, 2e-4]] ##

        plot_titles = ['Xtalk Coefficient', 'Nonlinearity (First Order)',
                       'Amp Pair Asymmetry']

        figure_names = ['xtalk_coeff_hist.png', 'xtalk_nonlinearity_hist.png',
                        'xtalk_asymmetry_hist_uncorrected.png']

        fdict = defaultdict(dict)

        # Loop over runs (rafts) one plot per raft
        for run in self.config.runs:
        
            # Make figures for coeffs, nonlin, asymmetry data
            for key in summary_keys:
                fig, axs = plot_raft()
                fdict[key]['fig'] = fig
                fdict[key]['axs'] = axs

            dataset_files = self.find_dataset_files(run)

            # Loop over datasets (sensors)
            for dataset_file in dataset_files:
                # initialize list to hold values, overwritten with each sensor
                for key in summary_keys:
                    fdict[key]['vals'] = [] 

                with open(dataset_file, 'rb') as f:
                    ds = pkl.load(f)
                    summ = ds.summary
                    detector = ds.detector
                    sensor = detectors_map[detector]['detectorName']
                    ax_idx = map_detector_to_raft(detector)
                    
                    # Loop over neighboring amps, store coeff, nonlinearity, asymmetry data
                    for amp_i, amp_j in amp_neighbors:
                        try:
                            fdict['xtalk_coeff']['vals'].append(summ[amp_i][amp_j]['xtalk_coeff'])
                            fdict['xtalk_coeff']['vals'].append(summ[amp_j][amp_i]['xtalk_coeff'])
                            fdict['nonlinearity']['vals'].append(summ[amp_i][amp_j]['nonlinearity'])
                            fdict['nonlinearity']['vals'].append(summ[amp_j][amp_i]['nonlinearity'])
                            fdict['diff']['vals'].append(summ[amp_i][amp_j]['diff'])
                        except KeyError:
                            continue

                # plot histograms of coeff, nonlin, asymmetry data
                for key, bin_scale, xlim in zip(summary_keys, bin_scales, xlims):
                    axs = fdict[key]['axs']
                    vals = fdict[key]['vals']
                    vals = np.array(vals)
                    use = vals[np.logical_and(vals > xlim[0], vals < xlim[1])]
                    outliers = vals[np.logical_or(vals < xlim[0], vals > xlim[1])]
                    outliers.sort()

                    max_val= vals.max()
                    n_outliers = len(outliers)

                    bmax = math.ceil(bin_scale * max(use))/bin_scale
                    bmin = math.floor(bin_scale * min(use))/bin_scale
                    bstep = 1/(5*bin_scale)
                    bins = np.arange(bmin, bmax + bstep, bstep)

                    ax = axs[ax_idx]
                    ax.hist(use, bins=bins, label=f'{n_outliers} Outliers: max = {max_val:.2e}')
                    ax.set_xlim(xlim)
                    ax.tick_params(labelsize=16)
                    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
                    ax.legend(loc = 'upper right', prop={'size': 20})

            # Get raftname for plot title
            stem = os.path.basename(dataset_files[0]).split('.')[0]
            detnum = stem.split('_')[1][3:] 
            raft = detectors_map[detnum]['raftName']

            # Save figures
            for key, title, fname in zip(summary_keys, plot_titles, figure_names):
                fig = fdict[key]['fig']
                fig.suptitle(f'{run} {raft} {title}', fontsize=20)
                fig.tight_layout()
                figname = f'{run}_{raft}_{fname}'
                pathname = os.path.join(self.config.write_to, 'plots/focal_plane')
                Path(pathname).mkdir(parents=True, exist_ok=True)
                full_name = os.path.join(pathname, figname)
                fig.savefig(full_name)
                print(f'Wrote {full_name}')
                plt.close()

    def plot_profiles(self):
        for run in self.config.runs:
            dataset_files = self.find_dataset_files(run)

            for dataset_file in dataset_files:
                with open(dataset_file, 'rb') as f:
                    dataset = pkl.load(f)
                    run = dataset.run
                    detector = dataset.detector

                    for amp_i, amp_j in self.amp_pairs:
                        if (amp_i, amp_j) in self.bad_pairs[run][f'det{detector}']:
                            continue
                        fig1 = plt.figure(figsize=(24, 16))
                        ax1 = fig1.add_axes((.1, .3, .8, .6))
                        fig2, ax3 = plt.subplots(figsize=(24, 16))
                        profiles = []

                        for amp1, amp2 in [(amp_i, amp_j), (amp_j, amp_i)]:
                            summary = dataset.summary[amp2][amp1]
                            norm = summary['xtalk_coeff']
                            profile = dataset.profiles[amp2][amp1]
                            profiles.append(profile)
                           
                            # Plot xtalk
                            self.plot_xtalk(profile, amp2, amp1, ax1) 
                            self.plot_xtalk(profile, amp2, amp1, ax3, norm=norm) 
       
                        # Plot difference
                        ax2 = fig1.add_axes((.1, .1, .8, .2), sharex=ax1)
                        diff = profiles[0] - profiles[1] #1 -> 2 - 2 -> 1
                        self.plot_diff(diff, amp_i, amp_j, ax2)
                        
                        # Add labels, grid, title, and save
                        for ax in [ax1, ax2, ax3]:
                            ax.legend(loc='lower left', prop={'size': 20})
                            ax.set_xlabel('Source Amp Pixel Signal (adu)', fontsize=18.0)
                            ax.grid(b=True)

                        ax1.set_ylabel('Xtalk Ratio Mean', fontsize=18.0)
                        ax2.set_ylabel('Xtalk difference', fontsize=18.0)
                        ax3.set_ylabel('Xtalk Ratio Mean (Normalized to Average)')

                        ax1.set_title(f'{run} det{detector} xtalk')
                        ax3.set_title(f'{run} det{detector} xtalk normalized')
 
                        fig1name = f'{run}_det{detector}_xtalk_{amp1}_{amp2}.png' 
                        fig2name = f'{run}_det{detector}_xtalk_{amp1}_{amp2}_normalized.png' 
                        pathname = os.path.join(self.config.write_to, 'plots/profiles')
                        Path(pathname).mkdir(parents=True, exist_ok=True)
                        full_name1 = os.path.join(pathname, fig1name)
                        fig1.savefig(full_name1)
                        print(f'Wrote {full_name1}')
                        plt.close(fig=fig1)
                        full_name2 = os.path.join(pathname, fig2name)
                        fig2.savefig(full_name2)
                        print(f'Wrote {full_name2}')
                        plt.close(fig=fig2)
    
    def plot_xtalk(self, profile, target, source, ax, norm=1.0):
        profile_colors = ['tab:blue', 'tab:green', 'k', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        xarr, yarr, xerr, yerr = profile.unpack()
        ax.errorbar(xarr, yarr/norm, xerr=xerr, yerr=yerr/norm,
                    label=f'{source}->{target} ratio. Norm = {norm:.03e}', color=profile_colors[int(source[2])])

    def plot_diff(self, profile, amp1, amp2, ax):
        xarr, yarr, xerr, yerr = profile.unpack()
        ax.errorbar(xarr, yarr, xerr=xerr, yerr=yerr,
                    label=f'{amp1}->{amp2} - {amp2}->{amp1}', color='tab:cyan')
 
def main(runs, **kwargs):
    data_loc = kwargs.get('data_loc')
    write_to = kwargs.get('write_to')
    make_datasets = kwargs.get('make_datasets')
    plot_profiles = kwargs.get('plot_profiles')
    plot_focalplane = kwargs.get('plot_focalplane')

    config = xtalk_nonlinearityConfig(runs, data_loc, write_to,
                                      make_datasets, plot_profiles, plot_focalplane)
    task = xtalk_nonlinearityTask(config)
    task.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Xtalk nonlinearity analysis')

    parser.add_argument('runs', nargs="+", help='A list of runs (as strings) to analyze')

    parser.add_argument('--data_loc', default='/home/adriansh/lsst_devel/analysis/BOT/xtalk/',
                        help='Where to look for xtalk ratios and fluxes')

    parser.add_argument('--write_to', default='/home/adriansh/lsst_devel/analysis/BOT/xtalk/',
                        help='Where to write the analysis products to.')

    parser.add_argument('--make_datasets', dest='make_datasets',
                        action='store_true')
    parser.add_argument('--no-make_datasets', dest='make_datasets',
                        action='store_false')

    parser.add_argument('--plot_profiles', dest='plot_profiles',
                        action='store_true')
    parser.add_argument('--no-plot_profiles', dest='plot_profiles',
                        action='store_false')

    parser.add_argument('--plot_focalplane', dest='plot_focalplane',
                        action='store_true')
    parser.add_argument('--no-plot_focalplane', dest='plot_focalplane',
                        action='store_false')

    parser.set_defaults(make_datasets=True, plot_focalplane=True, plot_profiles=True)

    args = parser.parse_args()

    runs = args.runs

    kwargs = dict(data_loc=args.data_loc,
                  write_to = args.write_to,
                  make_datasets=args.make_datasets,
                  plot_profiles=args.plot_profiles,
                  plot_focalplane=args.plot_focalplane)
 
    main(runs, **kwargs)
