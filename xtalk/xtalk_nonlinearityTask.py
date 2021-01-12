import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import pickle as pkl
import glob
import os
import itertools

from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from pathlib import Path
from utils.defaults import all_amps
from utils.plot_fp import plot_fp, map_detector_to_tup
from utils.profile import profile, make_profile

amps = all_amps
amp_pairs = list(itertools.combinations(amps, r=2))

profile_colors = ['tab:blue', 'tab:green', 'k', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

class xtalkDataset:
    def __init__(self, run, detector):
        self.run = run
        self.detector = detector
        self.summary = defaultdict(lambda: defaultdict(dict))

class xtalk_nonlinearityConfig():
    """ Config class for xtalk_nonlinearityTask """
    def __init__(self, runs, xtalk_data_loc, write_to,
                 make_datasets, plot_focalplane,
                 norm_cutoff=45000, fit_threshold=70000,
                 profile_xlims=(30000, 200000), bin_size=500):
        self.runs = runs
        self.xtalk_data_loc = xtalk_data_loc
        self.write_to = write_to
        self.make_datasets = make_datasets
        self.plot_focalplane = plot_focalplane
        self.norm_cutoff = norm_cutoff
        self.fit_threshold = fit_threshold
        self.profile_xlims = profile_xlims
        self.bin_size = bin_size

class xtalk_nonlinearityTask():
    def __init__(self, config):
        self.config = config

    def find_rat_flux_data(self, run):
        fname = f'ratios_fluxes_{run}*.pkl'
        globstring = os.path.join(self.config.xtalk_data_loc, fname)
        filepaths = glob.glob(globstring)
        return filepaths

    def find_xtalk_datasets(self, run):
        fname = f'{run}*dataset.pkl'
        globstring = os.path.join(self.config.xtalk_data_loc, 'datasets', fname)
        filepaths = glob.glob(globstring)
        return filepaths

    def run(self):
        if self.config.make_datasets:
            self.make_datasets()
        if self.config.plot_focalplane:
            self.plot_focalplane()

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

                    for amp1, amp2 in amp_pairs:

                        # Get the xtalk data for amp pair
                        amp_fluxes_21 = np.array(fluxes[amp2][amp1])
                        amp_fluxes_12 = np.array(fluxes[amp1][amp2])
                        amp_ratios_21 = np.array(ratios[amp2][amp1])
                        amp_ratios_12 = np.array(ratios[amp1][amp2])

                        # Calculate the norms
                        norm_21 = self.calculate_norm(amp_ratios_21, amp_fluxes_21)
                        norm_12 = self.calculate_norm(amp_ratios_12, amp_fluxes_12)

                        # Make profiles
                        profile_21 = make_profile(amp_fluxes_21, amp_ratios_21,
                                                  xlims=self.config.profile_xlims,
                                                  bin_size=self.config.bin_size)
                        profile_12 = make_profile(amp_fluxes_12, amp_ratios_12,
                                                  xlims=self.config.profile_xlims,
                                                  bin_size=self.config.bin_size)
 
                        # Difference of profiles
                        diff = profile_21 - profile_12

                        # Plot profiles
                        plot_info = {'norm_21': norm_21, 'norm_12': norm_12, 'run': run,
                                    'amp2': amp2, 'amp1': amp1, 'detector': detector}
                        self.plot_profiles(profile_21, profile_12, diff, plot_info)

                        # Fit line to med/high flux end
                        fit_21, cov_21 = self.fit_xtalk(profile_21, norm=norm_21)
                        fit_12, cov_12 = self.fit_xtalk(profile_12, norm=norm_12)
                        slope_21 = fit_21[0]
                        slope_12 = fit_12[0]

                        # Apply Savitzky-Golay filter to smooth diff profile and take rms
                        smoothed_diff = savgol_filter(diff.yarr, 51, 3)
                        rms_diff = np.sqrt(np.mean(np.power(smoothed_diff, 2)))

                        # Compile summary statistics for amp pair in detector level dataset
                        summary_21 = dict(norm=norm_21, slope=slope_21, rms_diff=rms_diff)
                        summary_12 = dict(norm=norm_12, slope=slope_12, rms_diff=rms_diff)

                        dataset.summary[amp2][amp1] = summary_21
                        dataset.summary[amp1][amp2] = summary_12
                    
                self.save_dataset(dataset)         

    def save_dataset(self, dataset):
        run = dataset.run
        detector = dataset.detector
        
        # turn default dicts into regular dicts for pickling
        summary_temp = dict(dataset.summary)
        for k, v in summary_temp.items():
            vtemp = dict(v)
            summary_temp[k] = vtemp
        dataset.summary = summary_temp

        fname = f'{run}_det{detector}_xtalk_dataset.pkl'
        pathname = os.path.join(self.config.write_to, 'datasets')
        full_name = os.path.join(pathname, fname)
        Path(pathname).mkdir(parents=True, exist_ok=True)

        with open(full_name, 'wb') as f:
            pkl.dump(dataset, f)
        print(f'Wrote {full_name}')

    def calculate_norm(self, ratios, fluxes):
        norm_cutoff = self.config.norm_cutoff
        return np.mean(ratios[fluxes < norm_cutoff])

    def plot_focalplane(self):
        globstring = os.path.join(self.config.xtalk_data_loc, 'datasets/*xtalk_dataset.pkl')
        dataset_files = glob.glob(globstring)
        amps = all_amps

        summary_keys = ['norm', 'slope', 'rms_diff']

        plot_titles = ['Low Flux Xtalk Ratio', 'High Flux Slope',
                  'Amp Pair RMS Difference']

        figure_names = ['fp_low_flux_ratio.png', 'fp_high_flux_slope.png',
                        'fp_amp_pair_rms_diff.png']

        colorbounds = [[-1e-2, -1e-3, -1e-4, -1e-5, -1e-6, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                       np.concatenate((np.arange(-8e-6, 0, 1e-6), np.arange(1e-6, 9e-6, 1e-6))),
                       np.arange(0, 10e-5, 1e-5)]
                       
        cmaps = ["seismic", "seismic", "Reds"]

        fdict = defaultdict(dict)

        for key, colorbound, cmap in zip(summary_keys, colorbounds, cmaps):
            fig, axs = plot_fp()
            fdict[key]['fig'] = fig
            fdict[key]['axs'] = axs
            fdict[key]['bounds'] = colorbound
            fdict[key]['cmap'] = cmap

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
                            val = vdict[key]
                            vals[i, j] = val
        
                ax_idx = map_detector_to_tup(detector)
                ax = axs[ax_idx]
        
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
  
    def plot_profiles(self, profile_21, profile_12, diff, plot_info):
        norm_21 = plot_info['norm_21']
        norm_12 = plot_info['norm_12']
        amp2 = plot_info['amp2']
        amp1 = plot_info['amp1']
        detector = plot_info['detector']
        run = plot_info['run']

        fig = plt.figure(figsize=(24, 16))
        ax1 = fig.add_axes((.1, .3, .8, .6))
        
        # Plot normalized xtalk
        self.plot_xtalk(profile_21, amp2, amp1, ax1, norm=norm_21)
        self.plot_xtalk(profile_12, amp1, amp2, ax1, norm=norm_12)
       
       
        # Plot difference
        ax2 = fig.add_axes((.1, .1, .8, .2), sharex=ax1)
        self.plot_diff(diff, amp1, amp2, ax2)
        
        # Add labels, grid, title, and save
        ax1.legend(loc='lower left', prop={'size': 12})
        ax2.legend(loc='lower left', prop={'size': 12})
        ax2.set_xlabel('Source Amp Pixel Signal (adu)', fontsize=18.0)
        ax1.set_ylabel('Xtalk Ratio Mean (Normalized at Low Flux)', fontsize=18.0)
        ax2.set_ylabel('Xtalk difference', fontsize=18.0)
        ax1.grid(b=True)
        ax2.grid(b=True)
        ax1.set_title(f'{run} det{detector} xtalk')
        figname = f'{run}_det{detector}_xtalk_{amp1}_{amp2}.png' 
        pathname = os.path.join(self.config.write_to, 'plots/profiles')
        Path(pathname).mkdir(parents=True, exist_ok=True)
        full_name = os.path.join(pathname, figname)
        fig.savefig(full_name)
        print(f'Wrote {full_name}')
        plt.close()
    
    def plot_xtalk(self, profile, target, source, ax, norm=1.0):
        xarr, yarr, xerr, yerr = profile.unpack()
        ax.errorbar(xarr, yarr/norm, xerr=xerr, yerr=yerr/norm,
                    label=f'{source}->{target} ratio. Norm = {norm:.03e}', color=profile_colors[int(source[2])])

    def plot_diff(self, profile, amp1, amp2, ax):
        xarr, yarr, xerr, yerr = profile.unpack()
        ax.errorbar(xarr, yarr, xerr=xerr, yerr=yerr,
                    label=f'{amp1}->{amp2} - {amp2}->{amp1}', color='tab:cyan')

    def fit_xtalk(self, profile, norm=1.0):
        threshold = self.config.fit_threshold
        xarr, yarr, xerr, yerr = profile.unpack()
        line = lambda x, a, b: a*x + b
        xdata = xarr[xarr >= threshold]
        ydata = yarr[xarr >= threshold]
        sigmas = yerr[xarr >= threshold]
        p0 = [0, 0]
        result = curve_fit(line, xdata, ydata/norm, p0=p0, sigma=sigmas/norm)
        return result

    def plot_diff(self, profile, amp1, amp2, ax):
        xarr, yarr, xerr, yerr = profile.unpack()
        ax.errorbar(xarr, yarr, xerr=xerr, yerr=yerr,
                    label=f'{amp1}->{amp2} - {amp2}->{amp1}', color='tab:cyan')
 
#def flag_jumps(profile, norm=1.0):
#    xarr, yarr, xerr, yerr = profile.unpack()
#    
#    # Normalize
#    yarr = yarr/norm
#    yerr = yerr/norm
#    
#    # Average difference and std
#    ydiffs = yarr[1:] - yarr[:-1]
#    diff_mean = np.mean(ydiff)
#    diff_std = np.std(ydiffs)
#    jumps = np.array(xarr[1:][ydiffs>(diff_mean + sigma)]) # x coords where ydiff exceeds mean by sigma
#    return jumps

   
#def plot_fit(pars, target, source, ax):
#    x = np.linspace(xlims[0], xlims[1], 5001)
#    fit = np.poly1d(pars)
#    pars_str = ', '.join([f'{par:.03e}' for par in pars])
#    ax.plot(x, fit(x), label=f'{source}->{target} fit coeffs (descending): {pars_str}',
#            color=profile_colors[int(source[2])])

def main(runs, data_loc, write_to, make_datasets, plot_focalplane):
    config = xtalk_nonlinearityConfig(runs, data_loc, write_to,
                                      make_datasets, plot_focalplane)
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
    parser.add_argument('--plot_focalplane', dest='plot_focalplane',
                        action='store_true')
    parser.add_argument('--no-plot_focalplent', dest='plot_focalplane',
                        action='store_false')
    parser.set_defaults(make_datasets=True, plot_focalplane=True)

    args = parser.parse_args()

    runs = args.runs
    data_loc = args.data_loc
    write_to = args.write_to
    make_datasets = args.make_datasets
    plot_focalplane = args.plot_focalplane

    main(runs, data_loc, write_to, make_datasets, plot_focalplane)
