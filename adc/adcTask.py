import argparse
import pickle as pkl
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 

from collections import defaultdict
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from multiprocessing import Pool
from lsst.daf.butler import Butler
from eo_testing.utils.array_tools import get_true_windows
from eo_testing.utils.statistics import var_se2
from eo_testing.utils.defaults import all_amps, amp_plot_order
from eo_testing.utils.plotting_utils import plot_fp, map_detName_to_fp
from eo_testing.utils.profile import profile


class adc():
    def __init__(self, dnl, err, measured_bins):
        self.dnl = dnl
        self.err = err
        self.measured_bins = measured_bins

    def make_bin_edges(self):
        adcmax = 2**18
        adc_edges = np.nan * np.arange(adcmax+1)
        edge_unc = np.nan * np.arange(adcmax+1)
        all_bins = np.arange(adcmax)
        all_dnl = np.nan * np.zeros(adcmax)
        all_unc = np.nan * np.zeros(adcmax)

        measured_bins = self.measured_bins
        all_dnl[measured_bins] = self.dnl
        all_unc[measured_bins] = self.err

        useable_windows = get_true_windows(np.in1d(all_bins, self.measured_bins))

        for uw in useable_windows:
            dnl_window = all_dnl[uw]
            dnl_unc_window = all_unc[uw]
            uw_min = np.min(uw)
            uw_max = np.max(uw)
            temp_edges = [uw_min]
            temp_unc = [dnl_unc_window[0]/2]*2

            for d in dnl_window:
                temp_edges.append(temp_edges[-1] + d + 1)

            for u in dnl_unc_window[1:]:
                temp_unc.append(u - temp_unc[-1])

            xvals = np.arange(uw_min, uw_max + 2)
            adc_edges[xvals] = temp_edges
            edge_unc[xvals] = temp_unc

        self.edges = adc_edges
        self.edge_unc = edge_unc

    def update(self, new_dnl, new_err, new_bins):
        x = np.arange(2**18)
        dnl_arrays = np.zeros((2, len(x)))
        weight_arrays = np.zeros(dnl_arrays.shape)

        measured_bins = self.measured_bins

        dnl_arrays[0][measured_bins] = self.dnl
        dnl_arrays[1][new_bins] = new_dnl

        weight_arrays[0][measured_bins] = 1/self.err**2
        weight_arrays[1][new_bins] = 1/new_err**2

        dw = dnl_arrays * weight_arrays

        # this is (weights**2 * var).sum() / (weights.sum())**2
        dnl_curve_var = 1/(weight_arrays.sum(axis=0))
        dnl_curve_err = np.sqrt(dnl_curve_var)

        dnl_curve = dw.sum(axis=0)/weight_arrays.sum(axis=0) # Takes weighted average when distributions overlap

        dnl_finite = np.isfinite(dnl_curve)
        updated_dnl = dnl_curve[dnl_finite]
        updated_err = dnl_curve_err[dnl_finite]
        updated_bins = x[dnl_finite]

        self.dnl = updated_dnl
        self.err = updated_err
        self.measured_bins = updated_bins

        
class adc_dataset():
    def __init__(self, detName, detType, summary_data, adcs):
        self.detName = detName
        self.detType = detType
        self.summary_data = summary_data 
        if adcs is not None:
            self.adcs = adcs


class adcTaskConfig():
    def __init__(self, runs, **kwargs):
        write_to = kwargs.pop('write_to', os.path.join(os.path.expanduser('~'), '/analysis/adc/'))

        self.runs = runs
        self.overwrite_datasets = kwargs.pop('overwrite_datasets', False)
        self.make_datasets = kwargs.pop('make_datasets', True)
        self.make_plots = kwargs.pop('make_plots', True)
        self.write_adcs = kwargs.pop('write_adcs', False)
        self.detectors = kwargs.pop('detectors', 'ALL_DETECTORS')
        self.repo = kwargs.pop('repo', '/sdf/group/rubin/repo/ir2/butler.yaml')
        self.amps = kwargs.pop('amps', all_amps)
        self.dataset_loc = os.path.join(write_to, 'datasets/dnl')
        self.plot_loc = os.path.join(write_to, f'plots/dnl')
        self.instrument = kwargs.pop('instrument', 'LSSTCam')
        self.observation_type = kwargs.pop('observation_type', ['dark'])


class adcTask():
    def __init__(self, config):
        self.config = config
        self.collections = 'LSSTCam/raw/all'
        self.butler = Butler(self.config.repo, collections=self.collections)

    def get_det_raft_pairs(self, where):

        if self.config.detectors == 'ALL_DETECTORS':
            new_where = where
        else:
            detstr = ', '.join(self.config.detectors)
            new_where = where + f'and detector in ({detstr})'
        
        recordClasses = self.butler.registry.queryDimensionRecords('detector', where=new_where)
        det_raft_pairs = sorted([(rc.id, rc.full_name) for rc in recordClasses])
        
        return det_raft_pairs

    def get_datasets(self, run):
        globstring = os.path.join(self.config.dataset_loc, 
                                  f'{run}*dnl_dataset.pkl')
        dataset_files = glob.glob(globstring)

        return dataset_files

    def get_expTimes(self, where):
        records = list(self.butler.registry.queryDimensionRecords('exposure', where=where))
        expTimes = sorted(list({record.exposure_time for record in records}))
        
        return expTimes

    def make_dnl_info(self, im_arr):
        min_counts = 150
        polyorder = 3
        sgw = 65

        hb_min = im_arr.min()
        hb_max = im_arr.max()

        # Extend hist bins for filtering if necessary
        if hb_max - hb_min < 250:
            hb_min -= 100
            hb_max += 100

        # bright pixels can make hist_bins much bigger than it "should" be
        hist_bins = np.arange(hb_min, hb_max+1)

        counts, mbins = np.histogram(im_arr, bins=hist_bins)
        
        smoothed = savgol_filter(counts, sgw, polyorder)

        useable = smoothed > min_counts

        smoothed *= (counts.sum() / smoothed.sum())
        dnl_measurement = counts[useable] / smoothed[useable] - 1 # this includes poisson noise on top of dnl measurement
        dnl_pois_err2 = counts[useable] / smoothed[useable]**2 # variance of dnl measurement due to poisson noise
        mbins = mbins[:-1][useable] # np.histogram function gives bin edges so take left edge

        dnl_info = {'dnl': dnl_measurement, 'err': np.sqrt(dnl_pois_err2), 'bins': mbins}

        return dnl_info
        
    def make_datasets(self):
        base_where = f"instrument='{self.config.instrument}'\n"

        obs_type_str = "', '".join(self.config.observation_type)

        for run in self.config.runs:
            det_raft_pairs = self.get_det_raft_pairs(base_where)
            amps_list = self.config.amps
            where = base_where + f"and exposure.science_program='{run}'\n" \
            f"and exposure.observation_type in ('{obs_type_str}')"

            for detector, detName in det_raft_pairs:
                # Skip Corner rafts
                if detName[-2] in 'GW':
                    continue

                if not self.config.overwrite_datasets:
                    if self.check_dataset_exists(detName, run):
                        print(f'Dataset for {detName} exists. Skipping.')
                        continue

                dataId = {'detector': detector}

                datarefs = list(self.butler.registry.queryDatasets(datasetType='raw', 
                                collections=self.collections, where=where, dataId=dataId))

                exp_data = defaultdict(lambda: defaultdict(list))

                adcs, detType = self.make_adcs_dict(datarefs, amps_list)

                dataset = self.initialize_dataset(detName, detType, adcs)

                self.write_dataset(dataset, run)

    def initialize_dataset(self, detName, detType, adcs):
        if adcs != {}:
            summary_data = self.make_summary_data(adcs)

            if self.config.write_adcs:
                ds_adcs = adcs
            else: # write None to save memory
                ds_adcs = None

            dataset = adc_dataset(detName, detType, summary_data, ds_adcs)
            return dataset

        else:
           return None 


    def make_adcs_dict(self, datarefs, amps_list):
        adcs = {}
        detType = ''

        for iref, dataref in enumerate(datarefs):
            exp = self.butler.get(dataref)
            det = exp.getDetector()

            ampNames = [amp.getName() for amp in det if amp.getName() in amps_list]

            # getting the image array
            trimmed_ims = [exp.getMaskedImage()[amp.getRawDataBBox()].getImage()
                           for amp in det if amp.getName() in amps_list]
            im_arrs = [trimmed_im.getArray().flatten().astype(int)
                      for trimmed_im in trimmed_ims]

            with Pool(4) as p:
                dnl_info_list = p.map(self.make_dnl_info, im_arrs)

            amp_dnl_info = {ampName: dnl_info 
                            for ampName, dnl_info in zip(ampNames, dnl_info_list)}

            if iref == 0:
                detType = det.getPhysicalType()
                for ampName, dnl_info in amp_dnl_info.items():
                    adcs[ampName] = self.make_adc(dnl_info)
                continue
            else:
                for ampName, adc in adcs.items():
                    dnl_info = amp_dnl_info[ampName]
                    self.update_adc(adc, dnl_info)

        return adcs, detType
 
    def update_adc(self, adc, dnl_info):
        dnl, err, mbins = dnl_info['dnl'], dnl_info['err'], dnl_info['bins']
        adc.update(dnl, err, mbins)

    def make_adc(self, dnl_info):
        dnl, err, mbins = dnl_info['dnl'], dnl_info['err'], dnl_info['bins']

        return adc(dnl, err, mbins)

    def make_summary_data(self, adcs):
        summary_data = {ampName: {} for ampName in adcs.keys()}

        profile_bin_size = 200

        for ampName, adc in adcs.items():
            dnl = adc.dnl
            err = adc.err
            bins = adc.measured_bins
            
            profile_bins = np.arange(bins[0], bins[-1]+profile_bin_size, profile_bin_size)

            dnl_std, edges, binnum = binned_statistic(bins, dnl, 
                                                      statistic='std', bins=profile_bins)
            dnl_var = dnl_std**2

            noise_estimate, edges, binnum = binned_statistic(bins, err**2, 
                                                             statistic='mean', bins=profile_bins)
            dnl_var_se2, edges, binnum = binned_statistic(bins, dnl, statistic=var_se2, bins=profile_bins)

            corrected_dnl_var = dnl_var - noise_estimate
            dnl_std = np.sqrt(corrected_dnl_var)
            dnl_std_se = dnl_var_se2 / (2 * dnl_std)
            
            bin_mids = (profile_bins[1:] + profile_bins[:-1]) / 2

            dnl_profile = profile(bin_mids, dnl_std, 
                                  np.ones(len(profile_bins)) * profile_bin_size / 2, 
                                  dnl_std_se)

            summary_data[ampName]['profile'] = dnl_profile
            summary_data[ampName]['dnl_std'] = dnl.std(ddof=1)

        return summary_data 

    def summarize_dnl_data(self, dataset): 
        result = np.zeros((2, 8))

        for iamp, (ampName, amp_summary) in enumerate(dataset.summary_data.items()):
            dnl_std = amp_summary['dnl_std']

            if iamp < 8:
                result[0, iamp] = dnl_std
            else:
                result[1, 15-iamp] = dnl_std

        return result

    def plot_dnl(self, run, dataset, fig, axs):
        adcmax = 2**18
        detName = dataset.detName
        detType = dataset.detType
        summary_data = dataset.summary_data

        fig.suptitle(f'{run} {detName} DNL Standard Deviation ({detType})', fontsize=30)

        maxy = 0.25

        has_data = False

        for amp, ax in zip(amp_plot_order, axs.ravel()):
            ax.cla()

            ylims = (0, maxy)

            if len(profile.xarr) > 3:
                # Throw out the first few points because they're too close to bias level if possible
                xarr, yarr, yerr = profile.xarr[3:], profile.yarr[3:], profile.yerr[3:] 
            else:
                xarr, yarr, yerr = profile.xarr, profile.yarr, profile.yerr 

            max_dnl_std = yarr.max()

            if max_dnl_std > maxy:
                ylims = (0, max_dnl_std+0.02)

            ax.set_ylim(ylims)

            ax.grid(visible=True)
            
            ax.errorbar(xarr, yarr, yerr=yerr, linestyle='None', marker='.', ms=8)
            ax.set_title(amp, fontsize=22)
            ax.tick_params(labelsize=24, rotation=30)

        for ax in axs[-1]:
            ax.set_xlabel('Signal (adu)', fontsize=24)
        for ax in axs[:, 0]:
            ax.set_ylabel(r'$\sigma_{DNL}$ (LSB)', fontsize=24)

        fname = os.path.join(self.config.plot_loc, f'{run}_{detName}_DNL_std.png')

        if has_data:
            fig.savefig(fname)
            print(f'Wrote {fname}')

    def make_plots(self):
        for run in self.config.runs:
            cmap = 'Reds'
            os.makedirs(self.config.plot_loc, exist_ok=True)

            fp_fig, fp_axs = plot_fp()
            fp_data = {}

            dataset_files = self.get_datasets(run)

            maxs = np.zeros(len(dataset_files))

            detfig, detaxs = plt.subplots(8, 2, figsize=(28, 20), sharex=True)

            for ifile, dataset_file in enumerate(dataset_files):
                with open(dataset_file, 'rb') as f:
                    dataset = pkl.load(f)
                    detName = dataset.detName

                    self.plot_dnl(run, dataset, detfig, detaxs)
                    det_summary = self.summarize_dnl_data(dataset)

                    fp_data[detName] = det_summary

                    maxs[ifile] = np.max(det_summary)

            title = f'{run}_DNL std'
            fname = f'{run}_fp_dnl_std.png'

            mu = np.mean(maxs)
            sig = np.std(maxs)

            max10 = int(np.ceil(np.max(10*maxs[maxs<=mu+3*sig])))
            bounds = np.linspace(0, max10+1, 11)/10

            for detName, data in fp_data.items():
                fp_tup = map_detName_to_fp(detName)
                if fp_tup == 0:
                    continue
                ax = fp_axs[fp_tup]
                im = ax.imshow(data, cmap=cmap,
                               norm=colors.BoundaryNorm(boundaries=bounds,
                                                        ncolors=256))
                ax.set_aspect(4)

            fp_fig.subplots_adjust(right=0.9, top=0.9)
            cbar_ax = fp_fig.add_axes([0.93, 0.2, 0.025, 0.6])
            fp_fig.colorbar(im, cax=cbar_ax, format='%.2e', ticks=bounds)
            cbar_ax.tick_params(labelsize=13)
            fp_fig.suptitle(title, fontsize=30)
            fp_figName = os.path.join(self.config.plot_loc, fname)
            fp_fig.savefig(fp_figName)
            print(f'Wrote {fp_figName}')
                                      
    def run(self):
        if self.config.make_datasets:
            self.make_datasets()
        if self.config.make_plots:
            self.make_plots()

    def get_det_dataset_file(self, detName, run):
        pkl_name = f'{run}_{detName}_dnl_dataset.pkl'
        pkl_file_name = os.path.join(self.config.dataset_loc, pkl_name)
        
        return pkl_file_name

    def check_dataset_exists(self, detName, run):
        pkl_file_name = self.get_det_dataset_file(detName, run)
        dataset_exists = os.path.exists(pkl_file_name)

        return dataset_exists

    def write_dataset(self, dataset, run):
        if dataset is not None:
            os.makedirs(self.config.dataset_loc, exist_ok=True)
            pkl_name = f'{run}_{dataset.detName}_dnl_dataset.pkl'
            pkl_file_name = os.path.join(self.config.dataset_loc, pkl_name)
            with open(pkl_file_name, 'wb') as pkl_file:
                pkl.dump(dataset, pkl_file)
            print(f'Wrote {pkl_file_name}')


def main(runs, **kwargs):
    """ Make and plot adc datasets"""
    adcConfig = adcTaskConfig(runs, **kwargs)
    task = adcTask(adcConfig)
    task.run()


if __name__ == "__main__":
    default_write_to = os.path.join(os.path.expanduser('~'), 'analysis/adc/')

    parser = argparse.ArgumentParser(description='Analyze flat files for given runs at amplifier level')
    parser.add_argument('runs', nargs="+", help='A list of runs to analyze')
    parser.add_argument('--write_to', default=default_write_to,
                        help='Where to save the results. '\
                        f'Default: {default_write_to}')
    parser.add_argument('--repo', default='/sdf/group/lsst/camera/IandT/repo_gen3/BOT_data/butler.yaml',
                        help='Where to look for data. Default: '\
                        '/sdf/group/lsst/camera/IandT/repo_gen3/BOT_data/butler.yaml')
    parser.add_argument('--instrument', default='LSSTCam', help='LSSTCam or LSST-TS8. Default: '\
                        'LSSTCam.')
    parser.add_argument('--observation_type', nargs='+', default='dark', help='flat or dark or both. '\
                        'Default: dark.')

    parser.add_argument('--make_datasets', action='store_true', dest='make_datasets')
    parser.add_argument('--no-make_datasets', action='store_false', dest='make_datasets')

    parser.add_argument('--make_plots', action='store_true', dest='make_plots')
    parser.add_argument('--no-make_plots', action='store_false', dest='make_plots')

    parser.add_argument('--write_adcs', action='store_true', dest='write_adcs')
    parser.add_argument('--no-write_adcs', action='store_false', dest='write_adcs')

    parser.add_argument('--overwrite_datasets', action='store_true', dest='overwrite_datasets')
    parser.add_argument('--no-overwrite_datasets', action='store_false', dest='overwrite_datasets')

    parser.add_argument('--detectors', nargs='+', default='ALL_DETECTORS', help='List of detectors. '
                        'Default: ALL_DETECTORS')
    parser.add_argument('--amps', nargs='+', default=all_amps, help=f'List of amps. '
                        "Example: 'C17' 'C10'. Default: {all_amps}")
    parser.set_defaults(make_datasets=True, make_plots=True, overwrite_datasets=False, write_adcs=False)


    kwargs = vars(parser.parse_args())

    runs = kwargs.pop('runs')

    main(runs, **kwargs)
