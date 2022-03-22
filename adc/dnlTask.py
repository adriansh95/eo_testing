import argparse
import pickle as pkl
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
from collections import defaultdict

from scipy.stats import binned_statistic, sigmaclip, moment
from scipy.signal import savgol_filter
from lsst.daf.butler import Butler
from eo_testing.utils.array_tools import get_true_windows
from eo_testing.utils.defaults import all_amps, amp_plot_order
from eo_testing.utils.plotting_utils import plot_fp, map_detName_to_fp


class adc():
    def __init__(self, dnl, dnl_err, bins):
        self.dnl = dnl
        self.dnl_err = dnl_err
        self.measured_bins = bins

    def make_bin_edges(self):
        adcmax = 2**18
        adc_edges = np.nan * np.arange(adcmax+1)
        edge_unc = np.nan * np.arange(adcmax+1)
        all_bins = np.arange(adcmax)
        all_dnl = np.nan * np.zeros(adcmax)
        all_unc = np.nan * np.zeros(adcmax)

        measured_bins = self.measured_bins
        all_dnl[measured_bins] = self.dnl
        all_unc[measured_bins] = self.dnl_err

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

        edge_cond = np.isfinite(adc_edges)
        adc_edges = adc_edges[edge_cond]
        edge_unc = edge_unc[edge_cond]

        self.edges = adc_edges
        self.edge_unc = edge_unc

    def update(self, new_dnl, new_dnl_err, new_bins):
        adcmax = 2**18
        x = np.arange(adcmax)
        old_dnl = 0 * x
        old_weights = 0 * x

        old_bins = self.measured_bins
        old_dnl[old_bins] = self.dnl
        old_weighst[old_bins] = 1/self.dnl_err**2

        
class dnl_dataset():
    def __init__(self, detName, detType, dnl_info):#, adcs):
        self.detName = detName
        self.detType = detType
        self.dnl_info = dnl_info
        #self.adcs = adcs


class bit_dataset():
    def __init__(self, detName, detType, bit_data):
        self.detName = detName
        self.detType = detType
        self.bit_data = bit_data


class adcTaskConfig():
    def __init__(self, run, **kwargs):#write_to='/u/ec/adriansh/lsst/analysis/adc/', 
                 #maxp=5, make_datasets=True, make_plots=True):
        self.run = run
        self.write_to = kwargs.pop('write_to', '/u/ec/adriansh/lsst/analysis/adc/')
        self.make_datasets = kwargs.pop('make_datasets', True)
        self.make_plots = kwargs.pop('make_plots', True)
        self.make_adcs = kwargs.pop('make_adcs', False)
        self.detectors = kwargs.pop('detectors', 'ALL_DETECTORS')
        self.repo = kwargs.pop('repo', '/sdf/group/lsst/camera/IandT/repo_gen3/bot_data/butler.yaml')
        self.amps = kwargs.pop('amps', all_amps)
        self.maxp = 5
        self.adcmax = 2**18
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
        self.markers = ['.', 'v', '^', '+', 'x']
        self.ds_type = 'dnl'
        self.dataset_loc = os.path.join(self.write_to, f'datasets/{self.ds_type}')
        self.min_counts = 200
        self.plot_loc = os.path.join(self.write_to, f'plots/{self.ds_type}')


class adcTask():
    def __init__(self, config):
        self.config = config

        runstr = f"'{self.config.run}'"

        self.where = f"""
        instrument = 'LSSTCam'
        and exposure.observation_type='flat'
        and exposure.science_program in ({runstr})
        """
        
        repo = self.config.repo
        self.collections = 'LSSTCam/raw/all'
        self.butler = Butler(repo, collections=self.collections)

    def get_det_raft_pairs(self):

        if self.config.detectors == 'ALL_DETECTORS':
            new_where = self.where
        else:
            detstr = ', '.join(self.config.detectors)
            new_where = self.where + f'and detector in (detstr)'
        
        recordClasses = self.butler.registry.queryDimensionRecords('detector', where=new_where)
        det_raft_pairs = sorted([(rc.id, rc.full_name) for rc in recordClasses])
        
        return det_raft_pairs

    def get_datasets(self):
        globstring = os.path.join(self.config.dataset_loc, 
                                  f'{self.config.run}*{self.config.ds_type}_dataset.pkl')
        dataset_files = glob.glob(globstring)

        return dataset_files

    def get_expTimes(self):
        dtype = 'raw'
        records = list(self.butler.registry.queryDimensionRecords('exposure', where=self.where))
        expTimes = sorted(list({record.exposure_time for record in records}))
        
        return expTimes

    def compute_probs(self, x, y, windows):
        min_useable = 7500
        w = windows[1] - windows[0]
        bit_counts = [y * ((x & bp) // bp) for bp in np.power(2, range(5))]

        bit_sums, tempedges, n = binned_statistic(x, bit_counts, statistic='sum', bins=windows)
        tot_sums, tempedges, n = binned_statistic(x, y, statistic='sum', bins=windows)

        useable = tot_sums > min_useable
        tot_sums = tot_sums[useable]
        probs = np.zeros((len(bit_sums), len(useable)))

        probs = np.array([bit_sum[useable]/tot_sums for bit_sum in bit_sums])

        tempedges = tempedges[:-1][useable]
        mids = tempedges + w/2
        prob_errs = np.sqrt((probs * (1-probs)) / tot_sums)
        result = dict(y=probs, yerr=prob_errs, x=mids)

        return result
 
    def make_adcs(self, dnl_data):
        adcmax = self.config.adcmax
        min_counts = self.config.min_counts
        polyorder = 3
        sw = 65
        adcs = {}

        for amp, exp_dict in dnl_data.items():
            x = np.arange(adcmax)
            dnl_arrays = np.zeros((len(exp_dict.keys), len(x)))
            weight_arrays = np.zeros(dnl_arrays.shape)
            dnl_err2_arrays = np.zeros(dnl_arrays.shape)

            for iexp, dnl_dict in exp_dict.items():
                dnl = 0 * x
                weights = 0 * x
                dnl_err2 = 0 * x
                measured_bins = dnl_dict['measured_bins']
                dnl[measured_bins] = dnl_dict['dnl']
                dnl_err2[measured_bins] = dnl_dict['dnl_err2']
                weights[measured_bins] = 1/dnl_dict['dnl_err2']

                dnl_arrays[iexp] = dnl
                weight_arrays[iexp] = weights
                dnl_err2_arrays[iexp] = dnl_err2 


            dw = dnl_arrays * weight_arrays

            # this is (weights**2 * var).sum() / (weights.sum())**2
            dnl_curve_var = 1/(weight_arrays.sum(axis=0))
            dnl_curve_err = np.sqrt(dnl_curve_var)

            dnl_curve = dw.sum(axis=0)/weight_arrays.sum(axis=0) # Takes weighted average when distributions overlap

            dnl_finite = np.isfinite(dnl_curve)
            adc_dnl = dnl_curve[dnl_finite]
            adc_dnl_err = dnl_curve_err[dnl_finite]
            adc_bins = x[dnl_finite]

            measured_adc = adc(adc_dnl, adc_dnl_err, adc_bins)
            adcs[amp] = measured_adc

        return adcs

    def analyze_counts(self, amp_dict, w=32):
        probs = {}

        for amp, counts_dict in amp_dict.items():
            data_dict = {}
            x = np.array(sorted([edge for edge in counts_dict.keys()]))
            y = np.array([counts_dict[edge] for edge in x])

            mav = np.convolve(y, np.ones(w+1), 'valid') / (w+1)
            mav = mav.round().astype(int)
            mavx = x[w//2: -w//2] 

            xmin = x.min()
            xmax = x.max() + w
            xmin -= xmin % w
            windows = np.arange(xmin, xmax, w)

            raw_probs = self.compute_probs(x, y, windows)
            mav_probs = self.compute_probs(mavx, mav, windows)

            mav_inter = np.in1d(mav_probs['x'], raw_probs['x'])
            raw_inter = np.in1d(raw_probs['x'], mav_probs['x']) ## placeholder for better code

            for d, inter in zip([mav_probs, raw_probs], [mav_inter, raw_inter]):
                for key in ['y', 'yerr']:
                    dvals = d[key]
                    d[key] = np.array([row[inter] for row in dvals])

                d['x'] = d['x'][inter]

            mpy = mav_probs['y']
            rpy = raw_probs['y']
            mpyerr = mav_probs['yerr']
            rpyerr = raw_probs['yerr']
            data_dict['y'] = raw_probs['y'] / (2*mav_probs['y'])
            data_dict['yerr'] = np.sqrt(((1/(2*mpy))**2)*(rpyerr**2)+\
                                        ((rpy/(2*mpy**2))**2)*(mpyerr**2))
            data_dict['x'] = raw_probs['x']
            probs[amp] = data_dict

        return probs

    def make_datasets(self):
        expTimes = self.get_expTimes()
        dtype = 'raw'

        det_raft_pairs = self.get_det_raft_pairs()

        amps_list = self.config.amps

        for detector, detName in det_raft_pairs:
            dataId = {'detector': detector}
            detType = ''
            #temp_counts = defaultdict(lambda: defaultdict(lambda: 0))

            #counts_data = defaultdict(lambda: defaultdict(lambda: dict()))
            #filtered_data = defaultdict(lambda: defaultdict(lambda: dict()))
            #errs_data = defaultdict(lambda: defaultdict(lambda: dict()))

            dnl_data = {amp: {iTime: {} for iTime in range(len(expTimes))} \
                        for amp in amps_list}
            summary_data = {amp: {'med': [], 'dnl_std': [], 'dnl_std_err': []} for amp in amps_list}

            for i, expTime in enumerate(expTimes):
                new_where = self.where + f'and exposure.exposure_time={expTime}'
                datarefs = list(self.butler.registry.queryDatasets(datasetType=dtype, 
                                collections=self.collections, where=new_where, dataId=dataId))

                n_exps = len(datarefs)
                t_counts = defaultdict(lambda: defaultdict(lambda: 0))

                for iref, dataref in enumerate(datarefs):
                    exp = self.butler.get(dataref)
                    det = exp.getDetector()

                    if i == 0 and iref == 0:
                        detType = det.getPhysicalType()

                    for amp in det:
                        ampName = amp.getName()
                        if ampName not in amps_list:
                            continue

                        # getting the image array
                        trimmed_im = exp.getMaskedImage()[amp.getRawDataBBox()].getImage()
                        im_arr = trimmed_im.getArray().flatten().astype(int)

                        hb_min = im_arr.min()
                        hb_max = im_arr.max()

                        # Extend hist bins for filtering if necessary
                        if hb_max - hb_min < 250:
                            hb_min -= 100
                            hb_max += 100

                        # bright pixels can make hist_bins much bigger than it "should" be
                        hist_bins = np.arange(hb_min, hb_max+1)

                        temp_hist, temp_edges = np.histogram(im_arr, bins=hist_bins)

                        for num, edge in zip(temp_hist, temp_edges[:-1]):
                            t_counts[ampName][edge] += num
                            #temp_counts[ampName][edge] += num

                amp_summary, temp_dnl_data = self.analyze_exps(t_counts, n_exps)

                for amp, data_dict in amp_summary.items():
                    dnl_data[amp][i] = temp_dnl_data[amp]

                    #for adcbin, ddict in temp_adc_amp_data.items(): ##
                    #    counts_data[amp][i][adcbin] = ddict['observed'] ##
                    #    filtered_data[amp][i][adcbin] = ddict['expected'] ##
                    #    errs_data[amp][i][adcbin] = ddict['error'] ##

                    if data_dict is not None:
                        summary_data[amp]['med'].append(data_dict['med'])
                        summary_data[amp]['dnl_std'].append(data_dict['dnl_std'])
                        summary_data[amp]['dnl_std_err'].append(data_dict['dnl_std_err'])

            if self.config.make_adcs:
                adcs = self.make_adcs(dnl_data)
            else:
                adcs = {}

            dataset = dnl_dataset(detName, detType, summary_data, adcs)

            #bit_data = self.analyze_counts(temp_counts)
            #dataset = bit_dataset(detName, detType, bit_data) 
            self.write_dataset(dataset)

    def analyze_exps(self, counts_dict, n_exps):
        sgw = 65
        polyorder = 3
        min_counts = self.config.min_counts
        amp_summary = defaultdict(dict)
        dnl_data = {amp: {} for amp in counts_dict.keys()}

        for amp, cdict in counts_dict.items():
            bins = np.array(sorted([k for k in cdict.keys()]))
            counts = np.array([cdict[b]/n_exps for b in bins])

            try:
                smoothed = savgol_filter(counts, sgw, polyorder)
            except ValueError:
                amp_summary[amp] = None
                continue

            count_vars = smoothed / n_exps
            useable = smoothed > min_counts 

            if len(smoothed[useable]) < 5:
                amp_summary[amp] = None
                continue

            dnl_measurement = counts[useable] / smoothed[useable] - 1 # this includes poisson noise on top of dnl measurement
            dnl_pois_err2 = count_vars[useable] / smoothed[useable]**2 # variance of dnl measurement due to poisson noise

            dnl_data[amp]['dnl'] = dnl_measurement
            dnl_data[amp]['dnl_err2'] = dnl_pois_err2
            dnl_data[amp]['measured_bins'] = bins[useable]

            dnl_var = dnl_measurement.var() - dnl_pois_err2.mean()
            dnl_std = np.sqrt(dnl_var)
            dnl_var_err2 = np.var(dnl_measurement - dnl_measurement.mean())/len(dnl_measurement)
            dnl_std_err2 = 1/(4*dnl_var) * dnl_var_err2
            dnl_std_err = np.sqrt(dnl_std_err2)
  
            amp_summary[amp]['med'] = np.median(bins[useable])
            amp_summary[amp]['dnl_std'] = dnl_std
            amp_summary[amp]['dnl_std_err'] = dnl_std_err

        return amp_summary, dnl_data

    def summarize_dnl_data(self, dataset):
        result = np.zeros((2, 8))

        for iamp, (amp, adc) in enumerate(dataset.adcs.items()):
            dnl = adc.dnl
            rms_dnl = np.sqrt(np.power(dnl, 2).mean())

            if iamp < 8:
                result[0, iamp] = rms_dnl
            else:
                result[1, 15-iamp] = rms_dnl

        return result

    def summarize_bit_data(self, dataset):
        result = np.zeros((2, 8))

        for iamp, (amp, data_dict) in enumerate(dataset.bit_data.items()):
            probs = data_dict['y']
            bit_bias = probs - 0.5
            bit_vals = np.diag(np.power(2, np.arange(self.config.maxp)))
            bias_effs = np.matmul(bit_vals, bit_bias)
            #tot_bias_eff = np.sum(bias_effs, axis=0)

            rms_bias_eff = np.sqrt(np.mean(bias_effs**2))
            #mean_tot_bias_eff = np.mean(tot_bias_eff)

            if iamp < 8:
                result[0, iamp] = rms_bias_eff
                #abs_array[0, iamp] = abs_bias_eff
                #tot_array[0, iamp] = mean_tot_bias_eff
            else:
                result[1, 15-iamp] = rms_bias_eff
                #abs_array[1, 15-iamp] = abs_bias_eff
                #tot_array[1, 15-iamp] = mean_tot_bias_eff

        return result #abs_array, tot_array

    def plot_dnl(self, dataset, fig, axs):
        adcmax = self.config.adcmax
        run = self.config.run
        detName = dataset.detName
        detType = dataset.detType
        dnl_info = dataset.dnl_info

        fig.suptitle(f'{run} {detName} DNL Standard Deviation ({detType})', fontsize=24)

        maxy = 0.25

        for amp, ax in zip(amp_plot_order, axs.ravel()):
            data_dict = dnl_info[amp]
            ax.cla()
            ylims = (0, maxy)
            meds = np.array(data_dict['med'])
            dnl_std = np.array(data_dict['dnl_std'])
            unc = np.array(data_dict['dnl_std_err'])

            try:
                max_dnl_std = dnl_std.max()
            except ValueError:
                print(f'No DNL data for {detName} {amp}')
                continue

            if max_dnl_std > maxy:
                ylims = (0, max_dnl_std+0.02)

            ax.set_ylim(ylims)

            ax.grid(visible=True)
            ax.errorbar(meds, dnl_std, yerr=unc, linestyle='None', marker='.', ms=8)
            ax.set_title(amp, fontsize=16)

        for ax in axs[-1]:
            ax.set_xlabel('Signal (adu)', fontsize=18)
        for ax in axs[:, 0]:
            ax.set_ylabel(r'$\sigma_{DNL}$ (LSB)', fontsize=14)

        fname = os.path.join(self.config.plot_loc, f'{run}_{detName}_DNL_std.png')
        fig.savefig(fname)
        print(f'Wrote {fname}')

    #def plot_adcs(self, dataset):
    #    adcmax = self.config.adcmax
    #    detName = dataset.detName
    #    detType = dataset.detType
    #    adcs = dataset.adcs
    #    av_over = 30

    #    fig, axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True)
    #    for ax in axs[-1]:
    #        ax.set_xlabel('Signal (adu)', fontsize=18)
    #    for ax in axs[:, 0]:
    #        ax.set_ylabel('RMS DNL (LSB)', fontsize=18)

    #    fig.suptitle(f'{detName} RMS DNL ({detType})', fontsize=24)

    #    for (amp, adc), ax in zip(adcs.items(), axs.ravel()):
    #        dnl = adc.dnl
    #        unc = adc.dnl_unc
    #        valid = np.isfinite(unc)
    #        windows = get_true_windows(valid)
    #        x = []
    #        y = []
    #        #yerr = []

    #        for window in windows:
    #            if len(window) > av_over + 1:
    #                med = int(np.median(window))
    #                med_idx = np.where(window == med)[0][0]
    #                sub_window = window[med_idx-(av_over//2): med_idx+(av_over//2)+1]
    #                av_dnl = dnl[sub_window].mean()
    #                rms_dnl = np.sqrt(np.power(dnl[sub_window], 2).mean())
    #                x.append(med)
    #                y.append(rms_dnl)
    #            else:
    #                continue

    #        ax.grid(visible=True)
    #        ax.errorbar(x, y, linestyle='None', marker='.', ms=14)
    #        ax.set_title(amp, fontsize=16)

    #    fname = os.path.join(self.config.plot_loc, 'RMS_DNL.png')
    #    fig.savefig(fname)
    #    print(f'Wrote {fname}')

    def plot_bit_data(self, dataset, figs, axs):
        p_fig, b_fig, t_fig = figs 
        p_axs, b_axs, t_axs = axs 

        detName = dataset.detName
        detType = dataset.detType
        bit_data = dataset.bit_data

        diff_cutoff = 0.07
        ylims = [(0.5-diff_cutoff, 0.5+diff_cutoff), (-0.15, 0.15), (-0.15, 0.15)]

        for p_ax, b_ax, t_ax, (amp, data_dict) in zip(p_axs.ravel(), b_axs.ravel(), t_axs.ravel(), bit_data.items()):
            x = data_dict['x']
            #x = np.log(x)

            for ax, ylim_tup in zip((p_ax, b_ax, t_ax), ylims):
                ax.clear()
                ax.grid(visible=True)
                ax.set_title(f'{amp}', fontsize=16)
                ax.set_ylim(ylim_tup)

            bit_vals = np.diag(np.power(2, np.arange(self.config.maxp)))
            bit_bias = data_dict['y'] - 0.5
            prob_errs = data_dict['yerr']

            bias_effs = np.matmul(bit_vals, bit_bias)
            bias_eff_errs = np.matmul(bit_vals, prob_errs)
            tot_bias_eff = np.sum(bias_effs, axis=0)
            tot_bias_eff_err = np.sum(bias_eff_errs, axis=0) 

            t_ax.errorbar(x, tot_bias_eff, yerr=tot_bias_eff_err, linestyle='None', marker='.', ms=14) 

            for ibit, (probs, color, marker) in enumerate(zip(data_dict['y'], self.config.colors, 
                                                              self.config.markers)):
                max_diff = np.max(np.abs(probs-0.5))

                #if max_diff > diff_cutoff:
                #    p_ax.set_ylim(0.5-max_diff-0.02, 0.5+max_diff+0.02)

                p_ax.errorbar(x, probs, yerr=prob_errs[ibit], label=f'bit {ibit}', 
                              linestyle='None', c=color, marker=marker)

                b_ax.errorbar(x, bias_effs[ibit], yerr=bias_eff_errs[ibit], 
                              label=f'bit {ibit}', linestyle='None', c=color, marker=marker)

        for ax in np.array([p_axs[-1], b_axs[-1], t_axs[-1]]).flatten():
            ax.set_xlabel('Signal (adu)', fontsize=18)
        
        for p_ax, b_ax, t_ax in zip(p_axs[:, 0], b_axs[:, 0], t_axs[:, 0]):
            p_ax.set_ylabel('Bit Probability') 
            b_ax.set_ylabel('Bit Bias Effect (ADU)') 
            t_ax.set_ylabel('Total Bias Effect (ADU)') 

        handles, labels = p_ax.get_legend_handles_labels()

        titles = [f'{self.config.run} {detName} Bit Probablilities ({detType})',
                  f'{self.config.run} {detName} Bit Bias Effect ({detType})',
                  f'{self.config.run} {detName} Total Bias Effect ({detType})']

        basenames = [f'{detName}_bit_probs_flux.png', f'{detName}_bit_bias_effect_flux.png',
                     f'{detName}_total_bias_effect_flux.png']

        for fig, title, basename in zip((p_fig, b_fig, t_fig), titles, basenames):
            fig.legend(handles, labels, loc='upper right', fontsize=24)
            fig.suptitle(title, fontsize=24) 
            fname = os.path.join(self.config.plot_loc, basename)
            fig.savefig(fname)
            print(f'Wrote {fname}')

    def make_plots(self):
        cmap = 'Reds'
        os.makedirs(self.config.plot_loc, exist_ok=True)

        fp_fig, fp_axs = plot_fp()
        fp_data = {}

        #p_fig, p_axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True) #probabilities
        #b_fig, b_axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True) #bias effect
        #t_fig, t_axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True) #total effect
        #det_figs = (p_fig, b_fig, t_fig)
        #det_axs = (p_axs, b_axs, t_axs) 

        dataset_files = self.get_datasets()

        maxs = np.zeros(len(dataset_files))

        detfig, detaxs = plt.subplots(8, 2, figsize=(28, 20), sharex=True)

        for ifile, dataset_file in enumerate(dataset_files):
            with open(dataset_file, 'rb') as f:
                dataset = pkl.load(f)
                detName = dataset.detName

                if self.config.ds_type == 'dnl':
                    self.plot_dnl(dataset, detfig, detaxs)
                    det_summary = self.summarize_dnl_data(dataset)
                elif self.config.ds_type == 'bit':
                    self.plot_bit_data(dataset, det_figs, det_axs)
                    det_summary = self.summarize_bit_data(dataset)

                #self.plot_adcs(dataset)

                fp_data[detName] = det_summary

                maxs[ifile] = np.max(det_summary)

        if self.config.ds_type == 'dnl':
            title = 'DNL std'
            fname = 'fp_rms_dnl.png'
        elif self.config.ds_type == 'bit':
            title = 'RMS Bit Bias Effect'
            fname = 'fp_rms_bit_bias_effect.png'

        mu = np.mean(maxs)
        sig = np.std(maxs)

        max10 = int(np.ceil(np.max(10*maxs[maxs<mu+3*sig])))
        bounds = np.linspace(0, max10+1, 11)/10

        for detName, data in fp_data.items():
            fp_tup = map_detName_to_fp(detName)
            ax = fp_axs[fp_tup]
            im = ax.imshow(data, cmap=cmap,
                           norm=colors.BoundaryNorm(boundaries=bounds,
                                                    ncolors=256))
            ax.set_aspect(4)

        fp_fig.subplots_adjust(right=0.9, top=0.9)
        cbar_ax = fp_fig.add_axes([0.93, 0.2, 0.025, 0.6])
        fp_fig.colorbar(im, cax=cbar_ax, format='%.2e', ticks=bounds)
        fp_fig.suptitle(title, fontsize=24)
        fp_figName = os.path.join(self.config.plot_loc, fname)
        fp_fig.savefig(fp_figName)
        print(f'Wrote {fp_figName}')
                                      
    def run(self):
        if self.config.make_datasets:
            self.make_datasets()
        if self.config.make_plots:
            self.make_plots()

    def write_dataset(self, dataset):
        os.makedirs(self.config.dataset_loc, exist_ok=True)
        pkl_name = f'{self.config.run}_{dataset.detName}_{self.config.ds_type}_dataset.pkl'
        pkl_file_name = os.path.join(self.config.dataset_loc, pkl_name)
        with open(pkl_file_name, 'wb') as pkl_file:
            pkl.dump(dataset, pkl_file)
        print(f'Wrote {pkl_file_name}')


def main(runs, **kwargs):
    """ Make and plot adc datasets"""
    for run in runs:
        bitConfig = adcTaskConfig(run, **kwargs)
        bitTask = adcTask(bitConfig)
        bitTask.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze flat files for given runs at amplifier level')
    parser.add_argument('runs', nargs="+", help='A list of runs to analyze')
    parser.add_argument('--write_to', default='/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/',
                        help='Where to save the results. '
                        'Default: \'/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/\'')
    parser.add_argument('--make_datasets', action='store_true', help='Boolean, Default: True')
    parser.add_argument('--make_plots', action='store_true', help='Boolean, Default: True')
    parser.add_argument('--make_adcs', action='store_false', help='Boolean, Default: False')
    parser.add_argument('--detectors', nargs='+', default='ALL_DETECTORS', help='List of detectors. '
                        'Default: ALL_DETECTORS')
    parser.add_argument('--amps', nargs='+', default=all_amps, help=f'List of amps. '
                        "Example: ['C17', 'C10']. Default: {all_amps}")


    kwargs = vars(parser.parse_args())

    runs = kwargs.pop('runs')

    main(runs, **kwargs)
