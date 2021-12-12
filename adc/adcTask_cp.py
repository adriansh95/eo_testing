import argparse
import pickle as pkl
import glob
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
from collections import defaultdict

import lsst.afw.image as afw_image
import lsst.geom as lsst_geom
from lsst.eo_utils.sflat.file_utils import get_sflat_files_run
from lsst.daf.butler import Butler
from eo_testing.utils.profile import profile, make_profile
from eo_testing.utils.defaults import all_amps, all_sensors 
from eo_testing.utils.plotting_utils import plot_fp, map_detName_to_fp


def get_bbox(keyword, dxmin=0, dymin=0, dxmax=0, dymax=0):
    """
    Parse an NOAO section keyword value (e.g., DATASEC = '[1:509,1:200]') from
    the FITS header and return the corresponding bounding box for sub-image
    retrieval.
    """
    xmin, xmax, ymin, ymax = [val - 1 for val in eval(keyword.replace(':', ','))]
    bbox = lsst_geom.Box2I(lsst_geom.Point2I(xmin + dxmin, ymin + dymin),
                           lsst_geom.Point2I(xmax + dxmax, ymax + dymax))
    return bbox


def im_to_array(imfile, hdu):
    image = afw_image.ImageF(imfile, hdu)
    mask = afw_image.Mask(image.getDimensions())
    masked_image = afw_image.MaskedImageF(image, mask)
    md = afw_image.ImageFitsReader(imfile, hdu).readMetadata()
    imsec = masked_image.Factory(masked_image, get_bbox(md.get('DATASEC')))
    values, mask, var = imsec.getArrays()
    values = values.flatten()
    values = values.astype(int)
 
    return values


def get_raftnames_runs(runs):
    """Get raftnames for a list of runs"""
    raftnames = set({})

    for run in runs:
        raftnames = raftnames.union(set(get_sflat_files_run(run).keys()))

    return sorted(raftnames)


def get_raftnames_run(run):
    """Get raftnames for a single run"""
    raftnames = get_sflat_files_run(run).keys()
    return sorted(raftnames)


class adc_bit_dataset():
    def __init__(self, detName, detType, amp_medians, bit_probs, bit_prob_errs):
        self.detName = detName
        self.detType = detType
        self.amp_medians = amp_medians
        self.bit_probs = bit_probs
        self.bit_prob_errs = bit_prob_errs


class adc_bitTaskConfig():
    def __init__(self, run, write_to='/u/ec/adriansh/lsst/analysis/adc/', 
                 maxp=5, make_datasets=True, make_plots=True):
        self.run = run
        self.maxp = maxp
        self.make_datasets = make_datasets
        self.make_plots = make_plots
        self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
        self.markers = ['.', 'v', '^', '+', 'x']
        self.dataset_loc = os.path.join(write_to, 'datasets/bit')
        self.plot_loc = os.path.join(write_to, 'plots/bit')
        self.detectors = [90, 91, 92, 93, 94, 95, 96,
                          97, 98, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                          126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 
                          136, 137, 138, 139, 140, 141, 142, 143]
#27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
#                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
#                          53, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
#                          84, 85, 86, 87, 88, 89,

class adc_bitTask():
    def __init__(self, config):
        self.config = config

        runstr = f"'{self.config.run}'"

        self.where = f"""
        instrument = 'LSSTCam'
        and exposure.observation_type='flat'
        and exposure.science_program in ({runstr})
        """
        
        repo = '/sdf/group/lsst/camera/IandT/repo_gen3/bot/butler.yaml'
        self.collections = 'LSSTCam/raw/all'
        self.butler = Butler(repo, collections=self.collections)

    def get_datasets(self):
        globstring = os.path.join(self.config.dataset_loc, f'{self.config.run}*bit_dataset.pkl')
        dataset_files = glob.glob(globstring)
        return dataset_files

    def get_expTimes(self):
        dtype = 'raw'
        
        records = list(self.butler.registry.queryDimensionRecords('exposure', where=self.where))
        expTimes = []
        
        for record in records:
            expTimes.append(record.exposure_time)
        
        expTimes = sorted(list(set(expTimes)))
        return expTimes

    def make_datasets(self):
        expTimes = self.get_expTimes()
        dtype = 'raw'
        maxp = self.config.maxp

        for detector in self.config.detectors:
            dataId = {'detector': detector}
            amp_medians = defaultdict(lambda: np.zeros(len(expTimes)))
            bit_probs = defaultdict(lambda: np.zeros((len(expTimes), maxp)))
            bit_prob_errs = defaultdict(lambda: np.zeros((len(expTimes), maxp)))
            detName = ''
            detType = ''

            for i, expTime in enumerate(expTimes):
                new_where = self.where + f'and exposure.exposure_time={expTime}'
                datarefs = list(self.butler.registry.queryDatasets(datasetType=dtype, 
                                collections=self.collections, where=new_where, dataId=dataId))
                # dict to store medians of each flat of equal exptime
                # will take the median of all of these as amp_median
                temp_meds = defaultdict(lambda: np.zeros(len(datarefs)))
                bits_on = defaultdict(lambda: np.zeros(maxp))
                npix = 0

                for iref, dataref in enumerate(datarefs):
                    exp = self.butler.get(dataref)
                    det = exp.getDetector()
                    npix += len(exp.getMaskedImage()[det[0].getRawDataBBox()].getImage().getArray().flatten())

                    if i == 0 and iref == 0:
                        detName = det.getName()
                        detType = det.getPhysicalType()

                    for amp in det:
                        ampName = amp.getName()

                        # getting the image array
                        trimmed_im = exp.getMaskedImage()[amp.getRawDataBBox()].getImage()
                        im_arr = trimmed_im.getArray().flatten().astype(int)

                        # computing and storing the median of the amp
                        med = np.median(im_arr)
                        temp_meds[ampName][iref] = med

                        # computing and storing bit numbers
                        for bitp in range(maxp):
                            bit = 2**bitp
                            bit_on = (im_arr & bit) // bit
                            n_on = np.sum(bit_on)
                            bits_on[ampName][bitp] += n_on

                for amp, meds in temp_meds.items():
                    amp_medians[amp][i] = np.median(meds)
                    bit_prob = bits_on[amp]/npix
                    bit_probs[amp][i] = bit_prob
                    bit_prob_errs[amp][i] = np.sqrt(bit_prob * (1-bit_prob) / npix)

            amp_medians = dict(amp_medians)
            bit_probs = dict(bit_probs)
            bit_prob_errs = dict(bit_prob_errs)
            dataset = adc_bit_dataset(detName, detType, amp_medians, bit_probs, bit_prob_errs)
            self.write_dataset(dataset)

    def analyze_dataset(self, dataset):
        abs_array = np.zeros((2, 8))
        tot_array = np.zeros((2, 8))

        for iamp, (amp, probs) in enumerate(dataset.bit_probs.items()):
            bit_bias = probs.transpose() - 0.5
            bit_vals = np.diag(np.power(2, np.arange(self.config.maxp)))
            bias_effs = np.matmul(bit_vals, bit_bias)
            tot_bias_eff = np.sum(bias_effs, axis=0)

            abs_bias_eff = np.mean(np.abs(bias_effs))
            #abs_bias_eff = np.std(bias_effs)
            mean_tot_bias_eff = np.mean(tot_bias_eff)

            if iamp < 8:
                abs_array[0, iamp] = abs_bias_eff
                tot_array[0, iamp] = mean_tot_bias_eff
            else:
                abs_array[1, 15-iamp] = abs_bias_eff
                tot_array[1, 15-iamp] = mean_tot_bias_eff

        return abs_array, tot_array

    def plot_detector(self, dataset, figs, axs):
        p_fig, b_fig, t_fig = figs 
        p_axs, b_axs, t_axs = axs 

        detName = dataset.detName
        detType = dataset.detType
        amp_medians = dataset.amp_medians
        bit_probs = dataset.bit_probs
        bit_prob_errs = dataset.bit_prob_errs

        for p_ax, b_ax, t_ax, (amp, medians) in zip(p_axs.ravel(), b_axs.ravel(), t_axs.ravel(), amp_medians.items()):
            for ax in (p_ax, b_ax, t_ax):
                ax.clear()
                ax.grid(b=True)
                ax.set_title(f'{amp}', fontsize=16)

            p_ax.set_ylim(0.45, 0.55) 
            b_ax.set_ylim(-0.15, 0.15) 
            t_ax.set_ylim(-0.15, 0.15) 

            bit_vals = np.diag(np.power(2, np.arange(self.config.maxp)))
            bit_bias = bit_probs[amp].transpose() - 0.5
            prob_errs = bit_prob_errs[amp].transpose()

            bias_effs = np.matmul(bit_vals, bit_bias)
            bias_eff_errs = np.matmul(bit_vals, prob_errs)
            tot_bias_eff = np.sum(bias_effs, axis=0)
            tot_bias_eff_err = np.sum(bias_eff_errs, axis=0) 

            t_ax.errorbar(medians, tot_bias_eff, yerr=tot_bias_eff_err, linestyle='None', marker='.', ms=14) 

            for ibit, (probs, color, marker) in enumerate(zip(bit_probs[amp].transpose(),
                                                              self.config.colors, self.config.markers)):
                max_diff = np.max(np.abs(probs-0.5))

                if max_diff > 0.05:
                    p_ax.set_ylim(0.5-max_diff-0.02, 0.5+max_diff+0.02)

                p_ax.errorbar(medians, probs, yerr=prob_errs[ibit], 
                              label=f'bit {ibit}', linestyle='None', c=color, marker=marker)

                b_ax.errorbar(medians, bias_effs[ibit], yerr=bias_eff_errs[ibit], 
                              label=f'bit {ibit}', linestyle='None', c=color, marker=marker)

        for ax in np.array([p_axs[-1], b_axs[-1], t_axs[-1]]).flatten():
            ax.set_xlabel('Amp Median (adu)', fontsize=18)
        
        for p_ax, b_ax, t_ax in zip(p_axs[:, 0], b_axs[:, 0], t_axs[:, 0]):
            p_ax.set_ylabel('Bit Probability') 
            b_ax.set_ylabel('Bit Bias Effect (ADU)') 
            t_ax.set_ylabel('Total Bias Effect (ADU)') 

        handles, labels = p_ax.get_legend_handles_labels()

        p_fig.suptitle(f'{self.config.run} {detName} Bit Probablilities ({detType})', fontsize=24) 
        b_fig.suptitle(f'{self.config.run} {detName} Bit Bias Effect ({detType})', fontsize=24) 
        t_fig.suptitle(f'{self.config.run} {detName} Total Bias Effect ({detType})', fontsize=24) 

        for fig in (p_fig, b_fig, t_fig):
            fig.legend(handles, labels, loc='upper right', fontsize=24)

        p_basename = f'{detName}_bit_probs_flux.png' 
        b_basename = f'{detName}_bit_bias_effect_flux.png' 
        t_basename = f'{detName}_total_bias_effect_flux.png' 

        p_fname = os.path.join(self.config.plot_loc, p_basename)
        b_fname = os.path.join(self.config.plot_loc, b_basename)
        t_fname = os.path.join(self.config.plot_loc, t_basename)

        p_fig.savefig(p_fname)
        print(f'Wrote {p_fname}')
        b_fig.savefig(b_fname)
        print(f'Wrote {b_fname}')
        t_fig.savefig(t_fname)
        print(f'Wrote {t_fname}')

    def make_plots(self):
        cmap = 'Reds'
        os.makedirs(self.config.plot_loc, exist_ok=True)

        fp_abs_fig, fp_abs_axs = plot_fp()
        fp_tot_fig, fp_tot_axs = plot_fp()
        fp_figs = {'abs': fp_abs_fig, 'tot': fp_tot_fig}
        fp_axs = {'abs': fp_abs_axs, 'tot': fp_tot_axs}
        fp_data = defaultdict(dict)

        p_fig, p_axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True) #probabilities
        b_fig, b_axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True) #bias effect
        t_fig, t_axs = plt.subplots(8, 2, figsize=(28, 20), sharex=True) #total effect
        det_figs = (p_fig, b_fig, t_fig)
        det_axs = (p_axs, b_axs, t_axs) 

        dataset_files = self.get_datasets()

        maxs = np.zeros((2, len(dataset_files))) # abs then tot

        for ifile, dataset_file in enumerate(dataset_files):
            with open(dataset_file, 'rb') as f:
                dataset = pkl.load(f)
                detName = dataset.detName
                #self.plot_detector(dataset, det_figs, det_axs)

                abs_array, tot_array = self.analyze_dataset(dataset)
                fp_data['abs'][detName] = abs_array
                fp_data['tot'][detName] = tot_array

                maxs[0, ifile] = np.max(abs_array)
                maxs[1, ifile] = np.max(tot_array)

        titles = ['Mean Absolute Bias Effect', 'Mean Total Bias Effect']
        fnames = ['fp_abs_bias_effect.png', 'fp_tot_bias_effect.png']

        for cmaxs, title, fname, (key, val) in zip(maxs, titles, fnames, fp_data.items()):
            fig = fp_figs[key]
            axs = fp_axs[key]

            mu = np.mean(cmaxs)
            sig = np.std(cmaxs)

            cmax10 = int(np.ceil(np.max(10*cmaxs[cmaxs<mu+3*sig])))
            bounds = np.linspace(0, cmax10+1, 11)/10
            #cmax = np.ceil(np.max(10*cmaxs[cmaxs<mu+3*sig]))/10

            for detName, data in val.items():
                fp_tup = map_detName_to_fp(detName)
                ax = axs[fp_tup]
                im = ax.imshow(data, cmap=cmap,
                               norm=colors.BoundaryNorm(boundaries=bounds,
                                                        ncolors=256))
                ax.set_aspect(4)

            fig.subplots_adjust(right=0.9, top=0.9)
            cbar_ax = fig.add_axes([0.93, 0.2, 0.025, 0.6])
            fig.colorbar(im, cax=cbar_ax, format='%.2e', ticks=bounds)
            fig.suptitle(title, fontsize=24)
            figName = os.path.join(self.config.plot_loc, fname)
            fig.savefig(figName)
            print(f'Wrote {figName}')
                                      
    def run(self):
        if self.config.make_datasets:
            self.make_datasets()
        if self.config.make_plots:
            self.make_plots()

    def write_dataset(self, dataset):
        os.makedirs(self.config.dataset_loc, exist_ok=True)
        pkl_name = f'{self.config.run}_{dataset.detName}_bit_dataset.pkl'
        pkl_file_name = os.path.join(self.config.dataset_loc, pkl_name)
        with open(pkl_file_name, 'wb') as pkl_file:
            pkl.dump(dataset, pkl_file)
        print(f'Wrote {pkl_file_name}')


class adc_profile_dataset():
    """Simple class to hold raft, sensor, and profile information"""
    def __init__(self, run, bit_powers, bin_power):
        self.__dict__['run'] = run
        self.__dict__['bit_powers'] = bit_powers
        self.__dict__['bin_size'] = 2**bin_power
        self.__dict__['profiles'] = {}
        self.__dict__['sensors'] = {}

        for raftname in get_raftnames_run(self.run):
            self.sensors[raftname] = {SNN: '' for SNN in all_sensors}
            self.profiles[raftname] = {SNN: {f'amp{hdu:02d}': {} for hdu in range(1, 17)} for SNN in all_sensors}

    def __setattr__(self, attribute, value):
        """Protect class attributes"""
        if attribute not in self.__dict__:
            raise AttributeError(f"{attribute} is not already a member of adc_profile_dataset, which"
                                 " does not support setting of new attributes.")
        else:
            self.__dict__[attribute] = value

    def fill_dataset(self):
        """Fill the datset with sensor and
        profile information"""
        run = self.run
        prof_bin_size = self.bin_size
        sflat_files_dict = get_sflat_files_run(run)

        for raftname, SNN_dict in sflat_files_dict.items():
            for SNN, imagetype_dict in SNN_dict.items():
                imagetype_dict = sflat_files_dict[raftname][SNN]
                files_list = imagetype_dict['SFLAT']

                try:
                    imfile = files_list[0]
                    print(f'Working on {raftname} {SNN}')
                except IndexError:
                    print(f'No files for {raftname} {SNN} run {run}')
                    continue

                im_md = afw_image.readMetadata(imfile, 0)
                CCD_name = im_md.get('LSST_NUM')
                self.sensors[raftname][SNN] = CCD_name

                for hdu in range(1, 17):
                    imarray = im_to_array(imfile, hdu)
                    imarray, low, high = scipy.stats.sigmaclip(imarray, low=3.0, high=3.0)

                    prof_xmin = min(imarray)
                    prof_xmin = prof_xmin - prof_xmin % prof_bin_size
                    prof_xmax = max(imarray)
                    prof_xmax = prof_xmax - prof_xmax % prof_bin_size
                    prof_nx = (prof_xmax - prof_xmin) / prof_bin_size

                    for pbit in self.bit_powers:
                        prof_bit = 2**pbit
                        bit_on = (imarray & prof_bit) // prof_bit
                        xval, yval, xerr, yerr = make_profile(imarray, bit_on, 
                                                              xlims=(prof_xmin, prof_xmax), bin_size=prof_bin_size)

                        if len(xval) == 0:
                            print(f'Len xval = 0 for {run} {raftname} {SNN} amp{hdu:02d}, '
                                  'possible dead channel')
                            break
                        else:
                            prof = profile(xval, yval, xerr, yerr)
                            self.profiles[raftname][SNN][f'amp{hdu:02d}'][f'{prof_bit}s_bit'] = prof

    def save(self, write_to):
        """Save the dataset"""
        run = self.run
        pkl_name = f'{run}_'
        for bit_power in self.bit_powers:
            bit = 2**bit_power
            pkl_name += f'{bit}s_'
        pkl_name += 'profile_dataset.pkl'
        pkl_file_name = os.path.join(write_to, 'datasets', pkl_name)
        with open(pkl_file_name, 'wb') as pkl_file:
            pkl.dump(self, pkl_file)
        print(f'Wrote {pkl_file_name}')


class adcProfileConfig():
    def __init__(self, runs, powers, shade, bin_power, write_to):
        self.runs = runs
        self.powers = powers
        self.shade = shade
        self.bin_power = bin_power
        self.write_to = write_to


class adcProfileTask():
    """Task to make ADC profiles"""
    def __init__(self, config):
        self.config = config

    def find_dataset_file(self, run):
        """Find adc dataset file if it exists"""
        fname = f'{run}_'
        for power in self.config.powers:
            bit = 2**power
            fname += f'{bit}s_'
        fname += 'profile_dataset.pkl'
        globstring = os.path.join(self.config.write_to, 'datasets', fname)
        filepath = glob.glob(globstring)
        return filepath

    def make_datasets(self):
        """Create and save adc datasets"""
        for run in self.config.runs:
            dataset_list = self.find_dataset_file(run)
            if len(dataset_list) == 0:
                adc_dataset = adc_profile_dataset(run, self.config.powers, self.config.bin_power)
                adc_dataset.fill_dataset()
                adc_dataset.save(self.config.write_to)
            else:
                print(f'ADC file {dataset_list[0]} exists')
                continue

    def get_datasets(self):
        """Retrieve adc datasets for runs"""
        datasets = {}
        for run in self.config.runs:
            dataset_file = self.find_dataset_file(run)[0]
            with open(dataset_file, 'rb') as f:
                datasets[run] = pkl.load(f)
        return datasets

    def make_plots(self):
        """Make profile plots from adc datasets"""
        raftnames = get_raftnames_runs(self.config.runs)
        datasets = self.get_datasets()
        shade_bit = 2**self.config.shade
        bits = np.power(2, self.config.powers)
        prof_bin_size = 2**self.config.bin_power

        for raftname in raftnames:
            for SNN in all_sensors:
                print(f'Making plots for {raftname} {SNN}')
                profs, p_axs = plt.subplots(8, 2, sharey=True, sharex=False, figsize=(30, 20))
                p_axs = p_axs.ravel()
                title = f'{raftname} {SNN} '
                fname = f'{raftname}_{SNN}_'

                for hdu, p_ax in enumerate(p_axs, start=1):
                    p_ax.set_title(f'amp{hdu:02d}')
                    array_mins = []
                    array_maxs = []

                    # I could change the dataset class to include to mins from the start and skip this step
                    for run in self.config.runs:
                        dataset = datasets[run]
                        sensor = dataset.sensors[raftname][SNN]
                        # This guy below could be fixed. Maybe change profiles class to include more than one bit
                        profiles = dataset.profiles[raftname][SNN][f'amp{hdu:02d}']

                        try:
                            profile_x = profiles[f'{bits[0]}s_bit'].xarr

                        except KeyError:
                            print((f'No profile data for {run} {raftname} {SNN} amp{hdu:02d}'))
                            continue

                        profile_xerr = profiles[f'{bits[0]}s_bit'].xerr
                        array_mins.append(min(profile_x))
                        array_maxs.append(max(profile_x))

                        for prof_bit in bits:
                            profile_y = profiles[f'{prof_bit}s_bit'].yarr
                            profile_yerr = profiles[f'{prof_bit}s_bit'].yerr
                            p_ax.errorbar(profile_x, profile_y, xerr=profile_xerr, yerr=profile_yerr,
                                          label=f'{run} {sensor} {prof_bit}s bit')

                    try:
                        prof_xmin = min(array_mins)
                        prof_xmin = prof_xmin - prof_xmin % prof_bin_size
                        prof_xmax = max(array_maxs)
                        prof_xmax = prof_xmax - prof_xmax % prof_bin_size
                    except ValueError:
                        continue

                    p_ax.grid()
                    p_ax.set_xlim([prof_xmin, prof_xmax])
                    p_ax.set_ylim([0.35, 0.65])

                    # Try to make this shading stuff better, it's a mess
                    p_range = np.arange(prof_xmin, prof_xmax + 1, dtype=int)
                    bar_lims = []
                    bit_flips = (p_range % shade_bit) == 0
                    bar_lims = p_range[bit_flips].tolist()

                    if len(bar_lims) == 0:
                        if ((p_range[0] & shade_bit) // shade_bit) == 1:
                            bar_lims.append(p_range[0])
                            bar_lims.append(p_range[-1])
                    else:
                        if ((min(bar_lims) & shade_bit) // shade_bit) == 0:
                            bar_lims.append(p_range[0])
                        if ((max(bar_lims) & shade_bit) // shade_bit) == 1:
                            bar_lims.append(p_range[-1])

                    bar_color = 'g'
                    bar_alpha = 0.10
 
                    if len(bar_lims) > 0:
                        bar_lims.sort()
                        for i in range(len(bar_lims) // 2):
                            p_ax.axvspan(bar_lims[2 * i], bar_lims[2 * i + 1], color=bar_color, alpha=bar_alpha)

                for prof_bit in bits:
                    title += f'{prof_bit}s '
                    fname += f'{prof_bit}s_'

                title += 'Bits Profiles'
                fname += 'bits_profiles.png'

                profs.suptitle(title, fontsize=16)
                profs.text(0.5, 0.0, 'Signal (adu)', ha='center')
                profs.text(0.0, 0.5, 'Bit Frequency in Bin', va='center', rotation='vertical')
                handles, labels = p_ax.get_legend_handles_labels()
                profs.legend(handles, labels, 'upper right', fontsize='xx-large')
                profs.tight_layout()
                profs.subplots_adjust(top=0.96)

                pathname = os.path.join(self.config.write_to, 'plots', f'{raftname}')
                os.makedirs(pathname, exist_ok=True)
                fig.savefig(os.path.join(pathname, fname))
                plt.close()


def main(runs, taskType, make_datasets, make_plots, **kwargs):
    """ Make and plot adc datasets"""
    if taskType == 'profile':
        powers = kwargs['powers']
        shade = kwargs['shade']
        bin_power = kwargs['bin_power']
        write_to = kwargs['write_to']
        profileConfig = adcProfileConfig(runs, powers, shade, bin_power, write_to) 
        profileTask = adcProfileTask(profileConfig)

        if make_datasets:
            profileTask.make_datasets()
        if make_plots:
            profileTask.make_plots()

    elif taskType == 'bit':
        write_to = kwargs['write_to']
        for run in runs:
            bitConfig = adc_bitTaskConfig(run, write_to=write_to, make_datasets=make_datasets, make_plots=make_plots)
            bitTask = adc_bitTask(bitConfig)
            bitTask.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze sflat files for given runs at amplifier level')
    parser.add_argument('runs', nargs="+", help='A list of runs (as strings) to analyze')
    parser.add_argument('--taskType', default='bit', help='Which task to run (bit or profile)')
    parser.add_argument('--powers', nargs='+', default=[0, 1], help='Bit powers to analyze')
    parser.add_argument('--shade', default=9, help='Bit power to shade for in profile plots')
    parser.add_argument('--binpower', default=4, help='bin size log base 2 in profiles')
    parser.add_argument('--write_to', default='/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/',
                        help='Where to save the results. '
                        'Default: \'/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/\'')
    parser.add_argument('--make_datasets', dest='make_datasets',
                        action='store_true')
    parser.add_argument('--no-make_datasets', dest='make_datasets',
                        action='store_false')
    parser.add_argument('--make_plots', dest='make_plots',
                        action='store_true')
    parser.add_argument('--no-make_plots', dest='make_plots',
                        action='store_false')

    parser.set_defaults(make_datasets=True, make_plots=True)

    args = parser.parse_args()

    runs = args.runs
    taskType = args.taskType
    make_datasets = args.make_datasets
    make_plots = args.make_plots

    kwargs = dict(powers = args.powers,
                  shade = args.shade,
                  bin_power = args.binpower,
                  write_to = args.write_to)

    main(runs, taskType, make_datasets, make_plots, **kwargs)
