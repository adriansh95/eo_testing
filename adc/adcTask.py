import argparse
import pickle as pkl
import glob
from pathlib import Path
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import lsst.afw.image as afw_image
import lsst.geom as lsst_geom
from lsst.eo_utils.sflat.file_utils import get_sflat_files_run

SNNs = ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S20', 'S21', 'S22']


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


def mkProfile(xarr, yarr, nx=100, xmin=0., xmax=1.0, ymin=0., ymax=1.0):
    """Make a profile"""
    dx = (xmax-xmin) / nx
    bins = np.arange(xmin, xmax, dx)
    ind = np.digitize(xarr, bins)
    xval = []
    xerr = []
    yval = []
    yerr = []
    for i in range(len(bins)-1):
        here = (ind == i+1)
        ygood = np.logical_and(yarr >= ymin, yarr <= ymax)
        ok = np.logical_and(ygood, here)
        yinthisbin = yarr[ok]
        yhere = np.array(yinthisbin)
        n = len(yinthisbin)
        if n > 0:
            xval.append(0.5*(bins[i+1]+bins[i]))
            xerr.append(0.5*(bins[i+1]-bins[i]))
            yval.append(yhere.mean())
            yerr.append(yhere.std()/n)

    return xval, yval, xerr, yerr


def im_to_array(imfile, hdu):
    image = afw_image.ImageF(imfile, hdu)
    mask = afw_image.Mask(image.getDimensions())
    masked_image = afw_image.MaskedImageF(image, mask)
    md = afw_image.readMetadata(imfile, hdu)
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


class profile:
    """Simple class holding x and y array and error bars"""
    def __init__(self, xarr, yarr, xerr, yerr):
        self.xarr = xarr
        self.yarr = yarr
        self.xerr = xerr
        self.yerr = yerr


class adcDataset:
    """Simple class to hold raft, sensor, and profile information"""
    def __init__(self, run, bit_powers, bin_power):
        self.__dict__['run'] = run
        self.__dict__['bit_powers'] = bit_powers
        self.__dict__['bin_size'] = 2**bin_power
        self.__dict__['profiles'] = {}
        self.__dict__['sensors'] = {}

        for raftname in get_raftnames_run(self.run):
            self.sensors[raftname] = {SNN: '' for SNN in SNNs}
            self.profiles[raftname] = {SNN: {f'amp{hdu:02d}': {} for hdu in range(1, 17)} for SNN in SNNs}

    def __setattr__(self, attribute, value):
        """Protect class attributes"""
        if attribute not in self.__dict__:
            raise AttributeError(f"{attribute} is not already a member of adcDataset, which"
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
                        xval, yval, xerr, yerr = mkProfile(imarray, bit_on, nx=prof_nx, xmin=prof_xmin, xmax=prof_xmax)

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
        pkl_name += 'dataset.pkl'
        pkl_file_name = os.path.join(write_to, 'datasets', pkl_name)
        with open(pkl_file_name, 'wb') as pkl_file:
            pkl.dump(self, pkl_file)
        print(f'Wrote {pkl_file_name}')


class adcTask():
    """Task to make ADC profiles"""
    def __init__(self, runs, powers, shade, bin_power, saveto):
        self.runs = runs
        self.powers = powers
        self.shade = shade
        self.bin_power = bin_power
        self.saveto = saveto

    def find_dataset_file(self, run):
        """Find adc dataset file if it exists"""
        fname = f'{run}_'
        for power in self.powers:
            bit = 2**power
            fname += f'{bit}s_'
        fname += 'dataset.pkl'
        globstring = os.path.join(self.saveto, 'datasets', fname)
        filepath = glob.glob(globstring)
        return filepath

    def make_adcDatasets(self):
        """Create and save adc datasets"""
        for run in self.runs:
            dataset_list = self.find_dataset_file(run)
            if len(dataset_list) == 0:
                adc_dataset = adcDataset(run, self.powers, self.bin_power)
                adc_dataset.fill_dataset()
                adc_dataset.save(self.saveto)
            else:
                print(f'ADC file {dataset_list[0]} exists')
                continue

    def get_datasets(self):
        """Retrieve adc datasets for runs"""
        datasets = {}
        for run in self.runs:
            dataset_file = self.find_dataset_file(run)[0]
            with open(dataset_file, 'rb') as f:
                datasets[run] = pkl.load(f)
        return datasets

    def make_plots(self):
        """Make profile plots from adc datasets"""
        raftnames = get_raftnames_runs(self.runs)
        datasets = self.get_datasets()
        shade_bit = 2**self.shade
        bits = np.power(2, self.powers)
        prof_bin_size = 2**self.bin_power

        for raftname in raftnames:
            for SNN in SNNs:
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
                    for run in self.runs:
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

                pathname = os.path.join(self.saveto, 'plots', f'{raftname}')
                Path(pathname).mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.join(pathname, fname))
                plt.close()


def main(runs, powers, shade, bin_power, saveto):
    """ Make and plot adc datasets"""
    task = adcTask(runs, powers, shade, bin_power, saveto)
    task.make_adcDatasets()
    task.make_plots()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze sflat files for given runs at amplifier level')
    parser.add_argument('runs', nargs="+", help='A list of runs (as strings) to analyze')
    parser.add_argument('--powers', nargs='+', default=[0, 1], help='Bit powers to analyze')
    parser.add_argument('--shade', default=9, help='Bit power to shade for in plots')
    parser.add_argument('--binpower', default=4, help='bin size log base 2 in profiles')
    parser.add_argument('--saveto', default='/gpfs/slac/lsst/fs1/u/adriansh/data/analysis/adc_analysis/',
                        help='Where to save the results. '
                        'Default: \'/gpfs/slac/lsst/fs1/u/adriansh/data/analysis/adc_analysis/\'')
    args = parser.parse_args()

    runs = args.runs
    powers = args.powers
    shade = args.shade
    bin_power = args.binpower
    saveto = args.saveto

    main(runs, powers, shade, bin_power, saveto)
