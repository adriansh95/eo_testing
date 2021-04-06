import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import glob
import math
from utils.BOT_9raft_detmap import detectors_map
from utils.profile import profile, make_profile
from scipy.stats import linregress

runs = ['6833D', '6834D']
noclip_path = 'analysis/BOT/xtalk_noclip/'
clip_path = 'analysis/BOT/xtalk/'

for run in runs:
    noclip_globstring = os.path.join(noclip_path, f'*{run}*.pkl')
    noclip_files = sorted(glob.glob(noclip_globstring))
    clip_globstring = os.path.join(clip_path, f'*{run}*.pkl')
    clip_files = sorted(glob.glob(clip_globstring))


    for nc_infile, c_infile in zip(noclip_files, clip_files):
        stem = os.path.basename(c_infile).split('.')[0]
        detector = stem.split('_')[3][3:]
        raft = detectors_map[detector]['raftName']
        sensor = detectors_map[detector]['detectorName']

        with open(nc_infile, 'rb') as nc_f, open(c_infile, 'rb') as c_f:
            c_d = pkl.load(c_f)
            c_d_fluxes = c_d['fluxes']
            c_d_ratios = c_d['ratios']
            nc_d = pkl.load(nc_f)
            nc_d_fluxes = nc_d['fluxes']
            nc_d_ratios = nc_d['ratios']

            for target, c_source_d in c_d_fluxes.items():
                nc_source_d = nc_d_fluxes[target]
                for source, c_fluxes in c_source_d.items():
                    nc_fluxes = nc_source_d[source]
                    if target == source:
                        continue
                    else:
                        c_ratios = c_d_ratios[target][source]
                        nc_ratios = nc_d_ratios[target][source]
                        c_prof = make_profile(c_fluxes, c_ratios)
                        nc_prof = make_profile(nc_fluxes, nc_ratios)

                        fig, ax = plt.subplots(figsize=(24,18))
                        ax.errorbar(c_prof.xarr, c_prof.yarr, xerr=c_prof.xerr, yerr=c_prof.yerr,
                                    color='tab:blue', label='Clipped')
                        ax.errorbar(nc_prof.xarr, nc_prof.yarr, xerr=nc_prof.xerr, yerr=nc_prof.yerr,
                                    color='tab:orange', label='Unclipped')
                        ax.legend(loc='upper right', prop={'size': 16})
                        ax.set_xlabel('Flux')
                        ax.set_ylabel('Ratio')
                        ax.set_title(f'{run} {raft} {sensor} {source}->{target} Xtalk Profiles')
                        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

                        saveto = os.path.join(clip_path, 'plots/clipping_compare/')
                        figname = f'{run}_{raft}_{sensor}_{target}_{source}_clipping_compare.png'
                        fullname = os.path.join(saveto, figname) 
                        fig.tight_layout()
                        fig.savefig(fullname)
                        plt.close(fig)
                        print(f'Wrote {fullname}')
