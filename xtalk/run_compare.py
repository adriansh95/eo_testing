import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import glob
from utils.BOT_9raft_detmap import detectors_map
from utils.profile import profile, make_profile
from scipy.stats import linregress

runs = ['6836D', '6834D']
run0 = runs[0]
run1 = runs[1]
noclip_path = 'analysis/BOT/xtalk_noclip/'
clip_path = 'analysis/BOT/xtalk/'

noclip_globstring0 = os.path.join(noclip_path, f'*{run0}*.pkl')
nc_files0 = sorted(glob.glob(noclip_globstring0))
clip_globstring0 = os.path.join(clip_path, f'*{run0}*.pkl')
c_files0 = sorted(glob.glob(clip_globstring0))

noclip_globstring1 = os.path.join(noclip_path, f'*{run1}*.pkl')
nc_files1 = sorted(glob.glob(noclip_globstring1))
clip_globstring1 = os.path.join(clip_path, f'*{run1}*.pkl')
c_files1 = sorted(glob.glob(clip_globstring1))

for nc_in0, c_in0, nc_in1, c_in1 in zip(nc_files0, c_files0, nc_files1, c_files1):
    stem = os.path.basename(c_in0).split('.')[0]
    detector = stem.split('_')[3][3:]
    raft = detectors_map[detector]['raftName']
    sensor = detectors_map[detector]['detectorName']

    with open(nc_in0, 'rb') as nc_f0, open(c_in0, 'rb') as c_f0, open(nc_in1, 'rb') as nc_f1, open(c_in1, 'rb') as c_f1:
        c_d0 = pkl.load(c_f0)
        c_d_fluxes0 = c_d0['fluxes']
        c_d_ratios0 = c_d0['ratios']

        nc_d0 = pkl.load(nc_f0)
        nc_d_fluxes0 = nc_d0['fluxes']
        nc_d_ratios0 = nc_d0['ratios']

        c_d1 = pkl.load(c_f1)
        c_d_fluxes1 = c_d1['fluxes']
        c_d_ratios1 = c_d1['ratios']

        nc_d1 = pkl.load(nc_f1)
        nc_d_fluxes1 = nc_d1['fluxes']
        nc_d_ratios1 = nc_d1['ratios']

        for target, c_source_d0 in c_d_fluxes0.items():
            nc_source_d0 = nc_d_fluxes0[target]
            c_source_d1 = c_d_fluxes1[target]
            nc_source_d1 = nc_d_fluxes1[target]
            for source, c_fluxes0 in c_source_d0.items():
                if target == source:
                    continue
                else:
                    nc_fluxes0 = nc_source_d0[source]
                    nc_fluxes1 = nc_source_d1[source]
                    c_fluxes1 = c_source_d1[source]

                    c_ratios0 = c_d_ratios0[target][source]
                    nc_ratios0 = nc_d_ratios0[target][source]
                    c_ratios1 = c_d_ratios1[target][source]
                    nc_ratios1 = nc_d_ratios1[target][source]

                    c_prof0 = make_profile(c_fluxes0, c_ratios0)
                    nc_prof0 = make_profile(nc_fluxes0, nc_ratios0)
                    c_prof1 = make_profile(c_fluxes1, c_ratios1)
                    nc_prof1 = make_profile(nc_fluxes1, nc_ratios1)

                    profiles = [c_prof0, nc_prof0, c_prof1, nc_prof1]
                    labels = [f'{run0} Clipped', f'{run0} Unclipped', f'{run1} Clipped', f'{run1} Unclipped']
                    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

                    fig, ax = plt.subplots(figsize=(24,18))

                    for prof, label, color in zip(profiles, labels, colors):
                        ax.errorbar(prof.xarr, prof.yarr, xerr=prof.xerr, yerr=prof.yerr, color=color, label=label)

                    ax.legend(loc='upper right', prop={'size': 16})
                    ax.set_xlabel('Flux', fontsize='x-large')
                    ax.set_ylabel('Ratio', fontsize='x-large')
                    ax.set_title(f'{run0} {run1} {raft} {sensor} {source}->{target} Xtalk Profiles', fontsize='xx-large')
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
                    ax.tick_params(axis='x', labelsize=12)
                    ax.tick_params(axis='y', labelsize=12)

                    saveto = os.path.join(clip_path, 'plots/run_compare/')
                    figname = f'{raft}_{sensor}_{target}_{source}_{run0}_{run1}_compare.png'
                    fullname = os.path.join(saveto, figname) 
                    fig.tight_layout()
                    fig.savefig(fullname)
                    print(f'Wrote {fullname}')
                    plt.close(fig)
