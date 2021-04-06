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
path = 'analysis/BOT/xtalk_noclip/'

globstring0 = os.path.join(path, f'*{run0}*.pkl')
files0 = sorted(glob.glob(globstring0))

globstring1 = os.path.join(path, f'*{run1}*.pkl')
files1 = sorted(glob.glob(globstring1))

for in0, in1 in zip(files0, files1):
    stem = os.path.basename(in0).split('.')[0]
    detector = stem.split('_')[3][3:]
    raft = detectors_map[detector]['raftName']
    sensor = detectors_map[detector]['detectorName']

    with open(in0, 'rb') as f0, open(in1, 'rb') as f1:
        d0 = pkl.load(f0)
        d_fluxes0 = d0['fluxes']
        d_ratios0 = d0['ratios']

        d1 = pkl.load(f1)
        d_fluxes1 = d1['fluxes']
        d_ratios1 = d1['ratios']

        for target, source_d0 in d_fluxes0.items():
            source_d1 = d_fluxes1[target]
            for source, fluxes0 in source_d0.items():
                if target == source:
                    continue
                else:
                    fluxes1 = source_d1[source]

                    ratios0 = d_ratios0[target][source]
                    ratios1 = d_ratios1[target][source]

                    targets0 = np.array(ratios0)*np.array(fluxes0)
                    targets1 = np.array(ratios1)*np.array(fluxes1)

                    linr0 = linregress(fluxes0, targets0)
                    linr1 = linregress(fluxes1, targets1)

                    targets0_intsubbed = targets0 - linr0.intercept
                    targets1_intsubbed = targets1 - linr1.intercept

                    ratios0_intsubbed = targets0_intsubbed/fluxes0
                    ratios1_intsubbed = targets1_intsubbed/fluxes1

                    prof0 = make_profile(fluxes0, ratios0)
                    prof1 = make_profile(fluxes1, ratios1)
                    prof0_intsubbed = make_profile(fluxes0, ratios0_intsubbed)
                    prof1_intsubbed = make_profile(fluxes1, ratios1_intsubbed)


                    profiles = [prof0, prof0_intsubbed, prof1, prof1_intsubbed]
                    labels = [f'{run0}', f'{run0} intercept subtracted', f'{run1}', f'{run1} intercept subtracted']
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
                    ax.grid(b=True)

                    saveto = os.path.join(path, 'plots/run_compare/')
                    figname = f'{raft}_{sensor}_{target}_{source}_{run0}_{run1}_bgsub_compare.png'
                    fullname = os.path.join(saveto, figname) 
                    fig.tight_layout()
                    fig.savefig(fullname)
                    print(f'Wrote {fullname}')
                    plt.close(fig)
