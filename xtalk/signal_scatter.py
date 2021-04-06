import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import glob
from utils.BOT_9raft_detmap import detectors_map
from scipy.stats import linregress
from utils.profile import profile, make_profile

runs = ['6833D', '6834D', '6836D']
basepath = 'analysis/BOT/xtalk_noclip/'

for run in runs:
    globstring = os.path.join(basepath, f'*{run}*.pkl')
    files = sorted(glob.glob(globstring))

    for infile in files:
        stem = os.path.basename(infile).split('.')[0]
        detector = stem.split('_')[3][3:]
        raft = detectors_map[detector]['raftName']
        sensor = detectors_map[detector]['detectorName']

        with open(infile, 'rb') as f:
            d = pkl.load(f)
            fluxes_dict = d['fluxes']
            ratios_dict = d['ratios']
            for target, source_dict in fluxes_dict.items():
                for source, fluxes in source_dict.items():
                    if target == source:
                        continue
                    else:
                        ratios = ratios_dict[target][source]
                        fig, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True, figsize=(24,18))
                        axs[0].scatter(fluxes, ratios, s=10)
                        #axs[0].set_xlabel('Fluxes')
                        axs[0].set_ylabel('Ratios', fontsize='x-large')
                        axs[0].set_title(f'Xtalk Ratios vs Fluxes', fontsize='x-large')

                        targets = np.array(ratios) * np.array(fluxes)
                        linr = linregress(fluxes, targets)
                        linx = np.linspace(min(fluxes), max(fluxes), 10000)
                        liny = linr.slope*linx + linr.intercept
                        axs[1].scatter(fluxes, targets, s=10)
                        axs[1].plot(linx, liny, color='tab:orange', label=f'{linr.slope:.2e}*x + {linr.intercept:.2f}; '
                                   f'r^2={linr.rvalue**2:.2f}')
                        axs[1].legend(loc='lower right', prop={'size': 18})
                        #axs[1].set_xlabel('Primary Signals')
                        axs[1].set_ylabel('Secondary Signals', fontsize='x-large')
                        axs[1].set_title(f'Secondary vs Primary Signals', fontsize='x-large')

                        ratios_int_subtracted = (targets - linr.intercept) / fluxes
                        axs[2].scatter(fluxes, ratios_int_subtracted, s=10)
                        axs[2].set_ylabel('Ratios', fontsize='x-large')
                        axs[2].set_title(f'Xtalk Ratios (Intercept Subtracted) vs Fluxes', fontsize='x-large')

                        r_prof = make_profile(fluxes, ratios)
                        ris_prof = make_profile(fluxes, ratios_int_subtracted)
                        profiles = [r_prof, ris_prof]
                        labels = ['Unaltered', 'Intercept Subtracted']
                        colors = ['tab:blue', 'tab:orange']

                        for prof, label, color in zip(profiles, labels, colors):
                            axs[3].errorbar(prof.xarr, prof.yarr, xerr=prof.xerr, yerr=prof.yerr,
                                           color=color, label=label)

                        axs[3].set_xlabel('Fluxes', fontsize='x-large')
                        axs[3].set_ylabel('Ratios', fontsize='x-large')
                        axs[3].set_title('Xtalk Profiles', fontsize='x-large')
                        axs[3].legend(loc='upper right', prop={'size': 16})

                        for i in [0, 2, 3]:
                            axs[i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

                        for ax in axs:
                            ax.grid(b=True)

                        fig.suptitle(f'{run} {raft} {sensor} {source}->{target}', fontsize='xx-large')
                        saveto = os.path.join(basepath, 'plots/scatters/')
                        figname = f'{run}_{raft}_{sensor}_{target}_{source}_xtalk_scatters.png'
                        fullname = os.path.join(saveto, figname) 
                        #fig.tight_layout()
                        fig.savefig(fullname)
                        print(f'Wrote {fullname}')
                        plt.close(fig)
