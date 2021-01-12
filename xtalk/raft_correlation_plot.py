from xtalk_nonlinearityTask import xtalkDataset
from utils.plot_fp import plot_fp, map_detector_to_tup
from utils.BOT_9raft_detmap import detectors_map
from utils.defaults import all_amps
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import numpy as np

fig, axs = plot_fp()
amps = all_amps

dataset_files = glob.glob('analysis/BOT/xtalk/datasets/*xtalk_dataset.pkl')
bounds = [-1e-2, -1e-3, -1e-4, -1e-5, -1e-6, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] 

for dataset_file in dataset_files:
    vals = np.full((16, 16), np.nan)
    
    with open(dataset_file, 'rb') as f:
        ds = pkl.load(f)
        summary = ds.summary
        detector = ds.detector
    
        for target, sdict in summary.items():
            i = amps.index(target)
            for source, vdict in sdict.items():
                j = amps.index(source)
                norm = vdict['norm']
                vals[i, j] = norm

    ax_idx = map_detector_to_tup(detector)
    ax = axs[ax_idx]

    im = ax.imshow(vals, cmap="seismic",# vmin=-1e-3, vmax=1e-3,
                   norm=colors.BoundaryNorm(boundaries=bounds, ncolors=256))
                   #norm=colors.SymLogNorm(linthresh=1e-6))
    #ax.set_xticks(np.arange(16))
    #ax.set_yticks(np.arange(16))
    #ax.set_xticklabels(amps)
    #ax.set_yticklabels(amps)
    #ax.set_xlabel('Target', fontsize='x-large')
    #ax.set_ylabel('Source', fontsize='x-large')
    
    #cbar = ax.figure.colorbar(im, ax=ax, format='%.2e', ticks=bounds)
    #cbar.ax.set_ylabel('xtalk', rotation=-90, va="bottom", fontsize='x-large')
    
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             #rotation_mode="anchor")
    
#fig.colorbar(im, ax=axs.ravel().tolist(), format='%.2e', ticks=bounds)
#fig.tight_layout()
fig.subplots_adjust(right=0.9, top=0.9)
cbar_ax = fig.add_axes([0.93, 0.2, 0.025, 0.6])
fig.colorbar(im, cax=cbar_ax, format='%.2e', ticks=bounds)
fig.suptitle('Low Flux Xtalk Ratio', fontsize='xx-large')
plt.savefig('fp_heatmap_test.png')
