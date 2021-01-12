import numpy as np
import transmute_image_poisson as tr
import glob


repo = '/home/adriansh/lsst_devel/analysis/ptc_comparison/simulated_pedestal/raw/'
days = ['2020-02-19/', '2020-02-21/', '2020-03-13/']
files = []
for day in days:
    fl = glob.glob(repo + day + '*.fits')
    files += fl

files = sorted(files)
print('Found {} files to transmute'.format(len(files)))

# set random seed, so if I make a mistake, we get the same data back
np.random.RandomState(254)

for f in files:
    print('Working on file {}'.format(f))
    tr.Transmuter(f, overwrite=True, add_read_noise=True, read_noise_mean=20.)
