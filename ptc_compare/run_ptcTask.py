from lsst.eotest.sensor.ptcTask import PtcTask
import glob
import os.path

repo = '/home/adriansh/lsst_devel/analysis/ptc_comparison/simulated_pedestal/raw/'
days = ['2020-02-19/', '2020-02-21/', '2020-03-13/']
filestr = '*det000.fits'
sensor_id = 'det000'
mask_files = []
gains = None

l = []
for day in days:
    globstr = os.path.join(repo, day, filestr)
    files = glob.glob(globstr)
    files.sort()
    l.append(files)

infiles = [infile for files_list in l for infile in files_list]

task = PtcTask()
task.run(sensor_id, infiles, mask_files, gains)
