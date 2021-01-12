# destinationPath should correspond to DATA_DIR from bash_part.sh
destinationPath = '/home/adriansh/lsst_devel/analysis/ptc_comparison/simulated_pedestal/raw/'
day1 = '2020-02-19'
seqNums1 = list(range(41,129))
day2 = '2020-02-21'
seqNums2 = list(range(13,61))
day3 = '2020-03-13'
seqNums3 = list(range(13,29))
seqNums3.extend(range(57,97))
import lsst.daf.persistence as dafPersist
import shutil
import os
def copyFile(butler, dayObs, seqNum, destinationPath):
    filenameAndPath = butler.getUri('_raw', dayObs=dayObs, seqNum=seqNum)[:-3]
    _ = shutil.copy(filenameAndPath, destinationPath)

butler = dafPersist.Butler('/project/shared/auxTel/')
for seqNum in seqNums1:
    copyFile(butler, day1, seqNum, destinationPath)
for seqNum in seqNums2:
    copyFile(butler, day2, seqNum, destinationPath)
for seqNum in seqNums3:
    copyFile(butler, day3, seqNum, destinationPath)
