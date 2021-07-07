import matplotlib.pyplot as plt
from utils.BOT_9raft_detmap import detectors_map

def plot_fp():
    borderless = [(0, 0), (0, 1), (0, 2), (0, 12), (0, 13), (0, 14),
                  (14, 0), (14, 1), (14, 2), (14, 12), (14, 13), (14, 14),
                  (1, 0), (1, 1), (1, 13), (1, 14), (13, 0), (13, 1),
                  (13, 13), (13, 14), (2, 0), (2, 14), (12, 0), (12, 14)]

    fig, axs = plt.subplots(15, 15, sharex=True, sharey=True, figsize=(24,24),
                            subplot_kw={'xticks': [], 'yticks': []},
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    
    for (t0, t1) in borderless:
        axs[t0, t1].set_frame_on(False)
    
    
    for i in range(5):
        axs[-1, 1+3*i].set_xlabel(f'RX{i}', fontsize='x-large')
        axs[-2-3*i, 0].set_ylabel(f'R{i}X', fontsize='x-large',
                                  rotation='horizontal', ha='right')
    
    return fig, axs

def map_detector_to_fp(detnum):
    raft = detectors_map[detnum]['raftName']
    sensor = detectors_map[detnum]['detectorName']
    ty = 3*int(raft[2]) + int(sensor[2])
    tx = 14 - 3*int(raft[1]) - int(sensor[1])
    tup = (tx, ty)

    return tup

def plot_raft():
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(24, 24))

    for i in range(3):
        axs[-1, i].set_xlabel(f'SX{i}', fontsize=18)
        axs[-1-i, 0].set_ylabel(f'S{i}X', fontsize=18,
                                rotation='horizontal', ha='right')

    return fig, axs

def map_detector_to_raft(detnum):
    sensor = detectors_map[detnum]['detectorName']
    ty = int(sensor[2])
    tx = 2 - int(sensor[1])
    tup = (tx, ty)

    return tup
