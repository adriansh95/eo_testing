import numpy as np

def get_true_windows(t):
    windows = []
    window = []
    for ib, b in enumerate(t):
        if b:
            window.append(ib)
        elif len(window) > 0 and not b:
            window = np.array(window)
            windows.append(window)
            window = []
            continue
        else:
            continue

    return windows
