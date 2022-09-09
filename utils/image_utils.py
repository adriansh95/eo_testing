import matplotlib.pyplot as plt
import numpy as np
import lsst.afw.display as afwDisplay
from .defaults import all_amps

def plot_image(exp, ampName, expId, show_overscans=False):
    if ampName not in all_amps:
        raise KeyError(f'{ampName} is not a valid amp name')

    fig = plt.figure(figsize=(24, 18))
    ampnum = all_amps.index(ampName)
    det = exp.getDetector()
    detName = det.getName()
    amp = det[ampnum]

    if show_overscans:
        ampExp = exp[amp.getRawBBox()]
    else:
        ampExp = exp[amp.getRawDataBBox()]

    afwDisplay.setDefaultBackend('matplotlib') 
    afw_display = afwDisplay.Display(1)
    afw_display.scale('asinh', 'zscale')
    afw_display.mtv(ampExp.getImage())
    plt.title(f'{detName} {ampName} ({expId})', fontsize=24)
    plt.gca().axis('off')
    fig.tight_layout()

    plt.show()
