import astropy.io.fits as pft
import numpy as np
from astropy.time import Time


class Transmuter():
    def __init__(self, filename, counts_per_second=40000, overwrite=False, add_read_noise=False,
                overscan_pedestal=1000, read_noise_mean=10):
        self._NUM_AMPS = 16
        self._IMAGE_ROWS = 4000
        self._IMAGE_COLS = 4072
        self._IMAGE_COLS_PER_AMP = 509
        self.counts_per_second = counts_per_second
        self.add_read_noise = add_read_noise
        self.overscan_pedestal = overscan_pedestal
        self.read_noise_mean = read_noise_mean

        self.fits_file = pft.open(filename, mode='update')
        self.exptime = self.fits_file[0].header['EXPTIME']
        for amp in self.fits_file[1:]:
            self.transmute_amp(amp)
        if overwrite:
            self.fits_file.flush()

    def transmute_amp(self, amp):
        # Figure out what section of the detector is data
        data_sec = amp.header['DATASEC']
        data_sec = data_sec.strip('[]').replace(',', ':').split(':')
        data_col_start = np.int(data_sec[0]) - 1  #indexing is different
        data_col_end = np.int(data_sec[1])
        data_row_start = np.int(data_sec[2]) - 1
        data_row_end = np.int(data_sec[3])
    
        det_sec = amp.header['DETSEC']
        det_sec = det_sec.strip('[]').replace(',', ':').split(':')
        det_col_start = np.int(det_sec[0]) - 1  #indexing is different
        det_col_end = np.int(det_sec[1])
        det_row_start = np.int(det_sec[2]) - 1
        det_row_end = np.int(det_sec[3])
    
        if det_col_start > det_col_end:
            col_stride = -1
            det_col_end -= 1
            if det_col_end == 0:
                det_col_end = None
            else:
                det_col_end -= 1
        else:
            col_stride = 1
    
        if det_row_start > det_row_end:
            row_stride = -1
            det_row_end -= 1
            if det_row_end == 0:
                det_row_end = None
            else:
                det_row_end -= 1
        else:
            row_stride = 1
        
        mask = np.zeros_like(amp.data).astype(np.bool)
        mask[data_row_start:data_row_end, data_col_start:data_col_end] = True
        poisson_rvs = np.random.poisson(lam=self.counts_per_second*self.exptime, size=mask.shape)
        amp.data[:,:] = poisson_rvs
        amp.data[~mask] = 0.
        if self.add_read_noise:
            read_noise = np.random.poisson(lam=self.read_noise_mean, size=mask.shape) + self.overscan_pedestal
            amp.data[:,:] += read_noise

    
        #amp.data[data_row_start:data_row_end, data_col_end:] = 0.
        return
