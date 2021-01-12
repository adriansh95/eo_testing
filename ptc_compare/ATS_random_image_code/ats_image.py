import astropy.io.fits as pft
import numpy as np
from astropy.time import Time


class ATSImage():
    def __init__(self, filename, apply_overscan=True, apply_gain=False):
        self._NUM_AMPS = 16
        self._IMAGE_ROWS = 4000
        self._IMAGE_COLS = 4072
        self._IMAGE_COLS_PER_AMP = 509
        self._apply_overscan = apply_overscan
        self._apply_gain = apply_gain

        self.fits_file = pft.open(filename)
        self.image = np.zeros([self._IMAGE_ROWS, self._IMAGE_COLS])*np.nan
        self.amp_masks = {}
        self.amp_images = {}
        self.amp_overscan_means = {}
        self.amp_overscan_images = {}

        for amp in self.fits_file[1:]:
            self.load_amp(amp)

        self.exptime = self.fits_file[0].header['EXPTIME']
        self.imagetype = self.fits_file[0].header['IMGTYPE']
        self.amp_names = list(self.amp_images.keys())
        self.dateobs = Time(self.fits_file[0].header['DATE-OBS'], format='isot')


    def load_amp(self, amp):
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
        
        data = amp.data
        sci = data[data_row_start:data_row_end, data_col_start:data_col_end]
    
        overscan_vector = np.mean(data[data_row_start:data_row_end, data_col_end:], axis=1)
        overscan_array = np.repeat(overscan_vector[:, np.newaxis], sci.shape[-1], axis=1)
    
        if self._apply_overscan:
            sci = sci - overscan_array
    
        self.image[det_row_start:det_row_end:row_stride, det_col_start:det_col_end:col_stride] = sci
        amp_mask = np.zeros_like(self.image).astype(np.bool)
        amp_mask[det_row_start:det_row_end:row_stride, det_col_start:det_col_end:col_stride] = True
        
        amp_name = amp.header['EXTNAME']
        self.amp_masks[amp_name] = amp_mask
        self.amp_images[amp_name] = sci
        self.amp_overscan_means[amp_name] = np.mean(overscan_vector.flatten())
        self.amp_overscan_images[amp_name] = data[data_row_start:data_row_end, data_col_end:]

        return
