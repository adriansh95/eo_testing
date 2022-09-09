import argparse
import pickle as pkl
import glob
import os
import numpy as np
import pandas as pd
import copy

from collections import defaultdict
from lsst.daf.butler import Butler
from eo_testing.utils.defaults import all_amps, amp_plot_order
from adcTask import adc, adc_dataset, adcTask, adcTaskConfig


class adcCovDataset():
    def __init__(self, detNum, detName, detType, cov_mats):
        self.detNum = detNum
        self.detName = detName
        self.detType = detType
        self.covarianceMatrices = cov_mats

class adcCovTask():
    def __init__(self, config):
        self.config = config

    def run(self, adcTask_config):
        adc_task = adcTask(adcTask_config)

        runstr = "('" + "', '".join(self.config.runs) + "')"
        where = adc_task.where + f"and exposure.science_program in {runstr}"

        det_raft_pairs = adc_task.get_det_raft_pairs(where)

        amps_list = self.config.amps

        for detector, detName in det_raft_pairs:
            x = np.arange(0, 2**18)
            dnl_measurements = {amp: [] for amp in amps_list}

            for run in self.config.runs:
                new_where = adc_task.where + f"and exposure.science_program = '{run}'"
                detType = ''
                adcs = {}

                if adc_task.check_dataset_exists(detName, run):
                    dataset_file = adc_task.get_det_dataset_file(detName, run)

                    with open(dataset_file, 'rb') as f:
                        dataset = pkl.load(f)

                        try:
                            adcs = dataset.adcs
                            detType = dataset.detType
                        except AttributeError:
                            pass
                else:
                    pass

                missing_amps = list(set(amps_list).difference(set(adcs.keys())))

                if len(missing_amps) > 0 or detType == 0:
                    dataId = {'detector': detector}
                    datarefs = list(adc_task.butler.registry.queryDatasets(datasetType='raw', 
                                    collections=adc_task.collections, where=new_where, dataId=dataId))

                    missing_adcs, detType = adc_task.make_adcs_dict(datarefs, missing_amps)

                for amp, adc in missing_adcs.items():
                    adcs[amp] = adc

                if adcs == {}:
                    continue

                if self.config.make_datasets:
                    dataset = adc_task.initialize_dataset(detName, detType, adcs)
                    adc_task.write_dataset(dataset, run)

                for amp, adc in adcs.items():
                    y = np.nan*x
                    y[adc.measured_bins] = adc.dnl
                    sparse_series = pd.Series(pd.arrays.SparseArray(y), name=run)
                    dnl_measurements[amp].append(sparse_series)
            
            dataframes = {amp: pd.concat(series_list, axis=1) 
                          for amp, series_list in dnl_measurements.items()
                          if len(series_list) > 1}

            cov_mats = {amp: df.cov() for amp, df in dataframes.items()}

            if cov_mats == {}:
                continue
            else:
                cov_ds = adcCovDataset(detector, detName, detType, cov_mats)
                self.write_cov_dataset(cov_ds)

    def write_cov_dataset(self, dataset):
        os.makedirs(self.config.dataset_loc, exist_ok=True)
        pkl_name = f'{dataset.detName}_{self.config.ds_type}_dataset.pkl'
        pkl_file_name = os.path.join(self.config.dataset_loc, pkl_name)

        with open(pkl_file_name, 'wb') as pkl_file:
            pkl.dump(dataset, pkl_file)
        print(f'Wrote {pkl_file_name}')
        
def main(runs, **kwargs):
        adcTask_config = adcTaskConfig(runs, **kwargs)
        kwargs['ds_type'] = 'dnl_cov'
        adcCov_task_config = adcTaskConfig(runs, **kwargs)
        adcCov_task = adcCovTask(adcCov_task_config)
        adcCov_task.run(adcTask_config)

if __name__ == "__main__":
    default_write_to = os.path.join(os.path.expanduser('~'), 'analysis/adc/')

    parser = argparse.ArgumentParser(description='Make covariance matrix for DNL measurements accross runs')
    parser.add_argument('runs', nargs="+", help='A list of runs to analyze')

    parser.add_argument('--write_to', default=default_write_to,
                        help='Where to save the results. '\
                        f'Default: {default_write_to}')
    parser.add_argument('--repo', default='/sdf/group/lsst/camera/IandT/repo_gen3/BOT_data/butler.yaml',
                        help='Where to look for data. Default: '\
                        '/sdf/group/lsst/camera/IandT/repo_gen3/BOT_data/butler.yaml')

    parser.add_argument('--detectors', nargs='+', default='ALL_DETECTORS', help='List of detectors. '
                        'Default: ALL_DETECTORS')
    parser.add_argument('--amps', nargs='+', default=all_amps, help=f'List of amps. '
                        "Example: 'C17' 'C10'. Default: {all_amps}")
    parser.add_argument('--instrument', default='LSSTCam', help='LSSTCam or LSST-TS8. Default: '\
                        'LSSTCam.')

    kwargs = vars(parser.parse_args())
    runs = kwargs.pop('runs')
    main(runs, **kwargs)
