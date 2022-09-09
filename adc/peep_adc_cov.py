import numpy as np
import pandas as pd
import pickle as pkl
from eo_testing.adc.adcCovTask import adcCovDataset

ds_file = 'analysis/adc/datasets/dnl_cov/R22_S01_dnl_cov_dataset.pkl'

with open(ds_file, 'rb') as f:
    ds = pkl.load(f)
    for k, v in ds.covarianceMatrices.items():
        print(k)
        print(v)
