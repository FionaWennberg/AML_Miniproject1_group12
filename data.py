import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal

from os.path import join

import pandas as pd
from tqdm.notebook import tqdm


#getting data
path_data = "egg_data_assignment_2"
data_oa_original = []
data_oc_original = []
sample_rate_original = 250

#Seperating open and closed
for folder in tqdm(os.listdir(path_data), desc="Loading data"):
    for filename in os.listdir(join(path_data, folder)):
        df = pd.read_csv(join(path_data, folder, filename), sep=",", index_col=0)
        if "oa" in filename:
            data_oa_original.append(df.values)
        else:
            data_oc_original.append(df.values)

