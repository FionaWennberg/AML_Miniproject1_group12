import os
import numpy as np
import pandas as pd
from os.path import join
import scipy.signal as signal


def set_reference_and_downsample(eeg_data, ref_channels, sample_rate=50, sample_rate_original=250):
    """
    eeg_data: (20, T_original)
    Downsample from sample_rate_original -> sample_rate, and re-reference by common average over ref_channels.
    Returns: (len(ref_channels), T_down)
    """
    eeg_masked = eeg_data[ref_channels]  # (20, T)

    # T_down = T_original / (250/50) = T_original / 5
    factor = sample_rate_original // sample_rate
    eeg_down = signal.resample(eeg_masked, eeg_masked.shape[1] // factor, axis=1)

    ref_signal = eeg_down.mean(axis=0)           # (T_down,)
    eeg_reref = eeg_down - ref_signal            # (20, T_down)
    return eeg_reref


def load_data(path_data, sample_rate=50, sample_rate_original=250):
    data_oa_original = []
    data_oc_original = []
    groups_oa = []
    groups_oc = []

    # Treat each folder as one patient (8 folders -> 8 patient IDs)
    patient_folders = sorted([f for f in os.listdir(path_data) if os.path.isdir(join(path_data, f))])

    for pid, folder in enumerate(patient_folders):
        folder_path = join(path_data, folder)

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".csv"):
                continue

            df = pd.read_csv(join(folder_path, filename), sep=",", index_col=0)
            arr = df.values  # expected (20, 50000)

            if "oa" in filename.lower():
                data_oa_original.append(arr)
                groups_oa.append(pid)
            else:
                data_oc_original.append(arr)
                groups_oc.append(pid)

    # Preprocess (downsample + reref)
    data_oa = []
    data_oc = []
    for i in range(len(data_oa_original)):
        data_oa.append(set_reference_and_downsample(
            data_oa_original[i], np.arange(20),
            sample_rate=sample_rate, sample_rate_original=sample_rate_original
        ))
    for i in range(len(data_oc_original)):
        data_oc.append(set_reference_and_downsample(
            data_oc_original[i], np.arange(20),
            sample_rate=sample_rate, sample_rate_original=sample_rate_original
        ))

    # Flatten each recording to 1D feature vector for sklearn: (20, 10000) -> (200000,)
    X_oa = np.array([x.reshape(-1) for x in data_oa])
    X_oc = np.array([x.reshape(-1) for x in data_oc])

    # Build full dataset
    X = np.vstack([X_oa, X_oc])
    y = np.concatenate([
        np.zeros(len(X_oa), dtype=int),
        np.ones(len(X_oc), dtype=int),
    ])

    groups = np.array(groups_oa + groups_oc, dtype=int)  # patient ID per sample

    return X, y, groups