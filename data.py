# import os
# import numpy as np
# from os.path import join
# import pandas as pd
# from sklearn.discriminant_analysis import StandardScaler
# from sklearn.model_selection import train_test_split
# from tqdm.auto import tqdm


# #getting data
# def load_data(path_data):
    
#     data_oa = []
#     data_oc = []

#     for folder in tqdm(os.listdir(path_data), desc="Loading data"):
#         for filename in os.listdir(join(path_data, folder)):
#             df = pd.read_csv(join(path_data, folder, filename), sep=",", index_col=0)
#             if "oa" in filename:
#                 data_oa.append(np.array(df))
#             else:
#                 data_oc.append(np.array(df))

#     X = np.vstack([data_oa, data_oc])  

#     y = np.concatenate([
#     np.zeros(len(data_oa[0]), dtype=int),
#     np.ones(len(data_oc[0]), dtype=int)])
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test

import os
import numpy as np
import pandas as pd
from os.path import join
from sklearn.model_selection import train_test_split
import scipy.signal as signal


def reref_and_downsample(eeg, target_len=10_000):
    # eeg shape: (20, T)
    eeg_ds = signal.resample(eeg, target_len, axis=1)   # (20, target_len)
    ref = eeg_ds.mean(axis=0)                           # (target_len,)
    return eeg_ds - ref                                 # (20, target_len)


def load_data(path_data, test_size=0.2, random_state=42, target_len=10_000):
    X = []
    y = []

    for folder in os.listdir(path_data):
        for filename in os.listdir(join(path_data, folder)):
            if not filename.endswith(".csv"):
                continue

            df = pd.read_csv(join(path_data, folder, filename), index_col=0)
            eeg = df.values                              # (20, 50000) typically

            eeg = reref_and_downsample(eeg, target_len)   # (20, 10000)
            X.append(eeg.reshape(-1))                     # (20*10000,)

            y.append(0 if "oa" in filename.lower() else 1)

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)