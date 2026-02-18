import os
import numpy as np
from os.path import join
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


#getting data
def load_data(path_data):
    
    data_oa = np.array([])
    label_oa = np.array([])
    data_oc = np.array([])
    label_oc = np.array([])
    sample_rate_original = 250

    #Seperating open and closed
    for folder in tqdm(os.listdir(path_data), desc="Loading data"):
        for filename in os.listdir(join(path_data, folder)):
            df = pd.read_csv(join(path_data, folder, filename), sep=",", index_col=0)
            if "oa" in filename:
                data_oa = np.append(data_oa, df.values)
                label_oa = np.append(label_oa, 0)
            else:
                data_oc = np.append(data_oc, df.values)
                label_oc = np.append(label_oc, 1)

    X = np.concatenate((data_oa, data_oc), axis=0)
    y = np.concatenate((label_oa, label_oc), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
