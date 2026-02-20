import os
import numpy as np
from os.path import join
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm


#getting data
def load_data(path_data):
    
    data_oa = []
    data_oc = []

    for folder in tqdm(os.listdir(path_data), desc="Loading data"):
        for filename in os.listdir(join(path_data, folder)):
            df = pd.read_csv(join(path_data, folder, filename), sep=",", index_col=0)
            if "oa" in filename:
                data_oa.append(np.array(df))
            else:
                data_oc.append(np.array(df))

    X = np.vstack([data_oa, data_oc])  

    y = np.concatenate([
    np.zeros(len(data_oa[0]), dtype=int),
    np.ones(len(data_oc[0]), dtype=int)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
