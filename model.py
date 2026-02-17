import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal

from os.path import join

import pandas as pd
from sklearn.model_selection import ParameterSampler
from tqdm.notebook import tqdm

#model Random forrest 
domain = {
    'n_estimators': [50,100,150],
    'criterion': ['gini', 'entropy'],
    'max_depth': [25,75,100],
    'max_features': ['sqrt','log2']
}

param_list = list(ParameterSampler(domain, n_iter=20, random_state=32))
print('Param list')
print(param_list)
    
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, probs))
