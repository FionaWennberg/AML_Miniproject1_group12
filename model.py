import os
from time import time
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
from os.path import join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from data import load_data
    
X_train, X_test, y_train, y_test = load_data("egg_data_assignment_2")

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
