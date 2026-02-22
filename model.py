import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score

from data import load_data

X, y, groups = load_data("egg_data_assignment_2", sample_rate=50, sample_rate_original=250)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

cv = LeaveOneGroupOut()
probs = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
preds = (probs >= 0.5).astype(int)

print(classification_report(y, preds))
print("ROC-AUC:", roc_auc_score(y, probs))