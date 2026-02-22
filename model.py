import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.decomposition import PCA
from bo import run_bo_rf

from data import load_data

X, y, groups = load_data("egg_data_assignment_2", sample_rate=50, sample_rate_original=250)

best_params, best_auc, opt = run_bo_rf(X, y, groups, n_calls=19)
print("Best params:", best_params)
print("Best AUC during BO:", best_auc)
model = RandomForestClassifier(
    **best_params,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)
cv = LeaveOneGroupOut()
probs = cross_val_predict(model, X_reduced, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
preds = (probs >= 0.5).astype(int)

print(classification_report(y, preds))
print("ROC-AUC:", roc_auc_score(y, probs))