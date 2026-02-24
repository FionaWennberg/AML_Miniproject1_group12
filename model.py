import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

from data import load_data
from bo import run_bo_rf

print("MODEL.PY STARTER")

X, y, groups = load_data("egg_data_assignment_2", sample_rate=50, sample_rate_original=250)

best_params, best_auc, opt = run_bo_rf(X, y, groups, n_calls=19)
print("Best params:", best_params)
print("Best AUC during BO:", best_auc)

# 1) Observed AUC values for each evaluated point
auc_values = [-v for v in opt.func_vals]  # because objective returns -AUC

print("\nBO evaluated AUC summary:")
print("Mean AUC:", float(np.mean(auc_values)))
print("Std  AUC:", float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else 0.0)
print("Best AUC:", float(np.max(auc_values)))

# 2) (Optional) GP surrogate mean/std at evaluated points (NOT the same as above)
gp = opt.models[-1]  # trained GP surrogate
x_all = opt.x_iters
x_all_transformed = opt.space.transform(x_all)
gp_mean, gp_std = gp.predict(x_all_transformed, return_std=True)

print("\nGP surrogate at evaluated points (mean/std of objective, not CV uncertainty):")
for x, m, s in zip(x_all, gp_mean, gp_std):
    print("Params:", x)
    print("GP mean objective:", float(m), " -> approx AUC:", float(-m))
    print("GP std:", float(s))
    print()

# 3) Convergence plot (observed AUC per iteration)
plt.figure()
plt.plot(auc_values, marker="o")
plt.xlabel("Iteration")
plt.ylabel("AUC")
plt.title("BO Convergence (AUC per iteration)")
plt.show()

# 4) Train/evaluate using best params with the same LOGO CV protocol
model = RandomForestClassifier(
    **best_params,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

cv = LeaveOneGroupOut()
probs = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
preds = (probs >= 0.5).astype(int)

print("\nFinal evaluation using best params:")
print(classification_report(y, preds))
print("ROC-AUC:", roc_auc_score(y, probs))