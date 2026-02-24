import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from data import load_data

def random_search_rf_baseline(X, y, groups, n_iters=19, random_state=42):
    rng = np.random.default_rng(random_state)
    cv = LeaveOneGroupOut()

    # Same domains as your BO
    n_estimators_range = (1, 150)
    max_depth_range = (1, 100)
    max_features_choices = ["log2", "sqrt", None]   # use None directly
    criterion_choices = ["gini", "entropy"]

    auc_list = []
    params_list = []

    for i in range(n_iters):
        # sample random hyperparameters (uniform)
        n_estimators = int(rng.integers(n_estimators_range[0], n_estimators_range[1] + 1))
        max_depth = int(rng.integers(max_depth_range[0], max_depth_range[1] + 1))
        max_features = rng.choice(max_features_choices)
        criterion = rng.choice(criterion_choices)

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "criterion": criterion,
        }

        model = RandomForestClassifier(
            **params,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        probs = cross_val_predict(
            model, X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]

        auc = roc_auc_score(y, probs)

        auc_list.append(auc)
        params_list.append(params)

        print(f"Iter {i+1}/{n_iters} | params={params} | AUC={auc:.4f}")

    auc_arr = np.array(auc_list)
    mean_auc = float(np.mean(auc_arr))
    std_auc = float(np.std(auc_arr, ddof=1)) if len(auc_arr) > 1 else 0.0  # sample std
    best_idx = int(np.argmax(auc_arr))
    best_params = params_list[best_idx]
    best_auc = float(auc_arr[best_idx])

    summary = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "best_auc": best_auc,
        "best_params": best_params,
        "all_auc": auc_list,
        "all_params": params_list,
    }
    return summary

X, y, groups = load_data("egg_data_assignment_2", sample_rate=50, sample_rate_original=250)

# Random baseline (19 iterations)
baseline = random_search_rf_baseline(X, y, groups, n_iters=19, random_state=42)
print("\nRANDOM BASELINE SUMMARY")
print("Mean AUC:", baseline["mean_auc"])
print("Std AUC:", baseline["std_auc"])
print("Best AUC:", baseline["best_auc"])
print("Best params:", baseline["best_params"])