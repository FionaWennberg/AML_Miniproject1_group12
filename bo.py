# Bayesian optimization for hyperparameter optimization of a Random Forest Classifier
# rf_bo.py
import numpy as np
from skopt import gp_minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score


def run_bo_rf(X, y, groups, n_calls=19, random_state=42):
    # start at same initial point (your style)
    x0 = [50, 25, "sqrt", "gini"]
    if x0[2] is None:
        x0[2] = "None"

    # define the domain of the considered parameters (your style)
    n_estimators = (1, 150)
    max_depth = (1, 100)
    max_features = ("log2", "sqrt", "None")
    criterion = ("gini", "entropy")

    cv = LeaveOneGroupOut()

    global i
    i = 1

    def objective_function(x):
        # map your "None" string to actual None
        if x[2] == "None":
            maxf = None
        else:
            maxf = x[2]

        model = RandomForestClassifier(
            n_estimators=x[0],
            max_depth=x[1],
            max_features=maxf,
            criterion=x[3],
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        # Use the SAME evaluation protocol as in your model.py
        probs = cross_val_predict(
            model, X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]

        auc = roc_auc_score(y, probs)

        global i
        i += 1
        print(i)
        print(x)
        print("AUC:", auc)

        # gp_minimize minimizes
        return -auc

    # IMPORTANT: compute y0 as objective(x0)
    y0 = objective_function(x0)

    # If you hit the np.int deprecation in your environment, keep this workaround:
    np.int = int

    opt = gp_minimize(
        objective_function,
        [n_estimators, max_depth, max_features, criterion],
        acq_func="EI",
        n_initial_points=0,     # keep your setting: only x0 as the initial point
        n_calls=n_calls,
        x0=[x0],
        y0=[y0],
        xi=0.1,
        noise=0.01**2,
        random_state=random_state,
    )

    best_x = opt.x
    best_params = {
        "n_estimators": best_x[0],
        "max_depth": best_x[1],
        "max_features": None if best_x[2] == "None" else best_x[2],
        "criterion": best_x[3],
    }

    best_auc = -opt.fun
    return best_params, best_auc, opt