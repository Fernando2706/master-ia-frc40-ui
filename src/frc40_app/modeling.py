from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .config import CHEMICAL_TARGETS

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def regression_candidates(random_state: int) -> dict[str, tuple[object, dict]]:
    """Return a dictionary of candidate models and their hyperparameter grids.

    The set covers four families, with grids sized so a full GridSearchCV
    over the candidates remains tractable on a 365-row dataset:

    * **Linear baselines** (Ridge, Lasso, ElasticNet, BayesianRidge) - cheap
      and serve as a sanity check for the problem.
    * **Tree ensembles** (Extra Trees, Random Forest, Gradient Boosting,
      Hist Gradient Boosting) - typically the strongest performers.
    * **Instance-based** (KNN) - non-parametric comparison.
    * **Other** (SVR, AdaBoost, Bagging) - diverse methods for completeness.
    * **Stacking** - meta-ensemble that combines several of the above.

    The `XGBoost` candidate is added when the optional dependency is
    installed in the environment.
    """
    # Pre-build a Ridge pipeline for KNN-like models that need scaling.
    ridge_for_meta = Ridge(random_state=random_state)
    base_tree = DecisionTreeRegressor(random_state=random_state, max_depth=4)

    candidates: dict[str, tuple[object, dict]] = {
        # ---- Linear baselines (fast, good for sanity checking) ---------
        "Ridge": (
            make_pipeline(StandardScaler(), Ridge(random_state=random_state)),
            {
                "ridge__alpha": [0.1, 1.0, 10.0, 100.0],
            },
        ),
        "Lasso": (
            make_pipeline(StandardScaler(), Lasso(random_state=random_state, max_iter=10000)),
            {
                "lasso__alpha": [0.001, 0.01, 0.1, 1.0],
            },
        ),
        "ElasticNet": (
            make_pipeline(StandardScaler(), ElasticNet(random_state=random_state, max_iter=10000)),
            {
                "elasticnet__alpha": [0.01, 0.1, 1.0],
                "elasticnet__l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
        "Bayesian Ridge": (
            make_pipeline(StandardScaler(), BayesianRidge()),
            {},  # defaults are already sensible; no grid needed
        ),
        # ---- Tree ensembles (the heavy hitters) ------------------------
        "Extra Trees": (
            ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 3],
                "max_features": [0.6, 0.8, 1.0],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 6, 12],
                "min_samples_leaf": [1, 3],
                "max_features": [0.6, 0.8, 1.0],
                "bootstrap": [True, False],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "n_estimators": [150, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3, 4],
                "min_samples_leaf": [3, 10],
                "subsample": [0.8, 1.0],
            },
        ),
        "Hist Gradient Boosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            {
                "max_iter": [200, 400],
                "learning_rate": [0.05, 0.1],
                "max_depth": [None, 6, 10],
                "min_samples_leaf": [10, 20],
                "l2_regularization": [0.0, 1.0],
            },
        ),
        # ---- Instance-based --------------------------------------------
        "KNN": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {
                "kneighborsregressor__n_neighbors": [3, 5, 7, 11],
                "kneighborsregressor__weights": ["uniform", "distance"],
                "kneighborsregressor__p": [1, 2],
            },
        ),
        # ---- Other diverse methods -------------------------------------
        "SVR (RBF)": (
            make_pipeline(StandardScaler(), SVR(kernel="rbf")),
            {
                "svr__C": [0.1, 1.0, 10.0],
                "svr__epsilon": [0.01, 0.1, 0.5],
                "svr__gamma": ["scale", "auto"],
            },
        ),
        "AdaBoost": (
            AdaBoostRegressor(
                estimator=base_tree,
                random_state=random_state,
            ),
            {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.5],
                "loss": ["linear", "square"],
            },
        ),
        "Bagging Trees": (
            BaggingRegressor(
                estimator=base_tree,
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "n_estimators": [50, 100],
                "max_samples": [0.6, 0.8, 1.0],
                "max_features": [0.6, 0.8, 1.0],
            },
        ),
    }

    if XGBRegressor is not None:
        candidates["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "reg_lambda": [1.0, 5.0],
            },
        )

    # ---- Stacking: meta-ensemble built from a couple of strong base learners
    # We build the candidates lazily and use a small grid (only ``passthrough``)
    # because the inner learners are already tuned via their own entries.
    stacking_estimators = [
        ("gbr", GradientBoostingRegressor(random_state=random_state, n_estimators=200, max_depth=3)),
        ("hgbr", HistGradientBoostingRegressor(random_state=random_state, max_iter=200)),
        ("ridge", make_pipeline(StandardScaler(), Ridge(random_state=random_state))),
    ]
    candidates["Stacking"] = (
        StackingRegressor(
            estimators=stacking_estimators,
            final_estimator=ridge_for_meta,
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
            n_jobs=-1,
        ),
        {
            "passthrough": [False, True],
        },
    )

    return candidates


def filter_chemical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    for chemical_col in CHEMICAL_TARGETS:
        q1 = model_df[chemical_col].quantile(0.25)
        q3 = model_df[chemical_col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        model_df = model_df[model_df[chemical_col] <= upper_limit].copy()
    return model_df


def train_best_model(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    random_state: int = 42,
    filter_outliers: bool = False,
) -> tuple[object, dict]:
    model_df = df.dropna(subset=features + [target]).copy()
    if filter_outliers:
        model_df = filter_chemical_outliers(model_df)
    if len(model_df) < 30:
        raise ValueError(f"No hay suficientes filas para entrenar {target}.")

    train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=random_state, shuffle=True)
    X_train = train_df[features].values
    X_test = test_df[features].values
    y_train = train_df[target].values
    y_test = test_df[target].values
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    best = None
    results = []
    for name, (estimator, params) in regression_candidates(random_state).items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        metrics = {
            "model_name": name,
            "r2_train": float(r2_score(y_train, pred_train)),
            "r2_test": float(r2_score(y_test, pred_test)),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "best_params": search.best_params_,
        }
        results.append(metrics)
        if best is None or metrics["r2_test"] > best["metrics"]["r2_test"]:
            best = {"model": model, "metrics": metrics}

    best["model"].fit(model_df[features].values, model_df[target].values)
    return best["model"], {"target": target, "features": features, "best": best["metrics"], "all_results": results}
