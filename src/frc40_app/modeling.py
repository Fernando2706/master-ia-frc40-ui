from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CHEMICAL_TARGETS

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def regression_candidates(random_state: int) -> dict[str, tuple[object, dict]]:
    candidates = {
        "Extra Trees": (
            ExtraTreesRegressor(random_state=random_state),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 5],
                "min_samples_leaf": [1, 3],
                "max_features": [0.8, 1.0],
            },
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=random_state),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 5],
                "min_samples_leaf": [1, 3],
                "max_features": [0.8, 1.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
                "min_samples_leaf": [3, 10],
            },
        ),
        "KNN": (
            make_pipeline(StandardScaler(), KNeighborsRegressor()),
            {
                "kneighborsregressor__n_neighbors": [3, 5, 7],
                "kneighborsregressor__weights": ["uniform", "distance"],
                "kneighborsregressor__p": [1, 2],
            },
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [2, 3, 5],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 5.0],
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
