from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

@dataclass
class ModelResult:
    name: str
    mae: float
    rmse: float
    spearman_rho: float | float
    preds: np.ndarray

def _clone(model):
    # simple rules to clone Ridge or Pipeline(Ridge)
    if isinstance(model, Pipeline):
        steps = []
        for n, step in model.steps:
            if isinstance(step, StandardScaler):
                steps.append((n, StandardScaler()))
            elif isinstance(step, Ridge):
                steps.append((n, Ridge(alpha=step.alpha, random_state=42)))
            else:
                steps.append((n, step.__class__(**getattr(step, "get_params", lambda: {})())))
        return Pipeline(steps)
    if isinstance(model, Ridge):
        return Ridge(alpha=model.alpha, random_state=42)
    if XGB_AVAILABLE and isinstance(model, xgb.XGBRegressor):
        return xgb.XGBRegressor(**model.get_params())
    raise ValueError("Unsupported model for cloning.")

def kfold_eval(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[np.ndarray, dict]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.zeros_like(y, dtype=float)
    maes, rmses, rhos = [], [], []
    for tr, te in kf.split(X):
        m = _clone(model)
        m.fit(X[tr], y[tr])
        yhat = m.predict(X[te])
        preds[te] = yhat
        maes.append(mean_absolute_error(y[te], yhat))
        rmses.append(mean_squared_error(y[te], yhat, squared=False))
        rhos.append(spearmanr(y[te], yhat).correlation)
    return preds, {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)), "spearman_rho": float(np.nanmean(rhos))}

def run_models(X: np.ndarray, y: np.ndarray) -> list[ModelResult]:
    results = []

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
    rpreds, rmetrics = kfold_eval(ridge, X, y, n_splits=min(5, max(2, len(y)//5)))
    results.append(ModelResult(name="Ridge", preds=rpreds, **rmetrics))

    if XGB_AVAILABLE:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )
        xpreds, xmetrics = kfold_eval(xgb_model, X, y, n_splits=min(5, max(2, len(y)//5)))
        results.append(ModelResult(name="XGBoost", preds=xpreds, **xmetrics))

    return results
