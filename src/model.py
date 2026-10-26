"""Reproducible, fold-local evaluation and fitted model bundles."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET_COLUMN


class LastSeason(RegressorMixin, BaseEstimator):
    """Predict the observed season's points unchanged."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, 0]


def regression_metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    rho = None
    if len(y) > 1 and np.ptp(y) > 0 and np.ptp(predicted) > 0:
        value = float(spearmanr(y, predicted).statistic)
        if np.isfinite(value):
            rho = value
    return {'mae': float(mean_absolute_error(y, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(y, predicted))),
            'spearman_rho': rho}


def _candidates(include_xgboost):
    candidates = {
        'LastSeason': LastSeason(),
        'Ridge': make_pipeline(SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True),
                               StandardScaler(), Ridge(alpha=10.0)),
        'RandomForest': make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True),
            RandomForestRegressor(n_estimators=160, min_samples_leaf=4,
                                  max_depth=10, random_state=42, n_jobs=1)),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ValueError('XGBoost requested but unavailable; install requirements-optional.txt') from exc
        candidates['XGBoost'] = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0, keep_empty_features=True),
            XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.05,
                         subsample=.9, colsample_bytree=.9, random_state=42,
                         n_jobs=1, tree_method='hist'))
    return candidates


def predict(bundle, frame):
    if bundle.get('format_version') != 1:
        raise ValueError('Unsupported saved model format; retrain the model')
    columns = bundle['features']
    X = frame.reindex(columns=columns).astype(float).replace([np.inf, -np.inf], np.nan)
    result = np.asarray(bundle['estimator'].predict(X.to_numpy()), dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Model produced non-finite predictions')
    return result


def evaluate_models(frame, include_xgboost=False):
    labeled = frame.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    if len(labeled) < 5 or labeled.player_id.nunique() < 2:
        raise ValueError('At least 5 labeled player-seasons and 2 players are required; supply adjacent seasons')
    seasons = sorted(labeled.season.unique())
    if len(seasons) > 1:
        splits = [(np.flatnonzero(labeled.season.to_numpy() < year),
                   np.flatnonzero(labeled.season.to_numpy() == year)) for year in seasons[1:]]
        strategy = 'expanding_season'
        note = 'Each test season follows training seasons. Earlier labels are available by the test feature season end.'
    else:
        splitter = GroupKFold(n_splits=min(5, labeled.player_id.nunique()))
        splits = list(splitter.split(labeled, groups=labeled.player_id))
        strategy = 'player_group_kfold'
        note = ('Only one labeled season transition is available. Players are disjoint across train/test; '
                'this is NOT a future-season backtest. Add at least one more season for chronological evaluation.')
    X = labeled[FEATURE_COLUMNS].astype(float).to_numpy()
    y = labeled[TARGET_COLUMN].to_numpy(dtype=float)
    candidates = _candidates(include_xgboost)
    scores, outputs = {}, {}
    evaluated = np.zeros(len(labeled), dtype=bool)
    folds = []
    for number, (train, test) in enumerate(splits, start=1):
        evaluated[test] = True
        folds.append({'fold': number, 'train_rows': len(train), 'test_rows': len(test),
                      'train_feature_seasons': sorted(labeled.iloc[train].season.unique().tolist()),
                      'train_target_seasons': sorted((labeled.iloc[train].season + 1).unique().tolist()),
                      'test_feature_seasons': sorted(labeled.iloc[test].season.unique().tolist()),
                      'player_overlap': len(set(labeled.iloc[train].player_id) & set(labeled.iloc[test].player_id))})
    for name, estimator in candidates.items():
        out = np.full(len(y), np.nan)
        for train, test in splits:
            fitted = clone(estimator).fit(X[train], y[train])
            out[test] = fitted.predict(X[test])
        scores[name] = regression_metrics(y[evaluated], out[evaluated])
        outputs[name] = out
    best = min(scores, key=lambda name: scores[name]['mae'])
    fitted = clone(candidates[best]).fit(X, y)
    bundle = {'format_version': 1, 'model_name': best, 'features': list(FEATURE_COLUMNS),
              'estimator': fitted, 'scoring': 'full_ppr',
              'training_feature_seasons': seasons, 'training_rows': len(y),
              'last_training_target_season': int(labeled.season.max()) + 1}
    predictions = labeled.loc[evaluated, [c for c in ['player_id', 'player', 'position', 'team', 'season',
                                                       TARGET_COLUMN] if c in labeled]].copy()
    predictions['target_season'] = predictions.season + 1
    predictions['predicted_points'] = outputs[best][evaluated]
    predictions['baseline_points'] = outputs['LastSeason'][evaluated]
    for name, values in outputs.items():
        predictions[f'prediction_{name}'] = values[evaluated]
    predictions['fold'] = 0
    for number, (_, test) in enumerate(splits, start=1):
        predictions.loc[predictions.index.intersection(test), 'fold'] = number
    metrics = {'metrics': scores, 'best_model': best, 'selection_metric': 'mae',
               'n_samples': len(y), 'n_evaluated': int(evaluated.sum()),
               'features': list(FEATURE_COLUMNS), 'validation': {'strategy': strategy, 'note': note, 'folds': folds},
               'limitations': ['Metrics cover returning players only; absent next-year players are unlabeled, not zero-point targets.',
                               'Model selection uses the same validation folds; selected-model scores are not an independent final test.',
                               'No injury, roster, rookie, schedule, or weekly inputs; historical team and position metadata only.']}
    return bundle, metrics, predictions.reset_index(drop=True)
