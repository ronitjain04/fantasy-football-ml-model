import joblib
import numpy as np
import pandas as pd
import pytest
from src import model
from src.features import build_feature_table


def fixture(seasons=(2023, 2024)):
    rows = [{'player_id': str(i), 'player': f'P{i}', 'season': y,
             'position': 'WR', 'games': 10, 'fantasy_points': i * 8 + (y - 2020) * 2}
            for y in seasons for i in range(12)]
    return build_feature_table(pd.DataFrame(rows))


def test_two_season_evaluation_is_group_disjoint_and_reloadable(tmp_path):
    bundle, metrics, predictions = model.evaluate_models(fixture())
    assert metrics['validation']['strategy'] == 'player_group_kfold'
    assert all(fold['player_overlap'] == 0 for fold in metrics['validation']['folds'])
    assert metrics['n_evaluated'] == 12
    assert len(predictions) == 12
    assert metrics['metrics']['LastSeason']['mae'] == 2
    for values in metrics['metrics'].values():
        assert np.isfinite(values['rmse'])
    path = tmp_path / 'model.joblib'
    joblib.dump(bundle, path)
    forecast = fixture().query('season == 2024')
    np.testing.assert_allclose(model.predict(bundle, forecast), model.predict(joblib.load(path), forecast))


def test_more_seasons_use_only_past_labels():
    _, metrics, predictions = model.evaluate_models(fixture((2021, 2022, 2023, 2024)))
    assert metrics['validation']['strategy'] == 'expanding_season'
    assert all(max(f['train_target_seasons']) <= min(f['test_feature_seasons'])
               for f in metrics['validation']['folds'])
    assert set(predictions.season) == {2022, 2023}


def test_undefined_correlation_is_json_null():
    metrics = model.regression_metrics(np.ones(5), np.ones(5))
    assert metrics['spearman_rho'] is None
    assert metrics['rmse'] == 0


def test_tiny_training_data_fails_clearly():
    with pytest.raises(ValueError, match='At least'):
        model.evaluate_models(fixture().iloc[:2])
