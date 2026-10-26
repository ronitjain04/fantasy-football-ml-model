"""Application CLI. Supports both python -m src.train and python src/train.py."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.etl import PROJECT_ROOT, load_all
from src.features import build_feature_table
from src.model import evaluate_models, predict, regression_metrics


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def _rankings(bundle, features, season=None, position=None, min_games=1, top=None):
    season = int(features.season.max()) if season is None else season
    if season < bundle['last_training_target_season']:
        raise ValueError('Forecast source season precedes training labels; use a later season or retrain')
    rows = features[(features.season == season) & (features.games.fillna(0) >= min_games)].copy()
    if position:
        rows = rows[rows.position == position]
    if rows.empty:
        raise ValueError('No forecast players match the season, position, and minimum games filters')
    rows['predicted_points'] = predict(bundle, rows)
    rows['target_season'] = rows.season + 1
    rows = rows.sort_values(['predicted_points', 'player_id'], ascending=[False, True]).reset_index(drop=True)
    rows['overall_rank'] = np.arange(1, len(rows) + 1)
    rows['position_rank'] = rows.groupby('position').cumcount() + 1
    columns = ['target_season', 'overall_rank', 'position_rank', 'player_id', 'player',
               'position', 'team', 'season', 'games', 'fantasy_points', 'predicted_points']
    return rows[columns].head(top) if top else rows[columns]


def _importance(bundle):
    estimator = bundle['estimator']
    if bundle['model_name'] == 'LastSeason':
        importance = np.zeros(len(bundle['features']))
        importance[0] = 1
        method = 'baseline uses prior fantasy_points only'
    else:
        final = estimator.steps[-1][1]
        if hasattr(final, 'feature_importances_'):
            importance = final.feature_importances_
            method = 'tree impurity/gain importance; not causal'
        else:
            importance = np.abs(final.coef_)
            method = 'absolute standardized Ridge coefficient; not causal'
    return pd.DataFrame({'feature': bundle['features'], 'importance': importance, 'method': method}).sort_values(
        'importance', ascending=False)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith('-'):
        argv.insert(0, 'train')
    parser = argparse.ArgumentParser(description='Train and forecast season-level full-PPR NFL player points.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['train', 'predict']:
        command = commands.add_parser(name)
        command.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data',
                             help='CSV root: YYYY directories or flat files with a season column')
        command.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'artifacts')
        command.add_argument('--season', type=int, help='Observed source season; forecasts season + 1 (default: latest)')
        command.add_argument('--position', choices=['QB', 'RB', 'WR', 'TE'])
        command.add_argument('--min-games', type=_positive, default=1, help='Forecast eligibility; does not filter evaluation')
        command.add_argument('--top', type=_positive, help='Limit exported forecast rankings')
    commands.choices['train'].add_argument('--include-xgboost', action='store_true', help='Requires requirements-optional.txt')
    commands.choices['predict'].add_argument('--model', type=Path, required=True, help='Trusted model.joblib produced by train')
    args = parser.parse_args(argv)
    try:
        data = load_all(data_dir=args.data_dir)
        features = build_feature_table(data)
        if args.command == 'train':
            bundle, metrics, predictions = evaluate_models(features, include_xgboost=args.include_xgboost)
        else:
            if not args.model.is_file():
                raise ValueError(f'Model not found: {args.model}')
            bundle = joblib.load(args.model)
            metrics = None
        rankings = _rankings(bundle, features, args.season, args.position, args.min_games, args.top)
        # Validate all inputs and compute outputs before creating the output directory.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        rankings.to_csv(args.output_dir / 'rankings.csv', index=False)
        if args.command == 'train':
            metrics['forecast_season'] = int(rankings.target_season.iloc[0])
            metrics['data_quality'] = data.attrs['quality']
            metrics['n_forecast_players'] = len(rankings)
            _write_json(args.output_dir / 'metrics.json', metrics)
            _write_json(args.output_dir / 'data_quality.json', data.attrs['quality'])
            predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
            _importance(bundle).to_csv(args.output_dir / 'feature_importance.csv', index=False)
            by_position = [{'position': pos, 'n': len(group),
                            **regression_metrics(group.target_next_pts, group.predicted_points)}
                           for pos, group in predictions.groupby('position')]
            pd.DataFrame(by_position).to_csv(args.output_dir / 'evaluation_by_position.csv', index=False)
            joblib.dump(bundle, args.output_dir / 'model.joblib')
            print(f"Selected {bundle['model_name']} by validation MAE ({metrics['metrics'][bundle['model_name']]['mae']:.2f} points).")
            print(metrics['validation']['note'])
        print(f"Wrote {len(rankings)} player forecasts for {int(rankings.target_season.iloc[0])} to {args.output_dir.resolve()}")
        print('Historical inputs only: forecast year follows the source data, regardless of the current calendar year.')
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
