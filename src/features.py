"""Observed-season features and labels joined by exactly the next calendar year."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['fantasy_points', 'ppg', 'games', 'age', 'adp',
                   'pass_yds', 'pass_td', 'interceptions', 'rush_yds', 'rush_td',
                   'rec_yds', 'rec_td', 'receptions', 'fumbles_lost',
                   'is_qb', 'is_rb', 'is_wr', 'is_te']
TARGET_COLUMN = 'target_next_pts'
ID_COLUMNS = ['player_id', 'player', 'position', 'team', 'season']


def fantasy_points_from_components(row):
    """Full PPR: 4/pass TD, .04/pass yard, 6/other TD, .1/other yard,
    1/reception, -2/interception or fumble lost, 2/two-point conversion.
    Missing components contribute zero; provided PPR totals are preferable.
    """
    weights = {'pass_yds': .04, 'pass_td': 4, 'interceptions': -2,
               'rush_yds': .1, 'rush_td': 6, 'rec_yds': .1, 'rec_td': 6,
               'receptions': 1, 'fumbles_lost': -2, 'two_point_conversions': 2}
    return sum(weight * float(row.get(col, 0)) for col, weight in weights.items()
               if pd.notna(row.get(col, 0)))


def build_feature_table(df):
    data = df.copy()
    required = {'player_id', 'season', 'fantasy_points'}
    if not required.issubset(data):
        raise ValueError(f'Missing feature inputs: {sorted(required - set(data))}')
    if data.duplicated(['player_id', 'season']).any():
        raise ValueError('Features require one row per player-season')
    games = pd.to_numeric(data.get('games', pd.Series(np.nan, index=data.index)), errors='coerce')
    data['ppg'] = data.fantasy_points / games.where(games > 0)
    for pos in ['QB', 'RB', 'WR', 'TE']:
        data[f'is_{pos.lower()}'] = (data.get('position', '') == pos).astype(int)
    for column in FEATURE_COLUMNS:
        if column not in data:
            data[column] = np.nan
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan)
    labels = data[['player_id', 'season', 'fantasy_points']].copy()
    labels['season'] -= 1
    labels = labels.rename(columns={'fantasy_points': TARGET_COLUMN})
    # Keep unlabeled latest-season rows for genuine inference.
    return data.merge(labels, on=['player_id', 'season'], how='left', validate='one_to_one').sort_values(
        ['season', 'player_id']).reset_index(drop=True)
