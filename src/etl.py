"""Read season CSVs without multiplying traded-player rows or conflating stats."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITIONS = {'QB', 'RB', 'WR', 'TE'}
KEYS = ['player_id', 'season']
METADATA = ['player', 'team', 'position', 'age', 'games']
COMPONENTS = ['pass_yds', 'pass_td', 'interceptions', 'rush_yds', 'rush_td',
              'rec_yds', 'rec_td', 'receptions', 'fumbles_lost', 'two_point_conversions']
COMMON = {
    'player': ['player', 'player_name', 'name'],
    'player_id': ['player_id', 'player-additional', '-9999'],
    'season': ['season', 'year'], 'team': ['team', 'tm'],
    'position': ['position', 'pos', 'fantpos'], 'age': ['age'],
    'games': ['games', 'g', 'gp'],
}
SCHEMAS = {
    'passing': {'pass_yds': ['pass_yds', 'passing_yards', 'yds'],
                'pass_td': ['pass_td', 'passing_tds', 'td'],
                'interceptions': ['interceptions', 'int', 'ints']},
    'rushing': {'rush_yds': ['rush_yds', 'rushing_yards', 'yds'],
                'rush_td': ['rush_td', 'rushing_tds', 'td']},
    'receiving': {'rec_yds': ['rec_yds', 'receiving_yards', 'yds'],
                  'rec_td': ['rec_td', 'receiving_tds', 'td'],
                  'receptions': ['receptions', 'rec']},
    'fantasy_ranks': {
        'fantasy_points': ['ppr', 'fantasy_points'],
        'pass_yds': ['pass_yds', 'passing_yards', 'yds'],
        'pass_td': ['pass_td', 'passing_tds', 'td'],
        'interceptions': ['interceptions', 'int'],
        'rush_yds': ['rush_yds', 'rushing_yards', 'yds.1'],
        'rush_td': ['rush_td', 'rushing_tds', 'td.1'],
        'rec_yds': ['rec_yds', 'receiving_yards', 'yds.2'],
        'rec_td': ['rec_td', 'receiving_tds', 'td.2'],
        'receptions': ['receptions', 'rec'],
        'two_point_conversions': ['two_point_conversions', '2pm'],
        # End-of-season Rk/OvRank is NOT preseason ADP.
        'adp': ['adp', 'avg_draft_position', 'average_draft_position'],
    },
}


def _numeric(series):
    return pd.to_numeric(series.astype('string').str.replace(',', '', regex=False), errors='coerce')


def _normalize(path, kind):
    try:
        raw = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f'{path}: empty or malformed CSV') from exc
    raw.columns = [str(c).strip().lower().replace(' ', '_') for c in raw.columns]
    mapping = {**COMMON, **SCHEMAS[kind], 'fumbles_lost': ['fumbles_lost', 'fl']}
    out = pd.DataFrame(index=raw.index)
    for column, aliases in mapping.items():
        found = next((a for a in aliases if a in raw), None)
        if found:
            out[column] = raw[found]
    if 'player' not in out:
        raise ValueError(f'{path}: missing player/name column')
    out = out[out.player.notna()].copy()
    out['player'] = out.player.astype(str).str.replace(r'[*+]+$', '', regex=True).str.strip()
    out = out[(out.player != '') & (out.player.str.lower() != 'player')].copy()
    if 'season' not in out:
        if not path.parent.name.isdigit() or len(path.parent.name) != 4:
            raise ValueError(f'{path}: provide a season/year column or put CSV in a YYYY directory')
        out['season'] = int(path.parent.name)
    for col in out.columns.difference(['player', 'player_id', 'team', 'position']):
        source = out[col]
        converted = _numeric(source)
        invalid = source.notna() & source.astype(str).str.strip().ne('') & converted.isna()
        if invalid.any() or (converted.notna() & ~np.isfinite(converted)).any():
            raise ValueError(f'{path}: invalid numeric values in {col}')
        out[col] = converted
    if out.season.isna().any() or (out.season % 1 != 0).any():
        raise ValueError(f'{path}: season must be an integer year')
    out['season'] = out.season.astype(int)
    for col in ['team', 'position']:
        if col not in out:
            out[col] = pd.NA
        out[col] = out[col].astype('string').str.strip().str.upper()
    if 'player_id' not in out:
        out['player_id'] = pd.NA
    out['player_id'] = out.player_id.astype('string').str.strip().replace('', pd.NA)
    return out


def _deduplicate(table):
    """Prefer published all-team totals; otherwise combine distinct team stints."""
    rows = []
    table = table.drop_duplicates()
    for _, group in table.groupby(KEYS, sort=False):
        aggregate = group[group.team.fillna('').str.match(r'^(TOT|[2-9]TM)$')]
        if not aggregate.empty:
            if len(aggregate) != 1:
                raise ValueError('Conflicting aggregate rows for the same player-season')
            row = aggregate.iloc[0].copy()
        elif len(group) > 1:
            if group.team.duplicated().any():
                raise ValueError('Conflicting duplicate rows for the same player-season-team')
            row = group.iloc[-1].copy()
            for col in set(group.columns) - set(KEYS + ['player', 'team', 'position', 'age', 'adp']):
                row[col] = group[col].sum(min_count=1)
            row['team'] = f'{len(group)}TM'
        else:
            row = group.iloc[0].copy()
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def load_all(data_dir=None, passing_path=None, rushing_path=None,
             receiving_path=None, fantasy_path=None):
    """Load data/YYYY/*.csv (or flat CSVs with season), with optional path overrides.

    Explicit arguments override environment variables, which override discovery for
    their respective table. Missing discovered tables are optional; explicit paths
    must exist. DataFrame attrs include source paths and quality counts.
    """
    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / 'data'
    overrides = {
        'passing': passing_path or os.getenv('PLAYER_PASSING'),
        'rushing': rushing_path or os.getenv('PLAYER_RUSHING'),
        'receiving': receiving_path or os.getenv('PLAYER_RECEIVING'),
        'fantasy_ranks': fantasy_path or os.getenv('FANTASY_RANKS'),
    }
    tables, sources = [], []
    # Fantasy supplies authoritative totals, metadata, and loss-only fumbles.
    for kind in ['fantasy_ranks', 'passing', 'rushing', 'receiving']:
        if overrides[kind]:
            paths = [Path(overrides[kind])]
            if not paths[0].is_file():
                raise ValueError(f'CSV not found: {paths[0]}')
        else:
            paths = sorted(root.glob(f'[0-9][0-9][0-9][0-9]/{kind}.csv'))
            if (root / f'{kind}.csv').is_file():
                paths.append(root / f'{kind}.csv')
        for path in paths:
            table = _normalize(path, kind)
            if not table.empty:
                tables.append(table)
                sources.append(str(path.resolve()))
    if not tables:
        raise ValueError(f'No usable tables found in {root}; expected data/YYYY/*.csv')
    # Reconcile tables without IDs only when the name maps unambiguously to one ID.
    name_ids = {}
    for table in tables:
        for name, pid in zip(table.player.str.casefold(), table.player_id):
            if pd.notna(pid):
                name_ids.setdefault(name, set()).add(pid)
    for table in tables:
        fallback = table.player.str.casefold().map(
            lambda name: next(iter(name_ids[name])) if len(name_ids.get(name, set())) == 1 else f'name:{name}')
        table['player_id'] = table.player_id.fillna(fallback)
    before = sum(len(t) for t in tables)
    combined = None
    for table in tables:
        table = _deduplicate(table).set_index(KEYS)
        if combined is None:
            combined = table
        else:
            combined = combined.combine_first(table)
    data = combined.reset_index()
    data = data[data.position.isin(POSITIONS)].copy()
    if data.empty:
        raise ValueError('No usable QB/RB/WR/TE rows; provide position/pos/FantPos')
    for column in COMPONENTS:
        if column not in data:
            data[column] = np.nan
    if 'games' not in data:
        data['games'] = np.nan
    if ((data.games.dropna() < 0) | (data.games.dropna() > 18)).any():
        raise ValueError('Games must be between 0 and 18 for regular-season inputs')
    from .features import fantasy_points_from_components
    computed = data.apply(fantasy_points_from_components, axis=1)
    missing_totals = data.get('fantasy_points', pd.Series(np.nan, index=data.index)).isna()
    no_scoring_data = missing_totals & data[COMPONENTS].isna().all(axis=1)
    if no_scoring_data.any():
        names = ', '.join(data.loc[no_scoring_data, 'player'].head(3))
        raise ValueError(f'Rows lack published points and scoring components: {names}')
    data['fantasy_points'] = pd.to_numeric(data.get('fantasy_points', computed), errors='raise').fillna(computed)
    data['points_reconstructed'] = missing_totals
    data = data.sort_values(KEYS).reset_index(drop=True)
    data.attrs['quality'] = {
        'source_files': sources, 'source_rows': before,
        'player_seasons': len(data), 'seasons': sorted(data.season.unique().tolist()),
        'reconstructed_point_rows': int(missing_totals.sum()),
        'notes': ['Fantasy PPR totals preferred; component reconstruction treats missing stats as zero.',
                  'Total fumbles are not assumed to be lost fumbles.',
                  'Kickers and team defenses excluded.'],
    }
    return data
