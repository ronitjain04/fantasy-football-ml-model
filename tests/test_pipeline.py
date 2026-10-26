from pathlib import Path
import pandas as pd
import pytest
from src.etl import load_all
from src.features import build_feature_table

DATA = Path(__file__).resolve().parents[1] / 'data'


def test_bundled_data_has_unique_players_and_correct_components():
    data = load_all(data_dir=DATA)
    assert not data.duplicated(['player_id', 'season']).any()
    cmc = data[(data.player_id == 'McCaCh01') & (data.season == 2023)].iloc[0]
    assert cmc.player == 'Christian McCaffrey'
    assert cmc.position == 'RB'
    assert cmc.fantasy_points == 391.3
    assert cmc.rush_yds == 1459
    assert cmc.rec_yds == 564
    assert cmc.pass_yds == 0
    assert set(data.position) <= {'QB', 'RB', 'WR', 'TE'}
    assert 'adp' not in data or data.adp.isna().all()


def test_traded_player_uses_aggregate_once():
    data = load_all(data_dir=DATA)
    adams = data[(data.player_id == 'AdamDa01') & (data.season == 2024)]
    assert len(adams) == 1
    assert adams.iloc[0].rec_yds == 1063
    assert adams.iloc[0].games == 14


def test_partial_table_uses_explicit_year_and_exact_aliases(tmp_path):
    (tmp_path / '2022').mkdir()
    (tmp_path / '2022' / 'receiving.csv').write_text(
        'Player,Pos,Team,G,Rec,Yds,TD,Fmb\nA*,WR,ATL,10,30,"1,000",5,4\n')
    data = load_all(data_dir=tmp_path)
    row = data.iloc[0]
    assert row.player == 'A'
    assert row.rec_yds == 1000
    assert row.receptions == 30
    assert row.fantasy_points == 160  # no lost-fumble column; total fumbles aren't losses
    assert row.season == 2022
    assert row.games == 10


def test_features_keep_forecast_rows_and_require_adjacent_seasons():
    data = pd.DataFrame({
        'player_id': ['a', 'a', 'b', 'b'], 'player': ['A', 'A', 'B', 'B'],
        'season': [2022, 2024, 2023, 2024], 'fantasy_points': [80, 90, 100, 150],
        'games': [10, 10, 10, 10], 'position': ['WR'] * 4,
    })
    result = build_feature_table(data)
    assert len(result) == 4
    assert result.loc[(result.player_id == 'a') & (result.season == 2022), 'target_next_pts'].isna().all()
    assert result.loc[(result.player_id == 'b') & (result.season == 2023), 'target_next_pts'].iloc[0] == 150
    assert result[result.season == 2024].target_next_pts.isna().all()
    assert result.loc[result.season == 2023, 'ppg'].iloc[0] == 10


def test_missing_data_is_actionable(tmp_path):
    with pytest.raises(ValueError, match='No usable'):
        load_all(data_dir=tmp_path)


def test_malformed_supplied_table_fails(tmp_path):
    year = tmp_path / '2023'
    year.mkdir()
    (year / 'passing.csv').write_text('Wrong,Yds\nA,100\n')
    with pytest.raises(ValueError, match='player'):
        load_all(data_dir=tmp_path)


def test_metadata_only_rows_are_not_invented_zero_point_seasons(tmp_path):
    year = tmp_path / '2023'
    year.mkdir()
    (year / 'receiving.csv').write_text('Player,Pos,G\nA,WR,10\n')
    with pytest.raises(ValueError, match='scoring'):
        load_all(data_dir=tmp_path)


def test_flat_files_aliases_and_published_points_take_precedence(tmp_path):
    (tmp_path / 'fantasy_ranks.csv').write_text(
        'player_name,year,position,games,player_id,fantasy_points,rush_yds,adp\nA+,2023,RB,12,a,200,900,20\n')
    (tmp_path / 'rushing.csv').write_text(
        'Player,season,Pos,G,Player-additional,Yds,TD\nA,2023,RB,12,a,950,4\n')
    data = load_all(data_dir=tmp_path)
    assert len(data) == 1
    assert data.iloc[0].rush_yds == 900
    assert data.iloc[0].fantasy_points == 200
    assert data.iloc[0].adp == 20


def test_same_team_conflicting_duplicates_are_rejected(tmp_path):
    year = tmp_path / '2023'
    year.mkdir()
    (year / 'rushing.csv').write_text('Player,Pos,Team,G,Yds\nA,RB,ATL,12,500\nA,RB,ATL,12,600\n')
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        load_all(data_dir=tmp_path)


def test_scoring_counts_losses_and_two_point_conversions():
    from src.features import fantasy_points_from_components
    points = fantasy_points_from_components(pd.Series({
        'pass_yds': 100, 'pass_td': 1, 'interceptions': 1, 'rush_yds': 10,
        'rush_td': 1, 'rec_yds': 20, 'rec_td': 1, 'receptions': 3,
        'fumbles_lost': 1, 'two_point_conversions': 1, 'fumbles': 9}))
    assert points == 24
