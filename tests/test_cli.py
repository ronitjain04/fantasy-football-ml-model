import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / 'main.py'), *map(str, args)],
                          cwd=ROOT, text=True, capture_output=True)


def test_train_and_reload_cli_preserve_data(tmp_path):
    before = {p: hashlib.sha256(p.read_bytes()).digest() for p in (ROOT / 'data').glob('*/*.csv')}
    out = tmp_path / 'trained'
    result = run('train', '--output-dir', out)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((out / 'metrics.json').read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    assert metrics['forecast_season'] == 2025
    assert metrics['data_quality']['seasons'] == [2023, 2024]
    ranks = pd.read_csv(out / 'rankings.csv')
    assert ranks.target_season.eq(2025).all()
    assert ranks.season.eq(2024).all()
    assert ranks.predicted_points.notna().all()
    assert ranks.overall_rank.tolist() == list(range(1, len(ranks) + 1))
    assert ranks.groupby('position').position_rank.min().eq(1).all()
    assert (out / 'feature_importance.csv').exists()
    prediction = run('predict', '--model', out / 'model.joblib', '--output-dir', tmp_path / 'reloaded')
    assert prediction.returncode == 0, prediction.stderr
    reloaded = pd.read_csv(tmp_path / 'reloaded' / 'rankings.csv')
    pd.testing.assert_frame_equal(ranks, reloaded)
    assert before == {p: hashlib.sha256(p.read_bytes()).digest() for p in before}
    filtered = run('predict', '--model', out / 'model.joblib', '--position', 'RB',
                   '--top', '5', '--output-dir', tmp_path / 'rb')
    assert filtered.returncode == 0, filtered.stderr
    rb = pd.read_csv(tmp_path / 'rb' / 'rankings.csv')
    assert len(rb) == 5
    assert rb.position.eq('RB').all()


def test_cli_bad_data_fails_without_partial_outputs(tmp_path):
    out = tmp_path / 'out'
    result = run('train', '--data-dir', tmp_path / 'missing', '--output-dir', out)
    assert result.returncode != 0
    assert 'No usable' in result.stderr
    assert not out.exists()


def test_legacy_entry_and_help_are_safe(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'src' / 'train.py'), '--help'],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0
    assert '--data-dir' in result.stdout
