# NFL Fantasy Football — Next-Season PPR Forecasts

A reproducible Python pipeline that loads season-level player statistics, evaluates baseline and regression models, and forecasts **next-season full-PPR points for QB, RB, WR, and TE**.

The bundled data covers **2023 and 2024**. Running it produces a **2025 retrospective forecast** from 2024 stats. It does not produce current 2026 rankings or incorporate current injuries, rosters, or rookies.

## Pipeline

```text
Season CSVs → player/season normalization → observed-season features
            → next-year labels → model comparison → saved model + forecasts
```

- Load multiple seasons while reconciling player IDs and traded-player totals.
- Compare a last-season baseline with Ridge and Random Forest; enable XGBoost optionally.
- Export held-out predictions, overall and position rankings, feature importance, and data-quality summaries.
- Reuse the saved estimator and its preprocessing to generate the same forecasts without retraining.

## Quickstart

Requires Python 3.10 or newer. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py train
```

For the exact tested Python 3.12 environment, including XGBoost and pytest:

```bash
python -m pip install -r requirements-lock.txt
python main.py train --include-xgboost
python -m pytest -q
```

The normal requirement files allow compatible dependency updates.

This compares last-season points, Ridge, and Random Forest using deterministic validation folds, selects the lowest validation MAE, fits that estimator on all labeled rows, and writes `artifacts/`.

```bash
# Train against another dataset; forecast the latest observed season + 1
python main.py train --data-dir ./my_data --output-dir ./artifacts-custom

# Reuse the fitted model without retraining
python main.py predict --model artifacts/model.joblib --output-dir artifacts-reloaded

# Export the top 20 running backs who played at least 6 games
python main.py predict --model artifacts/model.joblib --position RB --min-games 6 --top 20

# Optional XGBoost comparison (not required for normal operation)
python -m pip install -r requirements-optional.txt
python main.py train --include-xgboost --output-dir artifacts-xgb

# Both legacy training entry points also work
python src/train.py
python -m src.train
```

Running `python main.py` defaults to training. Use `python main.py train --help` or `python main.py predict --help` for options. Paths default to the repository's data and artifacts directories even when launched elsewhere. `--season 2024` selects the observed source year and forecasts 2025; it cannot turn old data into a later forecast. A saved model cannot forecast from source seasons earlier than its latest training target year.

Only load trusted joblib models: deserialization can execute code. Regenerate the model after changing Python or dependency versions.

## Example run

With the bundled data, the default training command reports:

```text
Selected Ridge by validation MAE (45.19 points).
Wrote 620 player forecasts for 2025
```

The evaluation covers 464 returning players. Ridge's MAE is 45.19 points versus 46.31 for the last-season baseline. This is a modest improvement on held-out players; it does not establish accuracy on an unseen future season. See the full model comparison below.

## Project structure

```text
main.py                    Train/predict command-line entry point
src/etl.py                 CSV loading, normalization, and deduplication
src/features.py            Observed-season features and exact-next-year labels
src/model.py               Validation, model comparison, and fitted estimators
src/train.py               CLI options and artifact generation
data/2023/, data/2024/     Bundled season-level CSVs
tests/                     Data, model, and end-to-end regression tests
.github/workflows/tests.yml  Python 3.10/3.12 test workflow
```

## Outputs

| File | Contents |
| --- | --- |
| `metrics.json` | Candidate MAE, RMSE, Spearman correlation, selected model, fold details, limitations, source-data summary |
| `predictions.csv` | Held-out next-season actuals and predictions, candidate predictions, baseline, validation fold |
| `rankings.csv` | Latest-season inputs with forecasts for `target_season`, overall and position ranks |
| `model.joblib` | Fitted estimator, preprocessing, feature order, scoring and training metadata |
| `feature_importance.csv` | Absolute standardized Ridge coefficients or tree importances; baseline feature when selected |
| `evaluation_by_position.csv` | Held-out metrics by position for the selected model |
| `data_quality.json` | Source files, source-row counts, canonical player-season counts, reconstructed-point counts |

`predict` writes `rankings.csv` only. Output filters affect forecasts, not model fitting or validation. When filtering by position, `overall_rank` is the rank within that exported population. `team` and `position` reflect the source season, not the forecast-year roster. Players with no observed source-season games are excluded by the default minimum of one game. Stable player ID breaks ranking ties. Undefined Spearman correlations are JSON `null`.

## Input data

Put CSVs in `data/YYYY/` (year inferred from the folder) or place them directly under a data root with a `season`/`year` column:

```text
data/
  2023/
    fantasy_ranks.csv
    passing.csv
    rushing.csv
    receiving.csv
  2024/
    fantasy_ranks.csv
    passing.csv
    rushing.csv
    receiving.csv
```

At least one usable table per season is needed. Training requires at least five player-season pairs with an immediately following year and at least two distinct players. Missing optional tables are allowed. Team defenses and kicking files are intentionally outside the model's scoring scope.

Each row needs `Player`/`player_name`/`name`, a year (column or folder), and position (`Pos`, `FantPos`, or `position`) in at least one matching source. `G`/`games` supports per-game features and forecast eligibility; missing games remain missing and do not qualify for forecasts. `Tm`/`Team`, age and explicit preseason ADP are optional. `Player-additional`, `-9999`, or `player_id` identifies players; cleaned names are a fallback only when IDs are unavailable. Distinct players with identical names should supply IDs.

Supported component aliases include `Yds` and `TD` within each respective passing/rushing/receiving table, or explicit names such as `pass_yds`, `rush_td`, `rec_yds`, and `receptions`. The bundled fantasy tables contain repeated `Yds`/`TD` headers; their fixed passing/rushing/receiving order is handled explicitly. Custom wide fantasy tables should use canonical named columns. `PPR`/`fantasy_points` means **full-PPR season totals**; do not supply standard or half-PPR totals under that name.

The loader strips trailing `*`/`+` award markers, reconciles stable IDs, and prefers `TOT`/`2TM`/`3TM` aggregate rows over individual team stints. Distinct stints without an aggregate are summed. Identical duplicates are removed; conflicting duplicates within the same team fail clearly. Numeric strings with commas are accepted. Rows with neither published points nor any observed scoring component fail. Invalid numeric cells and fractional/missing seasons fail rather than being silently converted to zero. A traded player can legitimately have 18 appearances because of different team bye weeks.

The fantasy table takes precedence for totals, games and metadata; component tables fill missing fields. No scraping or data downloading occurs. Inputs are never modified. Missing statistical components remain missing for model preprocessing; reconstructing points treats unavailable components as zero and is reported in the data quality file.

Legacy path overrides remain available. An explicit environment path replaces discovery for that table type, so a multi-year override file must include a season column:

```bash
PLAYER_PASSING=./custom/passing.csv PLAYER_RUSHING=./custom/rushing.csv \
PLAYER_RECEIVING=./custom/receiving.csv FANTASY_RANKS=./custom/fantasy_ranks.csv \
python main.py train
```

## Scoring and features

Published full-PPR totals are preferred. Otherwise points are reconstructed as:

```text
.04 × passing yards + 4 × passing TDs − 2 × interceptions
+ .1 × rushing yards + 6 × rushing TDs
+ .1 × receiving yards + 6 × receiving TDs + receptions
− 2 × fumbles lost + 2 × two-point conversions
```

Total fumbles (`Fmb`) are not assumed to be lost fumbles (`FL`). Return scores, bonuses and other uncommon scoring events may be absent from component-only reconstructions. Published PPR preserves the provider's scoring totals.

Features use observed-season points, points per game, games, age, stat components, optional explicit ADP and position indicators. Missing values are imputed inside each training fold. Ridge scaling is also fitted within each fold. End-of-season fantasy rank is never used as preseason ADP. Weekly boom/bust/consistency metrics are omitted because season totals cannot establish them.

## What the evaluation establishes

Labels match the **same player in exactly the following calendar year**, using stable IDs. A gap from 2022 to 2024 never becomes a 2023 label. Latest-season rows remain available for inference and are not used as labeled training samples.

- With the two bundled seasons there is only one labeled transition, 2023 → 2024. Five-fold player-group validation holds players out of training, but **does not test a future unseen season**.
- With additional consecutive seasons, validation automatically uses expanding chronological folds: train on earlier feature years and test the next feature year. Earlier labels end no later than the test input year. The first training year has no held-out predictions.
- MAE and RMSE are measured in season fantasy points; Spearman measures ordering. Scores are pooled over held-out predictions. Every candidate uses the same folds, including the last-season-points baseline.
- Only players observed in both adjacent seasons have labels. Missing players are not automatically assigned zero; this creates survivor bias and does not model retirement or complete absence.
- Choosing the best model on these folds makes its reported score a model-selection result, not an independent final-test score. More seasons and a separate future holdout are needed for stronger conclusions.
- Feature importance is descriptive, not causal. Small, zero-volume players dominate much of this dataset; inspect position metrics and active-player performance before relying on rankings.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover known bundled player totals, traded-player aggregation, optional tables, malformed inputs, exact-year labels, chronological and group-disjoint validation, finite metrics, saved-model roundtrips, command-line outputs and byte-for-byte preservation of source CSVs. GitHub Actions runs the base suite on Python 3.10 and 3.12. XGBoost is opt-in and not required by CI.

## Bundled-data result

Verified on the bundled 2023/2024 data with the locked environment:

| Candidate | MAE | RMSE | Spearman |
| --- | ---: | ---: | ---: |
| Last-season baseline | 46.31 | 66.49 | 0.732 |
| Ridge | 45.19 | 62.99 | 0.723 |
| Random Forest | 46.31 | 64.30 | 0.689 |
| XGBoost (optional) | 47.37 | 66.07 | 0.686 |

There are 464 labeled returning-player pairs and 620 eligible 2024 players for 2025 forecasts. Ridge improves point-error metrics modestly; the baseline has the stronger rank correlation. These are player-group model-selection results with the limitations above, not evidence of future-season accuracy.
