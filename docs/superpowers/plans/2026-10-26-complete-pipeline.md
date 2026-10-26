# Fantasy Football Pipeline Implementation Plan

> Execute inline using executing-plans and test-driven-development. User authorized judgment and immediate execution.

**Goal:** Complete the existing season-level Python pipeline, producing reproducible evaluation, saved models, and next-season PPR forecasts from the bundled data.

**Architecture:** Normalize each year into one row per player using stable IDs, retaining aggregate rows for traded players. Build features from observed seasons only; join labels by exactly the following year. Evaluate candidates against last-season-points baseline, fit the selected estimator, and forecast the latest year plus one.

**Tech stack:** Python 3.10+, pandas, NumPy, scipy, scikit-learn, joblib; optional XGBoost; pytest.

**Scope:** QB/RB/WR/TE full-PPR season totals. No unsupported weekly consistency estimates, current-roster assumptions, live scraping, or kicker/defense modeling. Bundled 2023/2024 data means a 2025 retrospective forecast, not current 2026 advice. Only returning players have labels; disclose survivor bias. Native PPR totals take precedence over reconstructed components. Never use same-season end-of-season rank as draft ADP.

## Global Constraints
- Preserve input CSV bytes and existing unrelated changes.
- No downloads required for normal operation; XGBoost opt-in.
- Missing optional tables allowed; malformed supplied tables fail clearly.
- Numeric output finite; JSON strictly valid; undefined correlation represented by null.
- Validation labels cannot enter feature columns or preprocessing fits.

## Review Focus
- Traded players and repeated CSV header names must not multiply rows or confuse passing/receiving yards.
- Gaps in seasons must not produce next-year labels.
- Two seasons cannot establish a temporal backtest; report player holdout explicitly.
- Empty/tiny/partial inputs must fail clearly or run without invented stats.
- Reloaded estimator must reproduce forecasts, and CLI errors must not overwrite source data.

### Task 1: Normalize data and exact-next-year features
Files: src/etl.py, src/features.py, tests/test_pipeline.py.
Interfaces: load_all(data_dir, optional explicit table paths) -> canonical DataFrame; build_feature_table(df) -> all rows with nullable target_next_pts; FEATURE_COLUMNS excludes targets/ranks.
- [ ] Write and run failing tests for real bundled totals, deduplication, aliases, incomplete tables, and season gaps.
- [ ] Implement schema normalization with stable player IDs, metadata coalescing, precise aliases, aggregate handling, numeric validation.
- [ ] Implement full-PPR observed-season features and exact-next-season join, retaining latest rows for inference.
- [ ] Run tests and inspect canonical counts and known player totals.

### Task 2: Honest evaluation and reusable models
Files: src/model.py, tests/test_model.py.
Interfaces: evaluate_models(frame, include_xgboost=False) -> fitted bundle, metrics, held-out predictions; predict(bundle, frame) -> vector.
- [ ] Write and run failing tests for finite RMSE, group-disjoint holdout, chronological splits with >=3 seasons, constant correlations, and saved-model roundtrip.
- [ ] Implement baseline, Ridge, RandomForest, optional XGBoost, fold-local imputation/scaling, deterministic splits, selection by MAE.
- [ ] Fit selected candidate to all labeled rows; persist preprocessing, features and model metadata together.
- [ ] Run tests.

### Task 3: CLI, artifacts and documentation
Files: main.py, src/train.py, src/__init__.py, requirements*.txt, README.md, .gitignore, tests/test_cli.py, .github/workflows/tests.yml.
Interfaces: python main.py [train|predict] with --data-dir and --output-dir; python src/train.py legacy training entry.
- [ ] Write and run failing CLI tests verifying output schemas, model reload, source-data preservation, invalid paths and position filtering.
- [ ] Add train/predict commands, complete metrics, predictions, rankings, model.joblib, feature importance, data quality summary.
- [ ] Document setup, scoring, input contract, validation limits, and examples; configure CI.
- [ ] Run full tests and both real-data entry points, inspect artifacts, review implementation, copy verified changes to original project without touching input CSVs.

## Ledger
- Initial checkout has an uncommitted model.py RMSE edit; preserve its intent by implementing compatible RMSE calculation.
- Working in a writable staging copy because original repository is outside sandbox writable roots; sync only explicit changed files after verification, with permission escalation.
- User chose judgment and execution; no additional planning approval round.
- Task 1: complete; normalized bundled data and passed real totals, aggregate, aliases, missing input and exact-next-year regression tests.
- Task 2: complete; grouped and chronological folds, baseline, optional XGBoost, fitted bundles, and reload tests verified.
- Task 3: complete in staging; train/predict/legacy CLI, source-byte preservation, filters, artifacts and documentation verified.
- Ruling: up to 18 games accepted because bundled traded player Mike Williams legitimately has 18 games in 2024.
- Review: fresh reviewer identified metadata-only rows being reconstructed as zero; added failing regression test, then reject rows lacking any scoring observation. No other actionable correctness issues reported.
- Fresh environment verified base and optional XGBoost real-data training. Ridge selected: MAE 45.1923 versus baseline 46.3099; 464 labeled player-seasons; 620 forecasts for 2025. These are grouped model-selection metrics, not future-season evidence.
