# NFL Player Output Prediction — ML Pipeline

This is a clean, production-lean Python pipeline that ingests season-level CSVs (passing, rushing, receiving, fantasy ranks), 
engineers features, and trains a regression model to predict next-season fantasy points.

It’s robust to common column-name variations and produces interview-friendly artifacts (predictions, rankings, metrics).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If your CSVs are in ./data (default)
python src/train.py

# Or point to custom paths
PLAYER_PASSING=./my_data/passing.csv PLAYER_RUSHING=./my_data/rushing.csv PLAYER_RECEIVING=./my_data/receiving.csv FANTASY_RANKS=./my_data/fantasy_ranks.csv python src/train.py
```

Outputs will be in `./artifacts`:
- `metrics.json` — cross-validated MAE, RMSE, Spearman ρ, features used
- `predictions.csv` — per player-season with true vs predicted next-season points
- `rankings.csv` — simple season-level rankings by predicted points

## Environment variables (optional)

- `PLAYER_PASSING` — path to passing CSV
- `PLAYER_RUSHING` — path to rushing CSV
- `PLAYER_RECEIVING` — path to receiving CSV
- `FANTASY_RANKS` — path to fantasy/ADP CSV

If any dataset is missing, the pipeline still runs with what’s available.

## Notes
- Weekly stats are optional; consistency/boom/bust are approximated from season-level info when weekly data is absent.
- XGBoost is optional; if installed, it will also be trained and compared to Ridge.
