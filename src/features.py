import numpy as np
import pandas as pd

def fantasy_points_from_components(row: pd.Series) -> float:
    g = lambda x: float(x) if pd.notna(x) else 0.0
    return (
        0.04 * g(row.get("pass_yds", 0)) + 4.0 * g(row.get("pass_td", 0))
        + 0.1 * g(row.get("rush_yds", 0)) + 6.0 * g(row.get("rush_td", 0))
        + 0.1 * g(row.get("rec_yds", 0)) + 6.0 * g(row.get("rec_td", 0))
        + 1.0 * g(row.get("receptions", 0))
        - 2.0 * g(row.get("fumbles", 0))
        - 2.0 * g(row.get("interceptions", 0))
    )

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

FEATURE_COLUMNS = [
    "ppg","ppg_consistency","boom_rate","bust_rate",
    "adp","adp_z",
    "pass_yds","pass_td","interceptions",
    "rush_yds","rush_td","fumbles",
    "rec_yds","rec_td","receptions",
    "games"
]

TARGET_COLUMN = "target_next_pts"
ID_COLUMNS = ["player","position","team","season"]

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "fantasy_points" not in data.columns:
        data["fantasy_points"] = data.apply(fantasy_points_from_components, axis=1)

    data["games"] = pd.to_numeric(data.get("games", np.nan), errors="coerce")
    data["ppg"] = data["fantasy_points"] / data["games"].replace(0, np.nan)
    data["ppg"] = data["ppg"].fillna(data["fantasy_points"] / 14.0)

    # Approximate weekly consistency from TD variance when weekly data isn't available
    data["ppg_consistency"] = data[["pass_td","rush_td","rec_td"]].std(axis=1, ddof=0)
    data["ppg_consistency"] = data["ppg_consistency"].fillna(data["ppg"].std())

    if "adp" in data.columns:
        data["adp_z"] = data.groupby("season")["adp"].transform(zscore)
    else:
        data["adp_z"] = 0.0

    # Boom/bust rate proxies using within-season ppg quantiles
    data["boom_rate"] = (data["ppg"] > data.groupby("season")["ppg"].transform(lambda s: s.quantile(0.75))).astype(float)
    data["bust_rate"] = (data["ppg"] < data.groupby("season")["ppg"].transform(lambda s: s.quantile(0.25))).astype(float)

    data = data.sort_values(["player","season"]).reset_index(drop=True)
    data["target_next_pts"] = data.groupby("player")["fantasy_points"].shift(-1)

    # drop last season rows without next-season label
    return data.dropna(subset=["target_next_pts"]).reset_index(drop=True)
