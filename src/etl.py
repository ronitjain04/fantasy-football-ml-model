import os
from pathlib import Path
import pandas as pd

DEFAULT_PASSING = os.getenv("PLAYER_PASSING", "data/passing.csv")
DEFAULT_RUSHING = os.getenv("PLAYER_RUSHING", "data/rushing.csv")
DEFAULT_RECEIVING = os.getenv("PLAYER_RECEIVING", "data/receiving.csv")
DEFAULT_FANTASY = os.getenv("FANTASY_RANKS", "data/fantasy_ranks.csv")

def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _find_col(df: pd.DataFrame, candidates) -> str | None:
    cols = list(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for c in cols:
            if isinstance(c, str) and cand in c:
                return c
    return None

def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ren = {}
    for src, tgt in [
        (_find_col(df, ["player","player_name","name"]), "player"),
        (_find_col(df, ["season","year"]), "season"),
        (_find_col(df, ["team","tm"]), "team"),
        (_find_col(df, ["position","pos"]), "position"),
    ]:
        if src and src != tgt:
            ren[src] = tgt
    if ren:
        df = df.rename(columns=ren)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "player" in df.columns:
        df["player"] = df["player"].astype(str).str.strip()
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip()
    if "position" in df.columns:
        df["position"] = df["position"].astype(str).str.upper().str.strip()
    return df

def _select(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = {}
    for canon, cands in mapping.items():
        col = _find_col(df, cands)
        if col is not None:
            out[canon] = df[col]
    sel = pd.DataFrame(out)
    for k in ("player","season","team","position"):
        if k in df.columns and k not in sel.columns:
            sel[k] = df[k]
    return sel

def load_all(
    passing_path: str = DEFAULT_PASSING,
    rushing_path: str = DEFAULT_RUSHING,
    receiving_path: str = DEFAULT_RECEIVING,
    fantasy_path: str = DEFAULT_FANTASY,
) -> pd.DataFrame:
    passing = _normalize_keys(_read_csv(passing_path))
    rushing = _normalize_keys(_read_csv(rushing_path))
    receiving = _normalize_keys(_read_csv(receiving_path))
    fantasy = _normalize_keys(_read_csv(fantasy_path))

    pass_map = {
        "pass_yds": ["pass_yds","passing_yards","yds","y"],
        "pass_td": ["pass_td","passing_tds","td"],
        "interceptions": ["int","ints","interceptions"],
        "games": ["g","games","gp"],
    }
    rush_map = {
        "rush_yds": ["rush_yds","rushing_yards","yds","y"],
        "rush_td": ["rush_td","rushing_tds","td"],
        "fumbles": ["fumbles","fum","fmb"],
        "games": ["g","games","gp"],
    }
    recv_map = {
        "rec_yds": ["rec_yds","receiving_yards","yds","y"],
        "rec_td": ["rec_td","receiving_tds","td"],
        "receptions": ["rec","receptions"],
        "games": ["g","games","gp"],
    }
    fant_map = {
        "adp": ["adp","avg_draft_position","average_draft_position"],
        "rank": ["rank","overall_rank","ecr"],
    }

    psel = _select(passing, pass_map) if not passing.empty else pd.DataFrame()
    rsel = _select(rushing, rush_map) if not rushing.empty else pd.DataFrame()
    rcv  = _select(receiving, recv_map) if not receiving.empty else pd.DataFrame()
    fsel = _select(fantasy, fant_map) if not fantasy.empty else pd.DataFrame()

    tables = [t for t in [psel, rsel, rcv, fsel] if not t.empty]
    if not tables:
        raise RuntimeError("No usable tables found. Provide at least one of passing/rushing/receiving/fantasy CSVs.")

    df = tables[0]
    for t in tables[1:]:
        keys = [k for k in ("player","season") if k in df.columns and k in t.columns]
        how = "outer" if keys else "left"
        df = df.merge(t, on=keys if keys else None, how=how, suffixes=("", "_dup"))

    # Drop merged duplicate columns that end with _dup if original exists
    dup_cols = [c for c in df.columns if c.endswith("_dup") and c[:-4] in df.columns]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    # Coerce numerics for all non-key columns
    for c in df.columns:
        if c not in ("player","team","position","season"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing season if entirely absent
    if "season" not in df.columns:
        df["season"] = pd.to_numeric(df.get("year", pd.Series([2024]*len(df))), errors="coerce").fillna(2024).astype(int)

    return df
