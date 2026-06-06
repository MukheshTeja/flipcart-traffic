"""
Traffic Demand Prediction — Improved Solution v2
=================================================
Improvements over v1:
- LightGBM + XGBoost + CatBoost ensemble
- Spatial neighbour demand features
- More target encoding combinations
- Better hyperparameters
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

TRAIN_PATH = os.environ.get("TRAIN_PATH", "dataset/train.csv")
TEST_PATH = os.environ.get("TEST_PATH", "dataset/test.csv")
SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "dataset/sample_submission.csv")
OUT_PATH = "submission.csv"
SEED = 42

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DEC = {c: i for i, c in enumerate(_B32)}


def gh_decode(gh):
    if not isinstance(gh, str) or not gh:
        return (np.nan, np.nan)
    lat, lon, is_lon = [-90.0, 90.0], [-180.0, 180.0], True
    for ch in gh.lower():
        cd = _DEC.get(ch)
        if cd is None:
            continue
        for mask in (16, 8, 4, 2, 1):
            if is_lon:
                mid = (lon[0] + lon[1]) / 2
                lon[0 if cd & mask else 1] = mid
            else:
                mid = (lat[0] + lat[1]) / 2
                lat[0 if cd & mask else 1] = mid
            is_lon = not is_lon
    return ((lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2)


def mod_of(ts):
    h, m = str(ts).split(":")
    return int(h) * 60 + int(m)


def base_features(df):
    out = pd.DataFrame(index=df.index)
    mod = df["timestamp"].map(mod_of)
    out["mod"] = mod
    out["hour"] = mod // 60
    out["minute"] = mod % 60
    ang = 2 * np.pi * mod / 1440.0
    out["mod_sin"], out["mod_cos"] = np.sin(ang), np.cos(ang)
    out["day"] = pd.to_numeric(df["day"], errors="coerce")
    gh = df["geohash"].astype(str)
    cache = {g: gh_decode(g) for g in gh.dropna().unique()}
    out["gh_lat"] = gh.map(lambda g: cache[g][0])
    out["gh_lon"] = gh.map(lambda g: cache[g][1])
    for p in (4, 5):
        out[f"gh_pre{p}"] = gh.str.slice(0, p).astype("category").cat.codes
    out["gh_code"] = gh.astype("category").cat.codes
    out["NumberofLanes"] = pd.to_numeric(df["NumberofLanes"], errors="coerce")
    out["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
    out["LargeVehicles"] = (
        df["LargeVehicles"].astype(str).str.strip().str.lower()
        .map({"allowed": 1, "not allowed": 0}).fillna(-1)
    )
    out["Landmarks"] = (
        df["Landmarks"].astype(str).str.strip().str.lower()
        .map({"yes": 1, "no": 0}).fillna(-1)
    )
    out["RoadType"] = df["RoadType"].astype("category").cat.codes
    out["Weather"] = df["Weather"].astype("category").cat.codes

    # Extra features
    out["is_rush_hour"] = out["hour"].apply(
        lambda x: 1 if (8 <= x <= 10) or (16 <= x <= 19) else 0
    )
    out["is_night"] = out["hour"].apply(lambda x: 1 if x <= 5 else 0)
    out["lanes_x_landmarks"] = out["NumberofLanes"] * out["Landmarks"]
    out["lanes_x_largevehicles"] = out["NumberofLanes"] * out["LargeVehicles"]

    return out, mod


def kfold_te(ktr, kte, y, folds, smoothing=10.0):
    gm = y.mean()
    oof = pd.Series(np.nan, index=ktr.index)
    for tr, va in folds:
        s = (
            pd.DataFrame({"k": ktr.iloc[tr], "y": y.iloc[tr]})
            .groupby("k")["y"].agg(["mean", "count"])
        )
        sm = (s["mean"] * s["count"] + gm * smoothing) / (s["count"] + smoothing)
        oof.iloc[va] = ktr.iloc[va].map(sm).values
    oof = oof.fillna(gm)
    full = pd.DataFrame({"k": ktr, "y": y}).groupby("k")["y"].agg(["mean", "count"])
    smf = (full["mean"] * full["count"] + gm * smoothing) / (full["count"] + smoothing)
    return oof.values, kte.map(smf).fillna(gm).values


def mkkey(df, cols):
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + "|" + df[c].astype(str)
    return s


def add_neighbour_features(Xtr, Xte, train, test):
    """Add spatial neighbour demand features."""
    print("Adding spatial neighbour features...")
    allg = pd.concat([train["geohash"], test["geohash"]]).astype(str).unique()
    coords = np.array([gh_decode(g) for g in allg])
    valid = ~np.isnan(coords).any(axis=1)
    allg_valid = allg[valid]
    coords_valid = coords[valid]

    k = 6
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(allg_valid))).fit(coords_valid)
    _, idx = nbrs.kneighbors(coords_valid)
    neigh = {g: [allg_valid[j] for j in idx[i][1:]] for i, g in enumerate(allg_valid)}

    ref = train.copy()
    ref["slot"] = ref["timestamp"].map(mod_of) // 15
    slot_demand = (
        ref.groupby([ref["geohash"].astype(str), "slot"])["demand"].mean().to_dict()
    )
    gh_mean = ref.groupby(ref["geohash"].astype(str))["demand"].mean().to_dict()
    gmean = float(pd.to_numeric(train["demand"], errors="coerce").mean())

    def feat(df):
        slot = df["timestamp"].map(mod_of) // 15
        out = []
        for g, s in zip(df["geohash"].astype(str), slot):
            vals = [
                slot_demand.get((nb, s), gh_mean.get(nb, gmean))
                for nb in neigh.get(g, [])
            ]
            out.append(np.mean(vals) if vals else gmean)
        return np.array(out)

    Xtr = Xtr.copy()
    Xte = Xte.copy()
    Xtr["nb_demand"] = feat(train)
    Xte["nb_demand"] = feat(test)
    return Xtr, Xte


def build_design(train, test, y, folds):
    Xtr, mod_tr = base_features(train)
    Xte, mod_te = base_features(test)
    Xtr = Xtr.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)
    yr = y.reset_index(drop=True)
    rt, re = train.copy(), test.copy()
    rt["mod"] = train["timestamp"].map(mod_of)
    re["mod"] = test["timestamp"].map(mod_of)
    rt["hour"] = rt["mod"] // 60
    re["hour"] = re["mod"] // 60
    for k in (4, 5):
        rt[f"gh{k}"] = train["geohash"].astype(str).str.slice(0, k)
        re[f"gh{k}"] = test["geohash"].astype(str).str.slice(0, k)

    specs = [
        ["geohash"],
        ["geohash", "hour"],
        ["geohash", "mod"],
        ["gh4"],
        ["gh4", "hour"],
        ["RoadType"],
        ["Weather"],
        ["geohash", "RoadType"],
        ["gh5", "hour"],
        ["RoadType", "hour"],
        ["Weather", "hour"],
        ["geohash", "Weather"],
        ["gh4", "Weather"],
        ["LargeVehicles", "hour"],
        ["Landmarks", "hour"],
    ]
    for cols in specs:
        otr, ote = kfold_te(
            mkkey(rt, cols).reset_index(drop=True),
            mkkey(re, cols).reset_index(drop=True),
            yr, folds,
        )
        Xtr["te_" + "_".join(cols)] = otr
        Xte["te_" + "_".join(cols)] = ote

    # Recent day-49 level per geohash
    d49 = train[train["day"] == 49]
    gma = train.groupby("geohash")["demand"].mean()
    gm = y.mean()
    rec = d49.groupby("geohash")["demand"].mean()

    def rf(df):
        return df["geohash"].map(rec).fillna(df["geohash"].map(gma)).fillna(gm).values

    Xtr["recent_gh"] = rf(train)
    Xte["recent_gh"] = rf(test)

    # Day-48 profile + lags
    d48 = train[train["day"] == 48].copy()
    d48["mod"] = d48["timestamp"].map(mod_of)
    prof = d48.groupby(["geohash", "mod"])["demand"].mean()
    piv = prof.unstack("mod").sort_index(axis=1)
    pivT = piv.T.sort_index()
    rsT = pivT.rolling(5, center=True, min_periods=1).sum()
    rcT = pivT.notna().rolling(5, center=True, min_periods=1).sum()
    sm_excl = ((rsT - pivT.fillna(0)) / (rcT - pivT.notna()).replace(0, np.nan)).T
    sm_incl = pivT.rolling(5, center=True, min_periods=1).mean().T
    pd_ = prof.to_dict()
    sx = sm_excl.stack().to_dict()
    si = sm_incl.stack().to_dict()

    def lk(d, gs, ms):
        return np.array([d.get((g, m), np.nan) for g, m in zip(gs, ms)])

    gtr, gte = train["geohash"].values, test["geohash"].values
    Xtr["d48_smooth"] = lk(sx, gtr, mod_tr.values)
    Xte["d48_smooth"] = lk(si, gte, mod_te.values)
    for off, nm in [(-15, "prev"), (15, "next"), (-30, "prev2"), (30, "next2")]:
        Xtr[f"d48_{nm}"] = lk(pd_, gtr, mod_tr.values + off)
        Xte[f"d48_{nm}"] = lk(pd_, gte, mod_te.values + off)
    Xtr["d48_trend"] = Xtr["d48_next"] - Xtr["d48_prev"]
    Xte["d48_trend"] = Xte["d48_next"] - Xte["d48_prev"]

    # Day-aware calibration
    gh5_tr = train["geohash"].astype(str).str.slice(0, 5).values
    gh5_te = test["geohash"].astype(str).str.slice(0, 5).values
    night = train.assign(
        mod=train["timestamp"].map(mod_of),
        gh5=train["geohash"].astype(str).str.slice(0, 5),
    )
    night = night[night["mod"] <= 120]
    g48 = night[night["day"] == 48]["demand"].mean()
    reg = night.groupby(["day", "gh5"])["demand"].agg(["mean", "count"])
    reg48 = night[night["day"] == 48].groupby("gh5")["demand"].mean()
    dglob = night.groupby("day")["demand"].mean()
    K = 30.0

    def fac(day, g5):
        den = reg48.get(g5, g48)
        den = den if den > 0 else g48
        if (day, g5) in reg.index:
            mn, cnt = reg.loc[(day, g5), "mean"], reg.loc[(day, g5), "count"]
        else:
            mn, cnt = dglob.get(day, g48), 0
        gr = dglob.get(day, g48) / g48
        return float(np.clip((mn * cnt + (gr * den) * K) / ((cnt + K) * den), 0.4, 3.5))

    facd = {}
    for day, g5 in set(zip(train["day"], gh5_tr)) | set(zip(test["day"], gh5_te)):
        facd[(day, g5)] = fac(day, g5)
    ftr = np.array([facd[(d, g)] for d, g in zip(train["day"], gh5_tr)])
    fte = np.array([facd[(d, g)] for d, g in zip(test["day"], gh5_te)])
    Xtr["cal_factor"] = ftr
    Xte["cal_factor"] = fte
    Xtr["cal_profile"] = np.nan_to_num(Xtr["d48_smooth"].values, nan=0.0) * ftr
    Xte["cal_profile"] = np.nan_to_num(Xte["d48_smooth"].values, nan=0.0) * fte

    # Add neighbour features
    Xtr, Xte = add_neighbour_features(Xtr, Xte, train, test)

    cols = sorted(set(Xtr.columns) & set(Xte.columns))
    return (
        Xtr[cols].replace([np.inf, -np.inf], np.nan),
        Xte[cols].replace([np.inf, -np.inf], np.nan),
    )


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = pd.to_numeric(train["demand"], errors="coerce")
    lo, hi = y.min(), y.max()
    print("train", train.shape, "test", test.shape)

    folds = list(KFold(5, shuffle=True, random_state=SEED).split(train))
    Xtr, Xte = build_design(train, test, y, folds)
    print("features:", Xtr.shape[1])

    pred_lgb = np.zeros(len(test))
    pred_xgb = np.zeros(len(test))
    pred_cat = np.zeros(len(test))
    n = 0

    # LightGBM
    try:
        import lightgbm as lgb
        print("Training LightGBM...")
        for seed in (42, 7, 2024):
            for tr, va in folds:
                m = lgb.LGBMRegressor(
                    n_estimators=3000,
                    learning_rate=0.02,
                    num_leaves=96,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    min_child_samples=40,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                m.fit(Xtr.iloc[tr], np.log1p(y.iloc[tr]))
                pred_lgb += np.expm1(m.predict(Xte))
                n += 1
        pred_lgb /= n
        print("LightGBM done!")
    except Exception as e:
        print("LightGBM failed:", e)

    # XGBoost
    try:
        from xgboost import XGBRegressor
        print("Training XGBoost...")
        n_xgb = 0
        for seed in (42, 7):
            for tr, va in folds:
                m = XGBRegressor(
                    n_estimators=2000,
                    learning_rate=0.02,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=0,
                )
                m.fit(Xtr.iloc[tr], np.log1p(y.iloc[tr]))
                pred_xgb += np.expm1(m.predict(Xte))
                n_xgb += 1
        pred_xgb /= n_xgb
        print("XGBoost done!")
    except Exception as e:
        print("XGBoost failed:", e)

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        print("Training CatBoost...")
        n_cat = 0
        for seed in (42, 7):
            for tr, va in folds:
                m = CatBoostRegressor(
                    iterations=2000,
                    learning_rate=0.02,
                    depth=6,
                    random_seed=seed,
                    verbose=0,
                )
                m.fit(Xtr.iloc[tr], np.log1p(y.iloc[tr]))
                pred_cat += np.expm1(m.predict(Xte))
                n_cat += 1
        pred_cat /= n_cat
        print("CatBoost done!")
    except Exception as e:
        print("CatBoost failed:", e)

    # Weighted ensemble — LightGBM gets most weight
    final = pred_lgb * 0.5 + pred_xgb * 0.3 + pred_cat * 0.2
    final = np.clip(final, lo, hi)

    samp = pd.read_csv(SAMPLE_PATH)
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": final})[
        list(samp.columns)
    ]
    assert sub.shape == (len(test), 2)
    sub.to_csv(OUT_PATH, index=False)
    print("wrote", OUT_PATH, sub.shape, "| mean", round(final.mean(), 4))
    print(sub.head())


if __name__ == "__main__":
    main()
