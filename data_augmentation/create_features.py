import numpy as np
import pandas as pd


def load_synth_data(path: str = 'data/return_synth.npy'):
    """Load synth data.

    Load synth data. This routine is part of the SBBTS workflow and related utilities.

    Args:
        path: Filesystem path to the dataset/file.

    Returns:
        Computed output(s) produced by the function.
    """
    return np.load(path)


def _roll(series: pd.Series, window: int, func: str) -> pd.Series:
    """ roll.

     roll. This routine is part of the SBBTS workflow and related utilities.

    Args:
        series: Input parameter `series` used by this computation.
        window: Rolling window length for feature extraction.
        func: Input parameter `func` used by this computation.

    Returns:
        Computed output(s) produced by the function.
    """
    shifted = series.shift(1)  # dont tke the return of the day no leakage

    if func == "sum":
        return shifted.rolling(window).sum()
    if func == "std":
        return shifted.rolling(window).std(ddof=0)
    if func == "mean":
        return shifted.rolling(window).mean()
    raise ValueError("func must be 'sum', 'std' or 'mean'")


def binarize_target(df: pd.DataFrame, target_col: str) -> None:
    """Binarize target.

    Binarize target. This routine is part of the SBBTS workflow and related utilities.

    Args:
        df: Input parameter `df` used by this computation.
        target_col: Input parameter `target_col` used by this computation.

    Returns:
        None.
    """
    series = df[target_col]
    df[target_col] = (series > 0).astype(int)


def load_real_data(date_col: str, path: str = 'data/training_data.csv', ):
    """Load real data.

    Load real data. This routine is part of the SBBTS workflow and related utilities.

    Args:
        date_col: Name of the date column in the raw CSV file.
        path: Filesystem path to the dataset/file.

    Returns:
        Computed output(s) produced by the function.
    """
    sp = pd.read_csv(path).sort_values(by=date_col)
    returns_matrix = sp.copy()[
        [f for f in sp.columns if f != date_col]].to_numpy()  # exclude the date columns and retain only the returns
    sp["perimeter.DateTime"] = pd.to_datetime(sp[date_col])
    sp.drop(columns=date_col, axis=1, inplace=True)

    sp = sp.melt(  # go from wide format to long
        id_vars="perimeter.DateTime",
        var_name="perimeter.instr_id",
        value_name="target.target",
    ).sort_values(by=["perimeter.DateTime", 'perimeter.instr_id'])

    return sp, returns_matrix


def make_path_dataframe(
        path_returns: np.ndarray,
        path_id: int = 0,
        cum_horizons: tuple = (5, 10, 21, 63, 126, 252),
        vol_horizons: tuple = (5, 10, 21, 63, 126, 252),
        lag_zscore_horizons: tuple = (3, 5, 10, 21),
        lag_market_return_horizons: tuple = (1,),
) -> pd.DataFrame:
    """Make path dataframe.

    Make path dataframe. This routine is part of the SBBTS workflow and related utilities.

    Args:
        path_returns: Input parameter `path_returns` used by this computation.
        path_id: Input parameter `path_id` used by this computation.
        cum_horizons: Input parameter `cum_horizons` used by this computation.
        vol_horizons: Input parameter `vol_horizons` used by this computation.
        lag_zscore_horizons: Input parameter `lag_zscore_horizons` used by this computation.
        lag_market_return_horizons: Input parameter `lag_market_return_horizons` used by this computation.

    Returns:
        Computed output(s) produced by the function.
    """
    if path_returns.ndim != 2:
        raise ValueError("path_returns must be 2‑D (n_days, n_instr)")

    n_days, n_instr = path_returns.shape

    day_idx = np.arange(1, n_days + 1)
    instr_idx = np.arange(n_instr)
    day_grid, instr_grid = np.meshgrid(day_idx, instr_idx, indexing="ij")

    df = pd.DataFrame(
        {
            "perimeter.path_id": path_id,
            "perimeter.Date": day_grid.ravel(),
            "perimeter.instr_id": instr_grid.ravel(),
            "target.target": path_returns.ravel(),
        }
    )

    grp = df.groupby("perimeter.instr_id", group_keys=False)
    market_series = pd.Series(path_returns.mean(axis=1), index=day_idx, name="market_return")

    for i in lag_market_return_horizons:
        assert i > 0, "lag values must be > 0 to respect temporal causality"
        df[f"feature.return_t-{i}_market"] = market_series.shift(i).reindex(df["perimeter.Date"]).values

    market_feat = pd.DataFrame(
        {"perimeter.Date": day_idx}
    )
    for h in lag_zscore_horizons:
        market_feat[f"feature.mkt_cumret_{h}d"] = _roll(market_series, h, "sum").values
        market_feat[f"feature.mkt_vol_{h}d"] = _roll(market_series, h, "std").values
        market_feat[f"feature.mkt_mean_{h}d"] = _roll(market_series, h, "mean").values
    df = df.merge(market_feat, on="perimeter.Date", how="left")

    for h in cum_horizons:
        df[f"feature.cum_ret_{h}d"] = grp["target.target"].apply(
            lambda s: _roll(s, h, "sum")
        )

    for h in vol_horizons:
        df[f"feature.vol_{h}d"] = grp["target.target"].apply(
            lambda s: _roll(s, h, "std")
        )

    lag_ret = grp["target.target"].shift(1)  # return at t‑1, per instrument
    for h in lag_zscore_horizons:
        mu = grp["target.target"].apply(lambda s: _roll(s, h, "mean"))
        sigma = grp["target.target"].apply(lambda s: _roll(s, h, "std")).replace(0, np.nan)
        df[f"feature.ret_t-1_zscore_{h}d"] = (lag_ret - mu) / sigma

    df.sort_values(
        by=["perimeter.Date", "perimeter.instr_id"], ignore_index=True, inplace=True
    )
    df['extra.Return'] = df['target.target'].copy()
    binarize_target(df, target_col='target.target')
    feature_cols = (
            [f"feature.cum_ret_{h}d" for h in cum_horizons]
            + [f"feature.vol_{h}d" for h in vol_horizons]
            + [f"feature.ret_t-1_zscore_{h}d" for h in lag_zscore_horizons]
            + [f"feature.mkt_cumret_{h}d" for h in lag_zscore_horizons]
            + [f"feature.mkt_vol_{h}d" for h in lag_zscore_horizons]
            + [f"feature.mkt_mean_{h}d" for h in lag_zscore_horizons]
            + [f'feature.return_t-{i}_market' for i in lag_market_return_horizons]
    )
    cols = ["perimeter.path_id", "perimeter.Date", "perimeter.instr_id"] + feature_cols + ["target.target",
                                                                                           'extra.Return']
    return df[cols]


def make_real_dataframe(
        sp_returns: pd.DataFrame,
        returns_matrix: np.ndarray,
        cum_horizons: tuple = (5, 10, 21, 63, 126, 252),
        vol_horizons: tuple = (5, 10, 21, 63, 126, 252),
        lag_zscore_horizons: tuple = (3, 5, 10, 21),
        lag_market_return_horizons: tuple = (1,),
) -> pd.DataFrame:
    """Make real dataframe.

    Make real dataframe. This routine is part of the SBBTS workflow and related utilities.

    Args:
        sp_returns: Input parameter `sp_returns` used by this computation.
        returns_matrix: Input parameter `returns_matrix` used by this computation.
        cum_horizons: Input parameter `cum_horizons` used by this computation.
        vol_horizons: Input parameter `vol_horizons` used by this computation.
        lag_zscore_horizons: Input parameter `lag_zscore_horizons` used by this computation.
        lag_market_return_horizons: Input parameter `lag_market_return_horizons` used by this computation.

    Returns:
        Computed output(s) produced by the function.
    """
    if sp_returns.ndim != 2:
        raise ValueError("path_returns must be 2‑D (n_days, n_instr)")

    n_days, n_instr = len(sp_returns['perimeter.DateTime'].unique()), len(sp_returns['perimeter.instr_id'].unique())
    day_idx = np.arange(1, n_days + 1)
    instr_idx = np.arange(n_instr)
    day_grid, instr_grid = np.meshgrid(day_idx, instr_idx, indexing="ij")

    df = pd.DataFrame(
        {
            "perimeter.path_id": "REAL SP500",
            "perimeter.Date": day_grid.ravel(),
            "perimeter.DateTime": sp_returns['perimeter.DateTime'].values,
            "perimeter.instr_id": instr_grid.ravel(),
            "perimeter.instr_name": sp_returns['perimeter.instr_id'].values,
            "target.target": sp_returns['target.target'].values,
        }
    )

    grp = df.groupby("perimeter.instr_id", group_keys=False)
    market_series = pd.Series(returns_matrix.mean(axis=1), index=day_idx, name="market_return")

    for i in lag_market_return_horizons:
        assert i > 0, "lag values must be > 0 to respect temporal causality"
        df[f"feature.return_t-{i}_market"] = market_series.shift(i).reindex(df["perimeter.Date"]).values

    market_feat = pd.DataFrame(
        {"perimeter.Date": day_idx}
    )
    for h in lag_zscore_horizons:  # use the same horizons you asked for
        market_feat[f"feature.mkt_cumret_{h}d"] = _roll(market_series, h, "sum").values
        market_feat[f"feature.mkt_vol_{h}d"] = _roll(market_series, h, "std").values
        market_feat[f"feature.mkt_mean_{h}d"] = _roll(market_series, h, "mean").values
    df = df.merge(market_feat, on="perimeter.Date", how="left")

    for h in cum_horizons:
        df[f"feature.cum_ret_{h}d"] = grp["target.target"].apply(
            lambda s: _roll(s, h, "sum")
        )

    for h in vol_horizons:
        df[f"feature.vol_{h}d"] = grp["target.target"].apply(
            lambda s: _roll(s, h, "std")
        )

    lag_ret = grp["target.target"].shift(1)  # return at t‑1, per instrument
    for h in lag_zscore_horizons:
        mu = grp["target.target"].apply(lambda s: _roll(s, h, "mean"))
        sigma = grp["target.target"].apply(lambda s: _roll(s, h, "std")).replace(0, np.nan)
        df[f"feature.ret_t-1_zscore_{h}d"] = (lag_ret - mu) / sigma

    df.sort_values(
        by=["perimeter.Date", "perimeter.instr_id"], ignore_index=True, inplace=True
    )

    feature_cols = (
            [f"feature.cum_ret_{h}d" for h in cum_horizons]
            + [f"feature.vol_{h}d" for h in vol_horizons]
            + [f"feature.ret_t-1_zscore_{h}d" for h in lag_zscore_horizons]
            + [f"feature.mkt_cumret_{h}d" for h in lag_zscore_horizons]
            + [f"feature.mkt_vol_{h}d" for h in lag_zscore_horizons]
            + [f"feature.mkt_mean_{h}d" for h in lag_zscore_horizons]
            + [f'feature.return_t-{i}_market' for i in lag_market_return_horizons]
    )
    cols = ["perimeter.DateTime", "perimeter.path_id", "perimeter.instr_name", "perimeter.Date",
            "perimeter.instr_id"] + feature_cols + ["target.target"]

    assert df.groupby(['perimeter.instr_id', 'perimeter.instr_name']).count().shape[
               0] == n_instr, "Instrument index are not unique"

    return df[cols]


def trading_strat(pred, real, day_start, normalise=True, periods_per_year=252):
    """Trading strat.

    Trading strat. This routine is part of the SBBTS workflow and related utilities.

    Args:
        pred: Input parameter `pred` used by this computation.
        real: Input parameter `real` used by this computation.
        day_start: Input parameter `day_start` used by this computation.
        normalise: Input parameter `normalise` used by this computation.
        periods_per_year: Input parameter `periods_per_year` used by this computation.

    Returns:
        Computed output(s) produced by the function.
    """
    pred = np.asarray(pred).ravel()
    real = np.asarray(real).ravel()
    if pred.shape != real.shape:
        raise ValueError("pred and real must have identical shape")

    day_start = np.asarray(day_start, dtype=int)
    if day_start.ndim != 1:
        raise ValueError("day_start must be a one‑dimensional sequence")
    if day_start.size == 0 or day_start[0] != 0:
        raise ValueError("day_start must start with 0")
    if np.any(np.diff(day_start) < 0):
        raise ValueError("day_start must be non‑decreasing")
    if day_start[-1] > pred.size:
        raise ValueError("the last start index must be smaller or equal than the total length")

    n_days = day_start.size
    daily_ret = np.empty(n_days - 1, dtype=float)

    for i in range(n_days - 1):
        s = day_start[i]
        e = day_start[i + 1]
        w = 2.0 * pred[s:e] - 1.0
        if normalise:
            d = np.abs(w).sum()
            w = 0.0 if d == 0 else w / d
        daily_ret[i] = np.dot(w, real[s:e])

    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(periods_per_year)
    return daily_ret.mean(), daily_ret.std(), sharpe
