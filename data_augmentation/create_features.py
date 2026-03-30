import numpy as np
import pandas as pd

def load_synth_data(path: str = 'data/return_synth.npy'):
    """Load synth data.

    Args:
        path: Filesystem path to the dataset/file.

    Returns:
        Computed output(s) produced by the function.
    """
    return np.load(path)

def _roll(series: pd.Series, window: int, func: str) -> pd.Series:
    """Apply a lagged rolling statistic.

    Args:
        series: Univariate series on which rolling statistics are computed.
        window: Rolling window length for feature extraction.
        func: Rolling statistic to apply: sum, std, or mean.

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

    Args:
        df: DataFrame containing the target column to binarize.
        target_col: Name of the target return column to convert into sign labels.

    Returns:
        None.
    """
    series = df[target_col]
    df[target_col] = (series > 0).astype(int)

def load_real_data(date_col: str, path: str = 'data/training_data.csv', ):
    """Load real data.

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

    Args:
        path_returns: Single synthetic return path (time x assets).
        path_id: Identifier of the synthetic path in the output table.
        cum_horizons: Horizons used to compute lagged cumulative returns.
        vol_horizons: Horizons used to compute lagged rolling volatility.
        lag_zscore_horizons: Horizons used for lagged z-score features.
        lag_market_return_horizons: Horizons for lagged market-average return features.

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

    Args:
        sp_returns: Long-format real return DataFrame.
        returns_matrix: Wide matrix of returns aligned with sp_returns timestamps.
        cum_horizons: Horizons used to compute lagged cumulative returns.
        vol_horizons: Horizons used to compute lagged rolling volatility.
        lag_zscore_horizons: Horizons used for lagged z-score features.
        lag_market_return_horizons: Horizons for lagged market-average return features.

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

    Args:
        pred: Predicted probabilities or scores for next-day direction.
        real: Realized next-day returns used as ground truth.
        day_start: Index offset where backtest evaluation starts.
        normalise: If True, annualize Sharpe-like metrics.
        periods_per_year: Number of periods used for annualization.

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
