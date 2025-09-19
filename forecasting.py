import pandas as pd
import numpy as np

try:
    from prophet import Prophet
except Exception:
    Prophet = None

def naive_forecast(hist: pd.DataFrame, horizon=12) -> pd.DataFrame:
    last = hist["value"].iloc[-1] if not hist.empty else 0.0
    future = pd.date_range(
        hist["date"].max() + pd.offsets.MonthBegin(1) if not hist.empty else pd.to_datetime("today"),
        periods=horizon, freq="MS"
    )
    return pd.DataFrame({"date": future, "yhat": last, "yhat_lower": last, "yhat_upper": last})

def fit_prophet(hist: pd.DataFrame, horizon=12) -> pd.DataFrame:
    if hist.empty or hist["value"].sum()==0 or hist["value"].nunique()<=1 or Prophet is None:
        return naive_forecast(hist, horizon=horizon)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode="additive", changepoint_range=0.9)
    dfp = hist.rename(columns={"date":"ds","value":"y"}).copy()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    fc = m.predict(future)
    out = fc[["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"date"})
    return out[out["date"] > hist["date"].max()]

def extrema_mask(y: np.ndarray, window: int = 2):
    n = len(y); is_max = np.zeros(n, dtype=bool); is_min = np.zeros(n, dtype=bool)
    for i in range(n):
        l = max(0, i-window); r = min(n, i+window+1); neigh = y[l:r]
        if len(neigh):
            if y[i] == np.max(neigh) and np.sum(neigh == y[i])==1: is_max[i] = True
            if y[i] == np.min(neigh) and np.sum(neigh == y[i])==1: is_min[i] = True
    return is_max, is_min
