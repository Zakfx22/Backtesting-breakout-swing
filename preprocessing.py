import pandas as pd
import numpy as np

def compute_atr(df, period=14):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    df = df.copy()
    df['ATR'] = atr
    return df

def compute_ma200(df, period=200):
    df = df.copy()
    df['MA200'] = df['Close'].rolling(window=period, min_periods=period).mean()
    return df

def compute_adx(df, period=14):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    def wilder_smooth(arr, p):
        res = np.full(len(arr), np.nan)
        if len(arr) >= p:
            res[p-1] = np.sum(arr[:p])
            for i in range(p, len(arr)):
                res[i] = res[i-1] - res[i-1]/p + arr[i]
        return res
    s_tr = wilder_smooth(tr, period)
    s_pdm = wilder_smooth(plus_dm, period)
    s_mdm = wilder_smooth(minus_dm, period)
    with np.errstate(invalid='ignore', divide='ignore'):
        pdi = 100.0 * s_pdm / s_tr
        mdi = 100.0 * s_mdm / s_tr
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-10)
    adx = np.full(n, np.nan)
    start = 2*period - 2
    if start < n:
        adx[start] = np.nanmean(dx[period-1:start+1])
        for i in range(start+1, n):
            adx[i] = (adx[i-1]*(period-1) + dx[i]) / period
    df['ADX'] = adx
    return df

def compute_swing_levels(df, n=5):
    high = df['High'].values
    low = df['Low'].values
    N = len(high)
    pivot_high = np.full(N, np.nan)
    pivot_low = np.full(N, np.nan)
    for i in range(n, N - n):
        left_hi = all(high[i] >= high[i - j] for j in range(1, n+1))
        right_hi = all(high[i] >= high[i + j] for j in range(1, n+1))
        if left_hi and right_hi:
            pivot_high[i] = high[i]
        left_lo = all(low[i] <= low[i - j] for j in range(1, n+1))
        right_lo = all(low[i] <= low[i + j] for j in range(1, n+1))
        if left_lo and right_lo:
            pivot_low[i] = low[i]
    resistance = np.full(N, np.nan)
    support = np.full(N, np.nan)
    cur_res, cur_sup = np.nan, np.nan
    for t in range(N):
        confirmed_idx = t - n
        if confirmed_idx >= 0:
            if not np.isnan(pivot_high[confirmed_idx]):
                cur_res = pivot_high[confirmed_idx]
            if not np.isnan(pivot_low[confirmed_idx]):
                cur_sup = pivot_low[confirmed_idx]
        resistance[t] = cur_res
        support[t] = cur_sup
    df = df.copy()
    df['SwingHigh'] = resistance
    df['SwingLow'] = support
    return df

def preprocess_data(df, swing_n=5):
    df = df.copy()
    print(f"[Preprocessing] Data awal: {len(df)} candle, swing_n={swing_n}")
    df = compute_atr(df)
    df = compute_ma200(df)
    df = compute_adx(df)
    df = compute_swing_levels(df, n=swing_n)
    before = len(df)
    df.dropna(subset=['ATR', 'MA200', 'ADX', 'SwingHigh', 'SwingLow'], inplace=True)
    # Jangan reset index! Biarkan datetime index untuk WFA
    print(f"[Preprocessing] Data valid: {len(df)} candle (drop {before - len(df)})")
    return df