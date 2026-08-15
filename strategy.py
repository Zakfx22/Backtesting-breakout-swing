import numpy as np
import pandas as pd

def generate_signals_breakout(df, params):
    atr_mult = params.get('atr_multiplier', 6.0)
    rr = params.get('rr_ratio', 2.0)
    sl_mult = params.get('sl_atr_mult', 0.8)
    buf_mult = params.get('buf_mult', 0.01)
    min_sw = params.get('min_sideways', 3)

    open_arr = df['Open'].values.astype(np.float64)
    high_arr = df['High'].values.astype(np.float64)
    low_arr = df['Low'].values.astype(np.float64)
    close_arr = df['Close'].values.astype(np.float64)
    atr_arr = df['ATR'].values.astype(np.float64)
    ma200_arr = df['MA200'].values.astype(np.float64)
    adx_arr = df['ADX'].values.astype(np.float64)
    sw_high_arr = df['SwingHigh'].values.astype(np.float64)
    sw_low_arr = df['SwingLow'].values.astype(np.float64)

    N = len(df)
    trades = []
    in_trade = False
    trade_dir = None
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_idx = 0
    consec_sw = 0

    for i in range(1, N):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue

        if in_trade:
            hi = high_arr[i]
            lo = low_arr[i]
            if trade_dir == 'long':
                if lo <= sl_price:
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_time': df.index[entry_idx], 'exit_time': df.index[i],
                        'duration': i - entry_idx,
                        'direction': 'long', 'entry': entry_price,
                        'exit': sl_price, 'pnl': sl_price - entry_price,
                        'result': 'loss'
                    })
                    in_trade = False
                elif hi >= tp_price:
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_time': df.index[entry_idx], 'exit_time': df.index[i],
                        'duration': i - entry_idx,
                        'direction': 'long', 'entry': entry_price,
                        'exit': tp_price, 'pnl': tp_price - entry_price,
                        'result': 'win'
                    })
                    in_trade = False
            else:  # short
                if hi >= sl_price:
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_time': df.index[entry_idx], 'exit_time': df.index[i],
                        'duration': i - entry_idx,
                        'direction': 'short', 'entry': entry_price,
                        'exit': sl_price, 'pnl': entry_price - sl_price,
                        'result': 'loss'
                    })
                    in_trade = False
                elif lo <= tp_price:
                    trades.append({
                        'entry_idx': entry_idx, 'exit_idx': i,
                        'entry_time': df.index[entry_idx], 'exit_time': df.index[i],
                        'duration': i - entry_idx,
                        'direction': 'short', 'entry': entry_price,
                        'exit': tp_price, 'pnl': entry_price - tp_price,
                        'result': 'win'
                    })
                    in_trade = False
            continue

        # Cek swing level
        res = sw_high_arr[i]
        sup = sw_low_arr[i]
        if np.isnan(res) or np.isnan(sup):
            continue

        # Sideways detection
        candle_range = high_arr[i] - low_arr[i]
        if candle_range < atr_mult * atr:
            consec_sw += 1
        else:
            consec_sw = 0
        if consec_sw < min_sw:
            continue

        # ADX filter (<25)
        adx = adx_arr[i]
        if np.isnan(adx) or adx >= 25.0:
            consec_sw = 0
            continue

        # Breakout condition
        buf = buf_mult * atr
        cl = close_arr[i]
        op = open_arr[i]
        hi = high_arr[i]
        lo = low_arr[i]
        ma = ma200_arr[i]
        body = abs(cl - op)
        rang = hi - lo if hi > lo else 1e-10

        # LONG
        if (cl > res + buf and body / rang >= 0.60 and cl > ma and not np.isnan(ma)):
            sl_dist = sl_mult * atr
            entry_p = cl
            sl_p = entry_p - sl_dist
            tp_p = entry_p + sl_dist * rr
            in_trade = True
            trade_dir = 'long'
            entry_price = entry_p
            sl_price = sl_p
            tp_price = tp_p
            entry_idx = i
            consec_sw = 0
            continue

        # SHORT
        if (cl < sup - buf and body / rang >= 0.60 and cl < ma and not np.isnan(ma)):
            sl_dist = sl_mult * atr
            entry_p = cl
            sl_p = entry_p + sl_dist
            tp_p = entry_p - sl_dist * rr
            in_trade = True
            trade_dir = 'short'
            entry_price = entry_p
            sl_price = sl_p
            tp_price = tp_p
            entry_idx = i
            consec_sw = 0
            continue

    return trades

def backtest(df, params):
    trades = generate_signals_breakout(df, params)
    if not trades:
        return {'trades': [], 'trades_df': pd.DataFrame(), 'return': 0.0}
    trades_df = pd.DataFrame(trades)
    total_return = trades_df['pnl'].sum() if not trades_df.empty else 0.0
    return {'trades': trades, 'trades_df': trades_df, 'return': total_return}