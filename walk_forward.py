import itertools
import pandas as pd
import numpy as np
from strategy import backtest

PARAM_GRID = {
    "atr_multiplier": [4.5, 6.0, 8.0],
    "rr_ratio":       [1.5, 2.0, 2.5],
    "sl_atr_mult":    [0.8, 1.2],
    "buf_mult":       [0.01],
    "min_sideways":   [3],
}
DEFAULT_PARAMS = {
    "atr_multiplier": 4.5,
    "rr_ratio":       2.0,
    "sl_atr_mult":    1.2,
    "buf_mult":       0.01,
    "min_sideways":   3,
}

def _profit_factor(trades_df):
    if trades_df is None or trades_df.empty:
        return 0.0
    wins = trades_df[trades_df["result"] == "win"]["pnl"].sum()
    losses = abs(trades_df[trades_df["result"] == "loss"]["pnl"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(wins / losses, 4)

def _winrate(trades_df):
    if trades_df is None or trades_df.empty:
        return 0.0
    return round((trades_df["result"] == "win").mean(), 4)

def grid_search_is(is_data, verbose=False):
    best_pf = -1.0
    best_params = dict(DEFAULT_PARAMS)
    n_combos = 0
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        n_combos += 1
        try:
            result = backtest(is_data, params)
        except Exception:
            continue
        tdf = result.get("trades_df", pd.DataFrame())
        if tdf.empty or len(tdf) < 3:
            continue
        pf = _profit_factor(tdf)
        if pf > best_pf:
            best_pf = pf
            best_params = dict(params)
        if best_pf >= 5.0:
            break
    return best_params, best_pf, n_combos

def walk_forward_analysis(data, is_months=3, oos_months=1, verbose=False):
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("index harus DatetimeIndex")
    months = (data.index[-1].year - data.index[0].year)*12 + (data.index[-1].month - data.index[0].month) + 1
    candles_per_month = max(int(len(data)/months), 300)
    is_size = is_months * candles_per_month
    oos_size = oos_months * candles_per_month
    print(f"Candles/bulan: {candles_per_month}, IS: {is_size}, OOS: {oos_size}")

    total_len = len(data)
    wfa_results = []
    start = 0
    window_num = 0
    while True:
        is_end = start + is_size
        oos_end = is_end + oos_size
        if oos_end > total_len:
            break
        is_data = data.iloc[start:is_end]
        oos_data = data.iloc[is_end:oos_end]
        if len(is_data) < 100 or len(oos_data) < 20:
            start += oos_size
            continue
        window_num += 1
        best_params, _, n_combos = grid_search_is(is_data, verbose=verbose)

        is_res = backtest(is_data, best_params)
        oos_res = backtest(oos_data, best_params)

        is_tdf = is_res.get("trades_df", pd.DataFrame())
        oos_tdf = oos_res.get("trades_df", pd.DataFrame())
        oos_trades = oos_res.get("trades", [])
        oos_pnls = [t['pnl'] for t in oos_trades]

        wfa_results.append({
            "window": window_num,
            "is_start": is_data.index[0].strftime("%Y-%m-%d"),
            "is_end": is_data.index[-1].strftime("%Y-%m-%d"),
            "oos_start": oos_data.index[0].strftime("%Y-%m-%d"),
            "oos_end": oos_data.index[-1].strftime("%Y-%m-%d"),
            "best_params": best_params,
            "is_pf": _profit_factor(is_tdf),
            "oos_pf": _profit_factor(oos_tdf),
            "is_return": is_res.get("return", 0),
            "oos_return": oos_res.get("return", 0),
            "n_is_trades": len(is_tdf),
            "n_oos_trades": len(oos_tdf),
            "oos_trades": oos_pnls,           # list of pnl floats
            "oos_trades_df": oos_tdf,
            "grid_combos": n_combos,
        })
        start += oos_size
    print(f"WFA selesai: {window_num} windows")
    return wfa_results

def combine_oos_trades(wfa_results):
    all_pnls = []
    for w in wfa_results:
        all_pnls.extend(w.get("oos_trades", []))
    return all_pnls

def combine_oos_trades_df(wfa_results):
    dfs = []
    for w in wfa_results:
        df = w.get("oos_trades_df")
        if df is not None and not df.empty:
            df = df.copy()
            df["wf_window"] = w["window"]
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_best_parameters(wfa_results):
    if not wfa_results:
        return DEFAULT_PARAMS
    best = max(wfa_results, key=lambda w: w.get("oos_pf", 0))
    return best.get("best_params", DEFAULT_PARAMS)

def calc_wfa_metrics(wfa_results):
    if not wfa_results:
        return {}
    MAX_PF = 5.0
    is_pfs = [min(w["is_pf"], MAX_PF) for w in wfa_results if w["is_pf"] > 0]
    oos_pfs = [min(w["oos_pf"], MAX_PF) for w in wfa_results if w["n_oos_trades"] > 0]
    if not is_pfs or not oos_pfs:
        return {"wfe":0, "consistency_score":0, "avg_degradation":0, "_no_trades":True}
    avg_is = np.mean(is_pfs)
    avg_oos = np.mean(oos_pfs)
    wfe = avg_oos/avg_is if avg_is>0 else 0
    n_prof = sum(1 for p in oos_pfs if p>1)
    consistency = n_prof/len(oos_pfs)*100
    degrad = []
    for w in wfa_results:
        if w["is_pf"]>0 and w["n_oos_trades"]>0:
            d = (min(w["is_pf"],MAX_PF) - min(w["oos_pf"],MAX_PF))/min(w["is_pf"],MAX_PF)
            degrad.append(d)
    avg_degrad = np.mean(degrad)*100 if degrad else 0
    best = max(wfa_results, key=lambda w: w["oos_pf"])
    worst = min(wfa_results, key=lambda w: w["oos_pf"])
    return {
        "wfe": round(wfe,4),
        "consistency_score": round(consistency,2),
        "avg_degradation": round(avg_degrad,2),
        "avg_is_pf": round(avg_is,4),
        "avg_oos_pf": round(avg_oos,4),
        "n_windows": len(wfa_results),
        "n_profitable_oos": n_prof,
        "best_window": best["window"],
        "best_oos_pf": best["oos_pf"],
        "worst_window": worst["window"],
        "worst_oos_pf": worst["oos_pf"],
    }

def print_wfa_summary(wfa_results):
    metrics = calc_wfa_metrics(wfa_results)
    if metrics.get("_no_trades"):
        print("Tidak ada trade")
        return
    print("\n"+"="*72)
    print("WALK FORWARD ANALYSIS SUMMARY")
    print("="*72)
    print(f"{'Win':>4} {'IS Period':>22} {'OOS Period':>22} {'IS_PF':>6} {'OOS_PF':>6} {'N_OOS':>5}")
    for w in wfa_results:
        prof = "✅" if w["oos_pf"]>1 else "❌"
        print(f"{w['window']:>4} {w['is_start']} → {w['is_end']} {w['oos_start']} → {w['oos_end']} {w['is_pf']:6.3f} {w['oos_pf']:6.3f} {w['n_oos_trades']:5} {prof}")
    print("-"*68)
    print(f"{'AVG':>4} {'':>22} {'':>22} {metrics['avg_is_pf']:6.3f} {metrics['avg_oos_pf']:6.3f}")
    print("="*72)
    print(f"Walk Forward Efficiency: {metrics['wfe']:.4f}")
    print(f"Consistency Score: {metrics['consistency_score']:.1f}% ({metrics['n_profitable_oos']}/{metrics['n_windows']})")
    print(f"Avg PF Degradation: {metrics['avg_degradation']:.1f}%")