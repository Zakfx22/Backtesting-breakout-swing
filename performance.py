import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 1: calc_consecutive()
# Tujuan : Hitung max consecutive profit dan max consecutive loss.
#          Metrik ini langsung diperbandingkan dengan prior study
#          Tjitrayudha & Kosasih (2024): consec profit=9, consec loss=3.
# ══════════════════════════════════════════════════════════════════════════════

def calc_consecutive(trades_df):
    """
    Hitung streak terpanjang: menang berturut-turut dan kalah berturut-turut.

    Parameters:
        trades_df : DataFrame dengan kolom 'result' ('win' atau 'loss')

    Returns:
        (max_profit_streak, max_loss_streak) : tuple of int

    Contoh:
        Urutan: win win loss win win win loss loss
        → max_profit_streak = 3  (win win win)
        → max_loss_streak   = 2  (loss loss)
    """
    if trades_df.empty:
        return 0, 0

    max_profit_streak = 0
    max_loss_streak   = 0
    cur_p = 0
    cur_l = 0

    for result in trades_df["result"]:
        if result == "win":
            cur_p += 1
            cur_l  = 0
            max_profit_streak = max(max_profit_streak, cur_p)
        else:
            cur_l += 1
            cur_p  = 0
            max_loss_streak = max(max_loss_streak, cur_l)

    return max_profit_streak, max_loss_streak


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 2: calc_monthly_breakdown()
# Tujuan : Breakdown P/L per bulan — untuk dibandingkan dengan tabel
#          monthly P/L di prior study Tjitrayudha & Kosasih (2024).
# ══════════════════════════════════════════════════════════════════════════════

def calc_monthly_breakdown(trades_df):
    """
    Hitung total PnL per bulan dari semua trade.

    Parameters:
        trades_df : DataFrame dengan kolom 'entry_time' dan 'pnl'

    Returns:
        dict: {"2022-01": 5.23, "2022-02": -3.11, ...}
    """
    if trades_df.empty:
        return {}

    trades_df = trades_df.copy()
    trades_df["month"] = pd.to_datetime(trades_df["entry_time"]).dt.strftime("%Y-%m")
    monthly = trades_df.groupby("month")["pnl"].sum().round(2).to_dict()
    return monthly


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 3: calc_profit_factor()
# Tujuan : Hitung Profit Factor = total profit / total loss.
#          Threshold kelulusan penelitian ini: PF ≥ 1.3
#          PF = 1.0 → breakeven; PF < 1.0 → sistem merugi
# ══════════════════════════════════════════════════════════════════════════════

def calc_profit_factor(trades_df):
    """
    Profit Factor = Σ(pnl trade menang) / |Σ(pnl trade kalah)|

    Interpretasi:
        PF = 1.5 → setiap $1 yang hilang, sistem menghasilkan $1.5
        PF = 0.8 → setiap $1 yang hilang, sistem hanya menghasilkan $0.8
    """
    if trades_df.empty:
        return 0.0
    wins   = trades_df[trades_df["result"] == "win"]["pnl"].sum()
    losses = abs(trades_df[trades_df["result"] == "loss"]["pnl"].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return round(float(wins / losses), 3)


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 4: calc_avg_rr()
# Tujuan : Hitung rata-rata Risk:Reward Ratio aktual per trade.
#          Berbeda dari RR yang di-set di parameter — ini adalah RR yang
#          benar-benar terjadi setelah exit (bisa lebih kecil karena slippage).
# ══════════════════════════════════════════════════════════════════════════════

def calc_avg_rr(trades_df):
    """
    Avg RR = rata-rata pnl trade menang / rata-rata |pnl trade kalah|

    Prior study nilai: 1.14 (setiap 1 unit risk, dapat 1.14 unit reward)
    """
    if trades_df.empty:
        return 0.0
    wins   = trades_df[trades_df["result"] == "win"]["pnl"]
    losses = abs(trades_df[trades_df["result"] == "loss"]["pnl"])
    if len(losses) == 0 or losses.mean() == 0:
        return 0.0
    return round(float(wins.mean() / losses.mean()), 3)


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 5: calc_sharpe_ratio()  ← DIPERBAIKI TOTAL
# Tujuan : Hitung Sharpe Ratio yang benar dari equity curve harian.
#
# MASALAH SEBELUMNYA:
#   mean(pnl_per_trade) / std(pnl_per_trade) × sqrt(6048)
#   → menghasilkan 11–14 karena satuan PnL adalah USD/oz, bukan % ekuitas
#   → angka tidak bermakna dan tidak bisa dibandingkan dengan literatur
#
# PERBAIKAN:
#   1. Bangun equity curve dari PnL kumulatif
#   2. Resample ke harian (1 titik per hari)
#   3. Hitung return harian dari equity harian
#   4. Sharpe = mean_daily_return / std_daily_return × sqrt(252)
#
# Target realistis: 0.5–2.0 (strategi terbaik dunia ~2–3)
# ══════════════════════════════════════════════════════════════════════════════

def calc_sharpe_ratio(trades_df, risk_free_daily=0.0):
    """
    Sharpe Ratio annualized dari equity curve harian.

    Parameters:
        trades_df        : DataFrame dengan kolom 'exit_time' dan 'pnl'
        risk_free_daily  : risk-free rate harian (default 0, sesuai praktik
                           umum untuk short-term trading)

    Returns:
        float : Sharpe Ratio annualized

    Rumus:
        SR = (E[R_daily] - Rf) / σ[R_daily] × √252

    Kenapa √252?
        252 = jumlah hari trading dalam setahun (standar industri keuangan)
    """
    if trades_df is None or trades_df.empty:
        return 0.0

    try:
        df = trades_df.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("exit_time")

        # Bangun equity curve: PnL kumulatif berdasarkan waktu exit
        df["cum_pnl"] = df["pnl"].cumsum()

        # Set index ke exit_time untuk resample
        df = df.set_index("exit_time")

        # Resample ke harian: ambil nilai equity terakhir setiap hari
        # fillna(method='ffill') → hari tanpa trade mewarisi equity sebelumnya
        daily_equity = df["cum_pnl"].resample("1D").last().ffill()

        # Hitung return harian: perubahan equity dari hari ke hari
        daily_returns = daily_equity.diff().dropna()

        # Butuh minimal 5 hari untuk kalkulasi bermakna
        if len(daily_returns) < 5 or daily_returns.std() == 0:
            return 0.0

        mean_r = float(daily_returns.mean())
        std_r  = float(daily_returns.std())

        sharpe = (mean_r - risk_free_daily) / std_r * (252 ** 0.5)
        return round(sharpe, 4)

    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 6: calc_calmar_ratio()  ← BARU
# Tujuan : Calmar Ratio = Total Return / |Max Drawdown|
#          Mengukur seberapa besar return yang dihasilkan per unit drawdown.
#          Melengkapi Sharpe Ratio untuk evaluasi risk-adjusted performance.
# ══════════════════════════════════════════════════════════════════════════════

def calc_calmar_ratio(total_return, max_drawdown):
    """
    Calmar Ratio = Total Return / |Max Drawdown|

    Interpretasi:
        Calmar = 1.0 → return = drawdown (moderat)
        Calmar > 2.0 → return jauh lebih besar dari drawdown (bagus)
        Calmar < 0.5 → drawdown terlalu besar relatif terhadap return

    Catatan: max_drawdown dalam parameter sudah negatif, jadi di-abs().
    """
    if max_drawdown == 0:
        return 0.0
    return round(float(total_return / abs(max_drawdown)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 7: calc_expectancy()  ← BARU
# Tujuan : Expectancy = rata-rata PnL yang diharapkan per trade.
#          Ini adalah metrik paling fundamental untuk menilai edge strategi.
#          Nilai positif = strategi punya edge; negatif = tidak ada edge.
# ══════════════════════════════════════════════════════════════════════════════

def calc_expectancy(trades_df):
    """
    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

    Contoh:
        Win Rate = 45%, Avg Win = 10, Avg Loss = 5
        → Expectancy = (0.45 × 10) - (0.55 × 5) = 4.5 - 2.75 = +1.75
        → Setiap trade, rata-rata menghasilkan +1.75 unit

    Interpretasi:
        Expectancy > 0 → strategi profitable secara matematis ✅
        Expectancy = 0 → breakeven
        Expectancy < 0 → tidak ada edge, strategi merugi jangka panjang ❌
    """
    if trades_df is None or trades_df.empty:
        return 0.0

    wins   = trades_df[trades_df["result"] == "win"]["pnl"]
    losses = trades_df[trades_df["result"] == "loss"]["pnl"]

    if len(trades_df) == 0:
        return 0.0

    win_rate  = len(wins) / len(trades_df)
    loss_rate = 1 - win_rate
    avg_win   = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0  # negatif

    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    return round(expectancy, 4)


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI 8: evaluate_performance()
# Tujuan : Fungsi utama yang dipanggil main.py. Menghitung semua metrik
#          sekaligus dan mengembalikannya sebagai satu dict.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_performance(trades, monte_carlo_results, trades_df=None):
    """
    Evaluasi performa lengkap dari hasil backtesting.

    Parameters:
        trades               : list of float — PnL per trade (dari WFA OOS)
        monte_carlo_results  : dict hasil monte_carlo_simulation()
        trades_df            : DataFrame lengkap semua trade (opsional)

    Returns:
        dict berisi semua metrik performa siap ditampilkan di report
    """
    # ── Jika tidak ada trade sama sekali ────────────────────────────────────
    if len(trades) == 0:
        return {
            "total_trade"          : 0,
            "total_return"         : 0.0,
            "winrate"              : 0.0,
            "max_drawdown"         : 0.0,
            "sharpe_ratio"         : 0.0,
            "calmar_ratio"         : 0.0,
            "expectancy"           : 0.0,
            "profit_factor"        : 0.0,
            "avg_rr_ratio"         : 0.0,
            "consecutive_profit"   : 0,
            "consecutive_loss"     : 0,
            "avg_duration_candles" : 0.0,
            "monthly_breakdown"    : {},
            "mc_mean"              : 0.0,
            "mc_worst"             : 0.0,
            "mc_best"              : 0.0,
            "mc_median"            : 0.0,
            "mc_mdd_95"            : 0.0,
            "prob_of_ruin"         : 0.0,
        }

    arr = np.array(trades)

    # ── Metrik dasar ─────────────────────────────────────────────────────────
    total_trade  = len(arr)
    total_return = float(np.sum(arr))
    winrate      = float(np.sum(arr > 0) / total_trade)

    # ── Equity Curve & Max Drawdown ──────────────────────────────────────────
    equity      = np.cumsum(arr)
    peak        = np.maximum.accumulate(equity)
    drawdown    = equity - peak
    max_drawdown = float(np.min(drawdown))

    # ── Sharpe Ratio (dari equity harian) ← DIPERBAIKI ──────────────────────
    sharpe_ratio = calc_sharpe_ratio(trades_df) if trades_df is not None else 0.0

    # ── Calmar Ratio ─────────────────────────────────────────────────────────
    calmar_ratio = calc_calmar_ratio(total_return, max_drawdown)

    # ── Metrik yang membutuhkan trades_df ────────────────────────────────────
    consecutive_profit = 0
    consecutive_loss   = 0
    profit_factor      = 0.0
    avg_rr             = 0.0
    monthly_breakdown  = {}
    avg_duration       = 0.0
    expectancy         = 0.0

    if trades_df is not None and not trades_df.empty:
        consecutive_profit, consecutive_loss = calc_consecutive(trades_df)
        profit_factor     = calc_profit_factor(trades_df)
        avg_rr            = calc_avg_rr(trades_df)
        monthly_breakdown = calc_monthly_breakdown(trades_df)
        avg_duration      = round(float(trades_df["duration"].mean()), 2)
        expectancy        = calc_expectancy(trades_df)

    # ── Monte Carlo results ─────────────────────────────────────────────────
    mc_mean   = round(monte_carlo_results.get("mean_return", 0), 4)
    mc_worst  = round(monte_carlo_results.get("worst_case", 0), 4)
    mc_best   = round(monte_carlo_results.get("best_case", 0), 4)
    mc_median = round(monte_carlo_results.get("median_return", 0), 4)
    mc_mdd_95 = round(monte_carlo_results.get("mdd_95", 0), 4)
    prob_ruin = round(monte_carlo_results.get("prob_of_ruin", 0), 2)

    return {
        "total_trade"          : total_trade,
        "total_return"         : round(total_return, 4),
        "winrate"              : round(winrate, 4),
        "max_drawdown"         : round(max_drawdown, 4),
        "sharpe_ratio"         : sharpe_ratio,
        "calmar_ratio"         : calmar_ratio,
        "expectancy"           : expectancy,
        "profit_factor"        : profit_factor,
        "avg_rr_ratio"         : avg_rr,
        "consecutive_profit"   : consecutive_profit,
        "consecutive_loss"     : consecutive_loss,
        "avg_duration_candles" : avg_duration,
        "monthly_breakdown"    : monthly_breakdown,
        "mc_mean"              : mc_mean,
        "mc_worst"             : mc_worst,
        "mc_best"              : mc_best,
        "mc_median"            : mc_median,
        "mc_mdd_95"            : mc_mdd_95,
        "prob_of_ruin"         : prob_ruin,
    }