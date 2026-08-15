

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI HELPER: _calc_max_drawdown()
# Tujuan : Hitung max drawdown dari satu equity curve simulasi.
#          Dipanggil 10.000 kali di dalam loop MC.
# ══════════════════════════════════════════════════════════════════════════════

def _calc_max_drawdown(pnl_sequence):
    """
    Hitung Maximum Drawdown dari urutan PnL.

    Parameters:
        pnl_sequence : array PnL per trade (sudah di-resample oleh MC)

    Returns:
        float : nilai max drawdown (selalu ≤ 0)

    Cara kerja:
        equity[i] = jumlah PnL kumulatif sampai trade ke-i
        peak[i]   = nilai equity tertinggi dari awal sampai trade ke-i
        dd[i]     = equity[i] - peak[i]  → selalu ≤ 0
        MDD       = min(dd)              → titik terendah drawdown
    """
    equity   = np.cumsum(pnl_sequence)
    peak     = np.maximum.accumulate(equity)
    drawdown = equity - peak
    return float(np.min(drawdown))


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA: monte_carlo_simulation()
# Tujuan : Jalankan 10.000 simulasi resampling untuk menghasilkan distribusi
#          probabilistik return dan risiko strategi.
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_simulation(trades, simulations=10000, confidence_level=5):
    """
    Monte Carlo Simulation dengan resampling-with-replacement.

    Parameters:
        trades           : list of float — PnL per trade dari OOS WFA
        simulations      : jumlah iterasi simulasi (default: 10.000)
        confidence_level : persentil untuk worst/best case (default: 5%)
                           → worst = P5, best = P95

    Returns:
        dict berisi distribusi lengkap:
            mean_return      : rata-rata total return dari 10.000 simulasi
            worst_case       : persentil ke-5  (skenario 5% terburuk)
            best_case        : persentil ke-95 (skenario 5% terbaik)
            median_return    : persentil ke-50 (skenario median)
            std_return       : standar deviasi distribusi return
            mean_mdd         : rata-rata Max Drawdown dari semua simulasi
            mdd_95           : Max Drawdown di CI 95% (threshold: < 30%)
            prob_of_ruin     : % simulasi yang berakhir rugi / total return < 0 (threshold: < 20%)
            positive_runs_pct: % simulasi yang menghasilkan return positif

    """

    # ── Guard: tidak ada trade ────────────────────────────────────────────────
    if len(trades) == 0:
        return {
            "mean_return"      : 0.0,
            "worst_case"       : 0.0,
            "best_case"        : 0.0,
            "median_return"    : 0.0,
            "std_return"       : 0.0,
            "mean_mdd"         : 0.0,
            "mdd_95"           : 0.0,
            "prob_of_ruin"     : 0.0,
            "positive_runs_pct": 0.0,
        }

    trades_arr = np.array(trades)
    n_trades   = len(trades_arr)

    print(f"   → {simulations:,} iterasi × {n_trades} trade per simulasi...")

    # ── Jalankan simulasi ─────────────────────────────────────────────────────
    # np.random.choice jauh lebih cepat dari random.choices karena vectorized
    # replace=True = resampling WITH replacement (bootstrap standard)
    total_returns = np.zeros(simulations)
    max_drawdowns = np.zeros(simulations)

    for i in range(simulations):
        # Resample: ambil N trade secara acak dengan penggantian
        # Urutan berbeda setiap iterasi → distribusi skenario berbeda
        sample = np.random.choice(trades_arr, size=n_trades, replace=True)

        total_returns[i] = float(np.sum(sample))
        max_drawdowns[i] = _calc_max_drawdown(sample)

    # ── Distribusi Return ─────────────────────────────────────────────────────
    mean_return   = float(np.mean(total_returns))
    std_return    = float(np.std(total_returns))
    worst_case    = float(np.percentile(total_returns, confidence_level))
    best_case     = float(np.percentile(total_returns, 100 - confidence_level))
    median_return = float(np.percentile(total_returns, 50))

    # ── Distribusi Max Drawdown ───────────────────────────────────────────────
    mean_mdd = float(np.mean(max_drawdowns))

    # MDD_95: drawdown di CI 95% — "95% dari semua skenario, drawdown tidak lebih buruk dari ini"
    # Menggunakan persentil ke-5 dari distribusi MDD (MDD makin negatif = makin buruk)
    # Persentil ke-5 dari MDD memberikan nilai terburuk yang dicapai 95% CI
    mdd_95 = float(np.percentile(max_drawdowns, confidence_level))

    # ── Probability of Ruin ───────────────────────────────────────────────────
    # PoR = proporsi simulasi yang BERAKHIR RUGI (total return < 0)
    # Definisi ini paling intuitif dan tidak bergantung pada skala PnL:
    #   - 0% PoR → semua 10.000 skenario profit ✅
    #   - 16% PoR → 1 dari 6 skenario berakhir rugi ⚠️
    #   - 50%+ PoR → sistem tidak reliable ❌
    # Threshold kelulusan penelitian ini: PoR < 20%
    prob_of_ruin = float(np.mean(total_returns < 0) * 100)

    # ── % simulasi yang profitable ────────────────────────────────────────────
    positive_runs_pct = float(np.mean(total_returns > 0) * 100)

    # ── Convergence check ─────────────────────────────────────────────────────
    # Verifikasi bahwa 10.000 iterasi sudah cukup (standar deviasi stabil)
    mid = simulations // 2
    mean_first_half  = float(np.mean(total_returns[:mid]))
    mean_second_half = float(np.mean(total_returns[mid:]))
    converge_diff    = abs(mean_first_half - mean_second_half)
    converged        = converge_diff < abs(mean_return) * 0.05 if mean_return != 0 else True

    print(f"   → Konvergensi: {'✅ Stabil' if converged else '⚠️  Periksa'} "
          f"(diff={converge_diff:.4f})")

    return {
        "mean_return"      : round(mean_return,    4),
        "worst_case"       : round(worst_case,     4),
        "best_case"        : round(best_case,      4),
        "median_return"    : round(median_return,  4),
        "std_return"       : round(std_return,     4),
        "mean_mdd"         : round(mean_mdd,       4),
        "mdd_95"           : round(mdd_95,         4),
        "prob_of_ruin"     : round(prob_of_ruin,   4),
        "positive_runs_pct": round(positive_runs_pct, 2),
        "converged"        : converged,
        "n_simulations"    : simulations,
    }