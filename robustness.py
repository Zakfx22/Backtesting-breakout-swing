import numpy as np
from strategy import backtest

DEFAULT_PARAMS = {
    "window"        : 20,
    "atr_multiplier": 5.0,
    "rr_ratio"      : 2.0,
    "sl_atr_mult"   : 0.5,
    "buf_mult"      : 0.1
}


def walk_forward_efficiency(total_is_return, total_oos_return):
    """
    WFE = OOS / IS
    >= 0.8 : sistem robust, tidak overfit
    = 1.0  : OOS sama baiknya dengan IS (ideal)
    """
    if total_is_return == 0:
        return 0.0
    return round(float(total_oos_return / total_is_return), 4)


def degradation_ratio(total_is_return, total_oos_return):
    """
    DR = (IS - OOS) / |IS|
    Mendekati 0 = performa IS dan OOS konsisten
    """
    if total_is_return == 0:
        return 0.0
    return round(float((total_is_return - total_oos_return) / abs(total_is_return)), 4)


def parameter_stability_test(data, best_param, backtest_function):
    """
    PSI: Uji sensitivitas terhadap perubahan parameter ±10%.
    Skor 1.0 = sangat stabil, perubahan parameter kecil tidak
    mengubah hasil secara signifikan.
    """
    base_result = backtest_function(data, best_param)
    base_return = base_result.get("return", 0)

    if base_return == 0:
        return 1.0

    perturbed = []
    for key in ["window", "atr_multiplier", "rr_ratio", "sl_atr_mult", "buf_mult"]:
        for delta in [-0.1, +0.1]:
            p = dict(best_param)
            if key == "window":
                p[key] = max(5, int(p[key] * (1 + delta)))
            else:
                p[key] = p[key] * (1 + delta)
            try:
                r = backtest_function(data, p)
                perturbed.append(r.get("return", 0))
            except Exception:
                perturbed.append(base_return)

    arr = np.array(perturbed)
    variation = np.std(arr) / abs(base_return)
    psi = max(0.0, 1.0 - variation)
    return round(float(psi), 4)


def robustness_score(metrics):
    """
    Robustness Score (0-100):
    Skor gabungan dari semua dimensi kualitas sistem.

    Formula didesain untuk strategi HIGH RR (risk:reward tinggi, winrate rendah):
    - Tidak menghukum WR rendah secara berlebihan — WR 33% dengan RR 3.0
      secara matematis setara dengan WR 50% dengan RR 1.0
    - Expectancy dan PF lebih mencerminkan edge strategi daripada WR saja
    - WFE dan Degradation menilai robustness temporal (anti-overfitting)

    Bobot komponen:
        Profit Factor    20%  — edge nyata strategi (lebih objektif dari WR)
        Sharpe Ratio     20%  — risk-adjusted return harian
        WFE              15%  — OOS mempertahankan IS performance
        PSI              15%  — stabilitas parameter
        MC Score         15%  — distribusi hasil Monte Carlo
        Expectancy       10%  — expected value per trade
        Degradation       5%  — konsistensi IS vs OOS
    """
    sharpe        = metrics.get("sharpe_ratio", 0)
    wfe           = metrics.get("wfe", 0)
    psi           = metrics.get("psi", 0)
    dr            = metrics.get("degradation_ratio", 0)
    mc_worst      = metrics.get("mc_worst", 0)
    mc_positive   = metrics.get("mc_mean", 0)
    profit_factor = metrics.get("profit_factor", 0)
    expectancy    = metrics.get("expectancy", 0)
    por           = metrics.get("prob_of_ruin", 100)

    # Profit Factor score: PF 1.0=0, PF 1.5=50%, PF 2.0=100%
    pf_score  = min(max((profit_factor - 1.0) / 1.0, 0), 1.0) if profit_factor > 0 else 0.0

    # Sharpe score: 0=0, 1.0=50%, 2.0=100%
    sh_score  = min(max(sharpe / 2.0, 0), 1.0)

    # WFE score: capped di 1.0
    wfe_score = min(max(wfe, 0), 1.0)

    # PSI score: langsung dipakai
    psi_score = min(max(psi, 0), 1.0)

    # MC score: kombinasi mc_worst > 0 dan PoR < 20%
    mc_worst_ok = 1.0 if mc_worst >= 0 else max(0.0, 1.0 + mc_worst / 200)
    por_score   = max(0.0, 1.0 - por / 20.0)   # PoR 0%=1.0, PoR 20%=0.0
    mc_score    = (mc_worst_ok + por_score) / 2

    # Expectancy score: positif = ada edge
    exp_score = min(max(expectancy / 5.0, 0), 1.0) if expectancy > 0 else 0.0

    # Degradation score: makin kecil makin bagus
    dr_score  = max(0.0, 1.0 - abs(dr) / 40.0)   # 0%=1.0, 40%=0.0

    score = (
        0.20 * pf_score   +
        0.20 * sh_score   +
        0.15 * wfe_score  +
        0.15 * psi_score  +
        0.15 * mc_score   +
        0.10 * exp_score  +
        0.05 * dr_score
    ) * 100

    return round(float(score), 4)