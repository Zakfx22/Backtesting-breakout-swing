import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


OUTPUT_DIR = "output_charts"


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_equity_curve(trades_pnl, filepath):
    """Equity curve - seperti Gambar 3 di artikel Marcel"""
    if not trades_pnl:
        return
    equity = np.cumsum(trades_pnl)
    peak   = np.maximum.accumulate(equity)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Equity Curve & Drawdown - Breakout Sideways XAUUSD H1", fontsize=13, fontweight="bold")

    ax1.plot(equity, color="#2E86AB", linewidth=1.5, label="Equity")
    ax1.fill_between(range(len(equity)), equity, alpha=0.15, color="#2E86AB")
    ax1.plot(peak, color="#A23B72", linewidth=1, linestyle="--", alpha=0.6, label="Peak")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cumulative P/L (pips/points)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    drawdown = equity - peak
    ax2.fill_between(range(len(drawdown)), drawdown, 0, color="#E84855", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Trade #")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📈 Equity curve saved: {filepath}")


def plot_monthly_breakdown(monthly_breakdown, filepath):
    """Monthly P/L - seperti Gambar 1 di artikel Marcel (tabel Jan-Des)"""
    if not monthly_breakdown:
        return

    months = sorted(monthly_breakdown.keys())
    values = [monthly_breakdown[m] for m in months]
    colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in values]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(months, values, color=colors, edgecolor="white", linewidth=0.8)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Monthly P/L Breakdown - Breakout Sideways XAUUSD H1", fontsize=13, fontweight="bold")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("P/L")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(values) * 0.02),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Monthly breakdown saved: {filepath}")


def plot_wfo_windows(wfa_results, filepath):
    """Plot IS vs OOS return per window - kontribusi utama sistem otomatis"""
    if not wfa_results:
        return

    windows = [f"W{w['window']}" for w in wfa_results]
    is_ret  = [w["is_return"]  for w in wfa_results]
    oos_ret = [w["oos_return"] for w in wfa_results]

    x = np.arange(len(windows))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, is_ret,  width, label="In-Sample",     color="#3498DB", alpha=0.8)
    ax.bar(x + width/2, oos_ret, width, label="Out-of-Sample", color="#E67E22", alpha=0.8)

    ax.set_title("Walk Forward Optimization: IS vs OOS Return per Window", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(windows)
    ax.set_ylabel("Return")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   🔄 WFO chart saved: {filepath}")


def plot_monte_carlo(mc_results, trades_pnl, filepath):
    """Distribusi hasil Monte Carlo Simulation"""
    if not trades_pnl:
        return

    import random
    sim_results = []
    for _ in range(1000):
        shuffled = random.choices(trades_pnl, k=len(trades_pnl))
        sim_results.append(sum(shuffled))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sim_results, bins=50, color="#9B59B6", alpha=0.7, edgecolor="white")
    ax.axvline(mc_results.get("mean_return", 0), color="#2ECC71", linewidth=2, label=f"Mean: {mc_results.get('mean_return',0):.2f}")
    ax.axvline(mc_results.get("worst_case", 0),  color="#E74C3C", linewidth=2, linestyle="--", label=f"Worst 5%: {mc_results.get('worst_case',0):.2f}")
    ax.axvline(mc_results.get("best_case", 0),   color="#3498DB", linewidth=2, linestyle="--", label=f"Best 95%: {mc_results.get('best_case',0):.2f}")
    ax.axvline(0, color="gray", linewidth=1, linestyle=":")

    ax.set_title("Monte Carlo Simulation - Distribusi 1000 Skenario", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Return")
    ax.set_ylabel("Frekuensi")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   🎲 Monte Carlo chart saved: {filepath}")


def generate_report(metrics, wfa_results=None, trades_pnl=None, mc_results=None):
    _ensure_dir()

    # ── Simpan chart ──────────────────────────────────────────
    if trades_pnl:
        plot_equity_curve(trades_pnl, f"{OUTPUT_DIR}/equity_curve.png")
        if mc_results:
            plot_monte_carlo(mc_results, trades_pnl, f"{OUTPUT_DIR}/monte_carlo.png")

    monthly = metrics.get("monthly_breakdown", {})
    if monthly:
        plot_monthly_breakdown(monthly, f"{OUTPUT_DIR}/monthly_breakdown.png")

    if wfa_results:
        plot_wfo_windows(wfa_results, f"{OUTPUT_DIR}/wfo_windows.png")

    # ── Print report ke console ───────────────────────────────
    baseline = {
        "total_trade"        : 132,
        "winrate"            : 0.7273,
        "consecutive_profit" : 9,
        "consecutive_loss"   : 3,
        "avg_rr_ratio"       : 1.14,
    }

    print("\n" + "=" * 65)
    print("         FINAL REPORT - BREAKOUT SIDEWAYS XAUUSD H1")
    print("=" * 65)
    print(f"{'Metrik':<28} {'Sistem Otomatis':>15}  {'Baseline Manuel':>15}")
    print("-" * 65)

    def row(label, key, fmt=".4f", baseline_key=None):
        val = metrics.get(key, "N/A")
        bval = baseline.get(baseline_key or key, "N/A")
        if isinstance(val, float):
            val_str = format(val, fmt)
        else:
            val_str = str(val)
        if isinstance(bval, float):
            bval_str = format(bval, fmt)
        else:
            bval_str = str(bval)
        print(f"{label:<28} {val_str:>15}  {bval_str:>15}")

    row("Total Trade",              "total_trade",         ".0f")
    row("Total Return",             "total_return",        ".4f")
    row("Winrate / Probabilitas",   "winrate",             ".4f")
    row("Expectancy (per trade)",   "expectancy",          ".4f")
    row("Max Drawdown",             "max_drawdown",        ".4f")
    row("Sharpe Ratio (annualized)","sharpe_ratio",        ".4f")
    row("Calmar Ratio",             "calmar_ratio",        ".4f")
    row("Profit Factor",            "profit_factor",       ".3f")
    row("Avg Risk:Reward",          "avg_rr_ratio",        ".3f")
    row("Consecutive Profit (max)", "consecutive_profit",  ".0f")
    row("Consecutive Loss (max)",   "consecutive_loss",    ".0f")
    row("Avg Duration (candles)",   "avg_duration_candles",".2f")
    print("-" * 65)
    row("MC Mean Return",           "mc_mean",             ".4f")
    row("MC Median Return (P50)",   "mc_median",           ".4f")
    row("MC Worst Case (5th %)",    "mc_worst",            ".4f")
    row("MC Best Case (95th %)",    "mc_best",             ".4f")
    row("MC Max DD CI-95%",         "mc_mdd_95",           ".4f")
    row("Probability of Ruin",      "prob_of_ruin",        ".2f")
    print("-" * 65)
    row("Walk Forward Efficiency",  "wfe",                 ".4f")
    row("WFA Consistency Score",    "consistency_score",   ".1f")
    row("Degradation Ratio (%)",    "degradation_ratio",   ".4f")
    row("Parameter Stability (PSI)","psi",                 ".4f")
    row("Robustness Score (0-100)", "robustness_score",    ".2f")
    print("=" * 65)

    # Monthly breakdown
    monthly = metrics.get("monthly_breakdown", {})
    if monthly:
        print("\n📅 MONTHLY P/L BREAKDOWN:")
        print(f"  {'Bulan':<12} {'P/L':>10}")
        print("  " + "-" * 24)
        total = 0
        for month in sorted(monthly.keys()):
            val = monthly[month]
            total += val
            sign = "✅" if val >= 0 else "❌"
            print(f"  {month:<12} {val:>10.2f}  {sign}")
        print(f"  {'TOTAL':<12} {total:>10.2f}")

    mc_pos = mc_results.get("positive_runs_pct", 0) if mc_results else 0
    n_sim = mc_results.get("n_simulations", 10000) if mc_results else 10000
    print(f"\n🎲 MC Positive Runs: {mc_pos:.1f}% dari {n_sim:,} simulasi")
    print(f"\n📁 Charts saved to: ./{OUTPUT_DIR}/")
    print("=" * 65)