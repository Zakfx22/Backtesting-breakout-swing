import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data
from preprocessing import preprocess_data
from walk_forward import (
    walk_forward_analysis, combine_oos_trades, combine_oos_trades_df,
    get_best_parameters, calc_wfa_metrics, print_wfa_summary,
)
from monte_carlo import monte_carlo_simulation
from performance import evaluate_performance
from robustness import parameter_stability_test, robustness_score
from report import generate_report
from report_html import generate_html_report

def main():
    print("\nStarting Breakout Swing Backtesting System v2.1 (Final)\n")

    # ─── STEP 1 ──────────────────────────────────────────────────────────
    print("[1/6] Loading & preprocessing data...")
    data = load_data("XAUUSD.csv")
    # Tambahkan parameter swing_n (disesuaikan dengan laporan)
    data = preprocess_data(data, swing_n=5)
    # Informasi tambahan agar sesuai dengan laporan
    print(f"   → Metode S/R: Swing High/Low (swing_n=5)")
    print(f"   → Data valid  : {len(data)} candle (drop {17722 - len(data)} karena NaN)\n")

    # ─── STEP 2 ──────────────────────────────────────────────────────────
    print("[2/6] Running Walk Forward Analysis...")
    print("   (Rolling window: IS=3 bulan, OOS=1 bulan)")
    # Panggil WFA (tambahkan parameter verbose=False)
    wfa_results = walk_forward_analysis(data, is_months=3, oos_months=1, verbose=False)
    print_wfa_summary(wfa_results)
    wfa_metrics = calc_wfa_metrics(wfa_results)
    aggregated_trades = combine_oos_trades(wfa_results)
    aggregated_trades_df = combine_oos_trades_df(wfa_results)
    print(f"   Total OOS trades terkumpul: {len(aggregated_trades)}\n")

    # ─── STEP 3 ──────────────────────────────────────────────────────────
    print("[3/6] Running Monte Carlo Simulation (10.000 iterasi)...")
    mc_results = monte_carlo_simulation(aggregated_trades, simulations=10000)
    # Cetak hasil MC dengan format sesuai laporan
    print(f"   Mean={mc_results['mean_return']:.4f} | Median={mc_results['median_return']:.4f}")
    print(f"   Worst(5%)={mc_results['worst_case']:.4f} | Best(95%)={mc_results['best_case']:.4f}")
    print(f"   MDD_95={mc_results['mdd_95']:.4f} | PoR={mc_results['prob_of_ruin']:.2f}%")
    print(f"   Positive runs: {mc_results['positive_runs_pct']:.1f}%\n")

    # ─── STEP 4 ──────────────────────────────────────────────────────────
    print("[4/6] Evaluating Performance Metrics...")
    metrics = evaluate_performance(aggregated_trades, mc_results, trades_df=aggregated_trades_df)
    metrics["mc_median"] = mc_results.get("median_return", 0)
    metrics["mc_mdd_95"] = mc_results.get("mdd_95", 0)
    metrics["prob_of_ruin"] = mc_results.get("prob_of_ruin", 0)
    # Opsional: cetak ringkasan performa di terminal
    print(f"   Total Trade    : {metrics['total_trade']}")
    print(f"   Total Return   : {metrics['total_return']:.4f}")
    print(f"   Winrate        : {metrics['winrate']:.4f}")
    print(f"   Profit Factor  : {metrics['profit_factor']:.3f}\n")

    # ─── STEP 5 ──────────────────────────────────────────────────────────
    print("[5/6] Calculating Robustness Metrics...")
    best_param = get_best_parameters(wfa_results)
    best_param["window"] = 20   # dummy untuk PSI
    psi = parameter_stability_test(data, best_param, __import__("strategy").backtest)
    metrics["wfe"] = wfa_metrics.get("wfe", 0)
    metrics["consistency_score"] = wfa_metrics.get("consistency_score", 0)
    metrics["degradation_ratio"] = wfa_metrics.get("avg_degradation", 0)
    metrics["psi"] = psi
    metrics["robustness_score"] = robustness_score(metrics)
    print(f"   WFE={metrics['wfe']:.4f} | Consistency={metrics['consistency_score']:.1f}%")
    print(f"   Degradation={metrics['degradation_ratio']:.2f}% | PSI={psi:.4f}")
    print(f"   Robustness Score = {metrics['robustness_score']:.2f}/100\n")

    # ─── STEP 6 ──────────────────────────────────────────────────────────
    print("[6/6] Generating Report & Charts...")
    generate_report(metrics, wfa_results=wfa_results,
                    trades_pnl=aggregated_trades, mc_results=mc_results)
    generate_html_report(metrics, wfa_results=wfa_results,
                         trades_pnl=aggregated_trades, mc_results=mc_results)

    # Tambahan informasi output file
    print("   📈 Equity curve saved: output_charts/equity_curve.png")
    print("   🎲 Monte Carlo chart saved: output_charts/monte_carlo.png")
    print("   📊 Monthly breakdown saved: output_charts/monthly_breakdown.png")
    print("   🔄 WFO chart saved: output_charts/wfo_windows.png")
    print("\n=================================================================")
    print("   🌐 HTML Report saved: D:\\project TA\\breakout_swing\\backtesting_report.html")
    print("   🌐 Dashboard dibuka di browser otomatis.")
    print("=================================================================")
    print("\nSystem Finished Successfully!")

if __name__ == "__main__":
    main()