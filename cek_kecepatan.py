"""
Jalankan file ini di folder proyek: python cek_kecepatan.py
Akan menunjukkan di mana bottleneck sebenarnya.
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 50)
print("DIAGNOSIS KECEPATAN SISTEM")
print("=" * 50)

# Cek versi strategy.py
try:
    from strategy import generate_signals_breakout
    import inspect
    src = inspect.getsource(generate_signals_breakout)
    if "arr_close = data" in src or "arr_close=data" in src:
        print("✅ strategy.py  → versi CEPAT (numpy)")
    else:
        print("❌ strategy.py  → versi LAMA (pandas.iloc) ← INI MASALAHNYA")
        print("   → Ganti strategy.py dengan file terbaru!")
except Exception as e:
    print(f"❌ strategy.py error: {e}")

# Benchmark backtest 1 call
try:
    from data_loader import load_data
    from preprocessing import preprocess_data
    from strategy import backtest

    print("\nLoading data untuk benchmark...")
    data = load_data("XAUUSD.csv")
    data = preprocess_data(data)

    params = {"window":20,"atr_multiplier":4.5,"rr_ratio":2.0,
              "sl_atr_mult":1.2,"buf_mult":0.01,"min_sideways":3}

    t0 = time.time()
    r  = backtest(data, params)
    t1 = time.time()
    elapsed = t1 - t0

    print(f"\n⏱  1 backtest call : {elapsed:.3f} detik")
    print(f"   Trades dihasilkan: {len(r['trades'])}")

    total_est = elapsed * 54 * 15
    print(f"\n📊 Estimasi total run:")
    print(f"   WFA (54 combo × 15 window) : {elapsed*54*15:.0f} detik ({elapsed*54*15/60:.1f} menit)")
    print(f"   MC + PSI                   : ~5 detik")
    print(f"   GRAND TOTAL                : ~{total_est/60+0.1:.1f} menit")

    if elapsed > 1.0:
        print(f"\n❌ LAMBAT! strategy.py masih versi lama")
        print(f"   Ganti strategy.py dengan versi numpy terbaru")
    elif elapsed > 0.2:
        print(f"\n⚠️  Cukup lambat. Pastikan strategy.py sudah diganti")
    else:
        print(f"\n✅ Kecepatan normal! Seharusnya selesai < 5 menit")

except Exception as e:
    print(f"❌ Error benchmark: {e}")

print("\n" + "=" * 50)
