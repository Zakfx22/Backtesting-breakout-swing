import json
import pandas as pd
import numpy as np
from preprocessing import preprocess_data
from strategy import backtest

# 1. Load data
print("Loading data...")
df = pd.read_csv("XAUUSD.csv", delimiter='\t', header=0)
df.columns = [col.strip('<>') for col in df.columns]
df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
df.set_index('Datetime', inplace=True)
df.drop(['DATE', 'TIME'], axis=1, inplace=True)
rename_map = {'OPEN':'Open','HIGH':'High','LOW':'Low','CLOSE':'Close'}
df.rename(columns=rename_map, inplace=True)
df = df[['Open','High','Low','Close']]

# 2. Preprocessing
df = preprocess_data(df, swing_n=5)

# 3. Parameter terbaik
params = {
    "atr_multiplier": 6.0,
    "rr_ratio": 2.5,
    "sl_atr_mult": 1.2,
    "buf_mult": 0.01,
    "min_sideways": 3,
}

# 4. Backtest
result = backtest(df, params)
trades_df = result['trades_df']
if trades_df.empty:
    print("Tidak ada trade.")
    exit()

print(f"Total trades: {len(trades_df)}")

# 5. Kumpulkan data trade
trade_list = []
for idx, trade in trades_df.iterrows():
    entry_idx = trade['entry_idx']
    exit_idx = trade['exit_idx']
    start = max(0, entry_idx - 40)
    end = min(len(df)-1, exit_idx + 40)
    sub_df = df.iloc[start:end+1].copy()
    sub_df.reset_index(inplace=True)
    ohlc_data = {
        'time': [str(t) for t in sub_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M')],
        'open': [float(x) for x in sub_df['Open']],
        'high': [float(x) for x in sub_df['High']],
        'low': [float(x) for x in sub_df['Low']],
        'close': [float(x) for x in sub_df['Close']]
    }
    atr_entry = float(df.iloc[entry_idx]['ATR'])
    sl_dist = params['sl_atr_mult'] * atr_entry
    direction = trade['direction']
    entry_price = float(trade['entry'])
    if direction == 'long':
        sl_price = entry_price - sl_dist
        tp_price = entry_price + sl_dist * params['rr_ratio']
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - sl_dist * params['rr_ratio']
    trade_list.append({
        'id': int(idx),
        'direction': direction,
        'entry_time': df.index[entry_idx].strftime('%Y-%m-%d %H:%M'),
        'exit_time': df.index[exit_idx].strftime('%Y-%m-%d %H:%M'),
        'entry_price': entry_price,
        'exit_price': float(trade['exit']),
        'sl_price': sl_price,
        'tp_price': tp_price,
        'pnl': float(trade['pnl']),
        'result': trade['result'],
        'ohlc': ohlc_data,
        'entry_idx_rel': entry_idx - start,
        'exit_idx_rel': exit_idx - start
    })

# 6. Simpan ke JSON untuk embedded
trades_json = json.dumps(trade_list, ensure_ascii=False)

# 7. Buat HTML dengan data langsung
html = f"""<!DOCTYPE html>
...
    const trades = {trades_json};
...
</html>"""

with open("all_trades_viewer.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ HTML siap: all_trades_viewer.html")
print("   Buka file tersebut di browser. Jika masih error, coba jalankan 'python -m http.server' lalu buka http://localhost:8000/all_trades_viewer.html")