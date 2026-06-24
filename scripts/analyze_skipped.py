import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

RAW_DIR = 'data/option_chain/raw'
HIST_DIR = 'data/historical'

for sym in ['NIFTY', 'BANKNIFTY']:
    files = sorted([f for f in os.listdir(os.path.join(RAW_DIR, sym)) if f.endswith('.csv')])
    idx = pd.read_csv(HIST_DIR + '/' + sym + '.csv', parse_dates=['Date'])
    idx = idx.sort_values('Date').reset_index(drop=True)
    idx['actual_dir'] = (idx['Close'] > idx['Open']).astype(int)
    idx['date_str'] = idx['Date'].dt.strftime('%Y-%m-%d')
    idx['prev_dir'] = idx['actual_dir'].shift(1)
    idx['prev2_dir'] = idx['actual_dir'].shift(2)
    delta = idx['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    idx['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    idx['ema20'] = idx['Close'].ewm(span=20).mean()
    idx['ema50'] = idx['Close'].ewm(span=50).mean()
    idx['above_ema'] = (idx['Close'] > idx['ema20']).astype(int)
    idx['ema_trend'] = (idx['ema20'] > idx['ema50']).astype(int)

    rows = []
    prev_oi_bull = None
    for f_name in files:
        date_str = f_name.replace('.csv', '')
        match = idx[idx['date_str'] == date_str]
        if match.empty:
            continue
        df = pd.read_csv(os.path.join(RAW_DIR, sym, f_name))
        if df.empty:
            continue
        last_ts = df['timestamp'].max()
        snap = df[df['timestamp'] == last_ts]
        ce_chg = snap['ce_chg_oi'].sum()
        pe_chg = snap['pe_chg_oi'].sum()
        tot_ce = snap['ce_oi'].sum()
        tot_pe = snap['pe_oi'].sum()
        pcr = tot_pe / max(tot_ce, 1)
        oi_bull = 1 if pe_chg > ce_chg else 0
        oi_ratio = abs(pe_chg - ce_chg) / max(abs(pe_chg) + abs(ce_chg), 1)
        r = match.iloc[0]
        dow = pd.Timestamp(date_str).weekday()

        step = 50 if sym == 'NIFTY' else 100
        spot = snap.iloc[len(snap) // 2]['strike']
        pains = []
        for k in snap['strike'].values:
            ce_pain = ((k - snap['strike']).clip(lower=0) * snap['ce_oi']).sum()
            pe_pain = ((snap['strike'] - k).clip(lower=0) * snap['pe_oi']).sum()
            pains.append(ce_pain + pe_pain)
        max_pain = snap['strike'].values[int(pd.Series(pains).idxmin())]
        mp_dist = (spot - max_pain) / spot * 100

        is_traded = False
        if oi_bull and pcr > 0.8 and oi_ratio > 0.3:
            is_traded = True
        elif not oi_bull and pcr < 0.8 and oi_ratio > 0.3:
            is_traded = True
        elif oi_ratio > 0.7:
            is_traded = True
        elif oi_bull and dow in (1, 4) and oi_ratio > 0.1:
            is_traded = True
        elif oi_bull and oi_ratio > 0.2 and r.get('prev_dir', 0) == 1 and r.get('above_ema', 0) == 1:
            is_traded = True
        elif not oi_bull and oi_ratio > 0.3 and r.get('prev_dir', 1) == 0 and sym == 'BANKNIFTY':
            is_traded = True

        rows.append({
            'date': date_str, 'pcr': pcr, 'oi_bull': oi_bull, 'oi_ratio': oi_ratio,
            'dow': dow, 'is_traded': is_traded,
            'prev_dir': r.get('prev_dir', np.nan),
            'prev2_dir': r.get('prev2_dir', np.nan),
            'rsi': r.get('rsi', 50),
            'above_ema': r.get('above_ema', 0),
            'ema_trend': r.get('ema_trend', 0),
            'mp_dist': mp_dist,
            'prev_oi_bull': prev_oi_bull if prev_oi_bull is not None else oi_bull,
            'actual_dir': r['actual_dir'],
        })
        prev_oi_bull = oi_bull

    df_r = pd.DataFrame(rows)
    skipped = df_r[~df_r['is_traded']]
    traded = df_r[df_r['is_traded']]
    n_total = len(df_r)

    def wr(s):
        return s.apply(lambda r: r['actual_dir'] if r['oi_bull'] else 1 - r['actual_dir'], axis=1).mean() * 100

    print(f"=== {sym}: {len(skipped)} SKIPPED ({len(skipped)/n_total*100:.0f}%) ===")

    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.3)]:
        sub = skipped[(skipped['oi_ratio'] >= lo) & (skipped['oi_ratio'] < hi)]
        if len(sub) > 0:
            print(f"  ratio {lo:.2f}-{hi:.2f}: {len(sub)}d  OI wr={wr(sub):.0f}%")

    print(f"\n  RESCUE STRATEGIES:")

    # Individual confirmations
    m_ema = ((skipped['oi_bull'] == 1) & (skipped['ema_trend'] == 1)) | ((skipped['oi_bull'] == 0) & (skipped['ema_trend'] == 0))
    m_consec = skipped['oi_bull'] == skipped['prev_oi_bull']
    m_mp = ((skipped['oi_bull'] == 1) & (skipped['mp_dist'] < -0.1)) | ((skipped['oi_bull'] == 0) & (skipped['mp_dist'] > 0.1))
    m_prev = ((skipped['oi_bull'] == 1) & (skipped['prev_dir'] == 1)) | ((skipped['oi_bull'] == 0) & (skipped['prev_dir'] == 0))
    m_mp_loose = ((skipped['oi_bull'] == 1) & (skipped['mp_dist'] < 0)) | ((skipped['oi_bull'] == 0) & (skipped['mp_dist'] > 0))

    for label, mask in [
        ('OI+EMA_trend', m_ema),
        ('OI+consecutive', m_consec),
        ('OI+max_pain', m_mp),
        ('OI+prev_day', m_prev),
        ('OI+EMA+consec', m_ema & m_consec),
        ('OI+EMA+MP', m_ema & m_mp_loose),
        ('OI+consec+MP', m_consec & m_mp_loose),
        ('OI+EMA+prev', m_ema & m_prev),
        ('OI+consec+prev', m_consec & m_prev),
        ('OI+EMA+consec+MP', m_ema & m_consec & m_mp_loose),
        ('OI+EMA+consec+prev', m_ema & m_consec & m_prev),
    ]:
        sub = skipped[mask]
        if len(sub) > 3:
            w = wr(sub)
            freq = (len(traded) + len(sub)) / n_total * 100
            tag = " ***" if w >= 85 else (" **" if w >= 80 else "")
            print(f"  {label}: {w:.0f}% ({len(sub)}d) -> freq {freq:.0f}%{tag}")

    # Score-based: any N of 4
    score = m_ema.astype(int) + m_consec.astype(int) + m_mp_loose.astype(int) + m_prev.astype(int)
    for n_req in [2, 3, 4]:
        sub = skipped[score >= n_req]
        if len(sub) > 3:
            w = wr(sub)
            freq = (len(traded) + len(sub)) / n_total * 100
            tag = " ***" if w >= 85 else (" **" if w >= 80 else "")
            print(f"  ANY {n_req} of 4: {w:.0f}% ({len(sub)}d) -> freq {freq:.0f}%{tag}")

    print(f"\n  ALL skipped OI alone: {wr(skipped):.0f}% ({len(skipped)}d)")
    print(f"  Current traded: {wr(traded):.0f}% on {len(traded)}d ({len(traded)/n_total*100:.0f}%)")
    print()
