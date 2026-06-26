import ast
for f in ['engines/edge_tracker.py', 'aos/user_wallet.py']:
    ast.parse(open(f).read())
print('syntax OK')
import engines.edge_tracker as et

# 1) No live data -> unproven defaults (walls trusted, equity half)
et._closed_auto_trades = lambda *a, **k: []
et.invalidate()
print('empty -> wall:', et.trust('wall_selling'), '| equity:', et.trust('intraday_equity'))

# 2) Walls holding up (80% win, profitable); equity LOSING money live
def synth(*a, **k):
    rows = []
    for i in range(15):   # walls: 12 win / 3 loss
        rows.append({'segment': 'options', 'side': 'short',
                     'reason': '[ML auto] OI wall CE sell', 'net_pnl': 100 if i < 12 else -50})
    for i in range(15):   # equity: 5 win / 10 loss -> net negative
        rows.append({'segment': 'equity', 'side': 'long',
                     'reason': '[ML auto] ML rank #1', 'net_pnl': 50 if i < 5 else -100})
    return rows
et._closed_auto_trades = synth
et.invalidate()
import json
print(json.dumps(et.report(), indent=2, default=str))
