import json, os, sys
w = open(os.path.expanduser("~/.tr04_work_path")).read().strip(); ph = sys.argv[1] if len(sys.argv) > 1 else "train"
def last(path, n=1):
    try: return open(path).read().splitlines()[-n:]
    except OSError: return []
for l in last(f"{w}/watch_{ph}.log", 1): d = json.loads(l); print("WATCH", {k: d.get(k) for k in ("event","n","elapsed_total","clock_drift","sup_log_age","wall")})
ev = [json.loads(l) for l in last(f"{w}/{ph}/events.jsonl", 400) if l.strip()]
from collections import Counter
print("event kinds:", dict(Counter(e.get("kind") for e in ev)))
for e in ev[-3:]: print("EV", json.dumps({k: v for k, v in e.items() if k not in ("preflight","report")})[:400])
for l in last(f"{w}/{ph}.supervisor/supervisor.jsonl", 1):
    d = json.loads(l); print("SUP", json.dumps({k: v for k, v in d.items() if k != "kind"})[:500])
for f in ("STOP.json", "COMPLETE.json"):
    p = f"{w}/{ph}.supervisor/{f}"
    if os.path.exists(p): print(f, "EXISTS", open(p).read()[:600])
