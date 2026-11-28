"""Independent monotonic-time watcher for one EXP-TR04 supervised phase (outside the repo).

Polls time.monotonic() every 15 s, logs each poll, and sends SIGINT to the supervisor python
process if the total monotonic elapsed time since T0 exceeds the total cap, the phase exceeds its
cap, the wall clock jumps against monotonic by more than 30 s, or the supervisor stops answering
(supervisor.jsonl not growing). Afterwards it verifies that no tr04.py process remains.
"""
import argparse, json, os, signal, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)  # the phase output dir (its .supervisor sibling is watched)
ap.add_argument("--total-cap", type=float, default=2700.0)
ap.add_argument("--phase-cap", type=float, required=True)
ap.add_argument("--poll", type=float, default=15.0)
a = ap.parse_args()
LOG = open(f"{a.work}/watch_{a.tag}.log", "a", buffering=1)
def log(**kw):
    kw.update(mono=time.monotonic(), wall=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    LOG.write(json.dumps(kw, sort_keys=True) + "\n")
def procs():
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        p = line.strip().split(None, 2)
        if len(p) == 3 and p[0].isdigit():
            rows.append((int(p[0]), int(p[1]), p[2]))
    return rows
def is_tr04(cmd, sub=None):
    """True only for a real python/uv process whose argv has a tr04.py token (not a shell wrapper)."""
    t = cmd.split()
    if not t or not (os.path.basename(t[0]).lower().startswith("python") or os.path.basename(t[0]) == "uv"):
        return False
    return any(x.endswith("training/tr04/tr04.py") for x in t) and (sub is None or sub in t)
def tr04_procs():
    me = os.getpid()
    return [(pid, c) for pid, _, c in procs() if is_tr04(c) and pid != me]
def alive(pid):
    try: os.kill(pid, 0)
    except OSError: return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")

log(event="watcher_started", tag=a.tag, total_cap=a.total_cap, phase_cap=a.phase_cap, poll=a.poll)
t0 = pid_info = None
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        t0 = json.load(open(f"{a.work}/T0.json")); pid_info = json.load(open(f"{a.work}/{a.tag}.pid.json")); break
    except (OSError, ValueError):
        log(event="ready_waiting"); time.sleep(1)
if not (t0 and pid_info):
    log(event="never_armed"); sys.exit(2)
offset0 = t0["wall"] - t0["mono"]; sup_pid = pid_info["pid"]; phase_start = pid_info["mono"]
side = a.out.rstrip("/") + ".supervisor"; sup_log = f"{side}/supervisor.jsonl"
log(event="armed", t0_mono=t0["mono"], phase_start=phase_start, sup_pid=sup_pid)

def supervisors():
    return [pid for pid, _, c in procs() if is_tr04(c, "supervise") and a.out in c.split() and os.path.basename(c.split()[0]) != "uv"]
def intervene(reason):
    log(event="intervene", reason=reason)
    pids = supervisors() or ([sup_pid] if alive(sup_pid) else [])
    for pid in pids:
        try: os.kill(pid, signal.SIGINT); log(event="sigint_sent", pid=pid)
        except OSError as e: log(event="sigint_failed", pid=pid, err=str(e))
    end = time.monotonic() + 120
    while time.monotonic() < end and (supervisors() or alive(sup_pid)): time.sleep(2)
    if supervisors() or alive(sup_pid):
        log(event="supervisor_survived_sigint")
    return reason

reason = None; n = 0
while True:
    time.sleep(a.poll); n += 1
    now = time.monotonic(); elapsed_total = now - t0["mono"]; elapsed_phase = now - phase_start
    drift = (time.time() - now) - offset0
    up = alive(sup_pid) or bool(supervisors())
    # supervisor.jsonl is block-buffered by the supervisor: use the newest of its three files as progress
    mt = []
    for f in (sup_log, f"{side}/child.log", f"{a.out}/events.jsonl"):
        try: mt.append(os.stat(f).st_mtime)
        except OSError: pass
    age = (time.time() - max(mt)) if mt else None
    try: lines = sum(1 for _ in open(f"{a.out}/events.jsonl", "rb"))
    except OSError: lines = 0
    log(event="poll", n=n, elapsed_total=round(elapsed_total, 1), elapsed_phase=round(elapsed_phase, 1),
        clock_drift=round(drift, 2), supervisor_alive=up, sup_log_age=None if age is None else round(age, 1), sup_log_lines=lines)
    if not up: break
    if elapsed_total > a.total_cap: reason = f"total_cap:{elapsed_total:.0f}>{a.total_cap:.0f}"
    elif elapsed_phase > a.phase_cap: reason = f"phase_cap:{elapsed_phase:.0f}>{a.phase_cap:.0f}"
    elif abs(drift) > 30: reason = f"clock_jump:{drift:.1f}s"
    elif age is None and elapsed_phase > 300: reason = "supervisor_log_never_appeared"
    elif age is not None and age > 300: reason = f"supervisor_unresponsive:{age:.0f}s"
    if reason: intervene(reason); break
# post: nothing must remain
time.sleep(2)
left = tr04_procs()
for pid, c in left:
    log(event="survivor_killed", pid=pid, cmd=c[:200])
    try: os.kill(pid, signal.SIGKILL)
    except OSError: pass
time.sleep(2)
log(event="done", intervened=reason, survivors=len(left), remaining=len(tr04_procs()))
