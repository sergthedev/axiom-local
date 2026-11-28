"""Launch one supervised EXP-TR04 phase detached, with the independent watcher armed first."""
import json, os, subprocess, sys, time
work, phase = sys.argv[1], sys.argv[2]
repo = os.path.expanduser("~/tr04-preflight-repo")
SHA = "374ae529ee20a49f2db9995e9860df098de4adec"; MODEL = "$HOME/axiom-models/qwen3-1.7b"
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
cmd = ["uv", "run", "--locked", "--project", "envs/model", "python", "training/tr04/tr04.py", "supervise",
       "--phase", phase, "--model-dir", MODEL, "--export", f"{work}/export", "--out", f"{work}/{phase}",
       "--expect-commit", SHA]
if phase == "reload": cmd += ["--after", f"{work}/train"]
phase_cap = 2700.0 if phase == "train" else 900.0
if phase == "reload":
    t0 = json.load(open(f"{work}/T0.json")); remain = 2700.0 - (time.monotonic() - t0["mono"])
    print("remaining of total:", round(remain, 1))
    if remain < 900: print("REFUSED: less than 900 s of the total remain"); sys.exit(3)
w = subprocess.Popen(["<local-path>", f"{work}/ctl/watch.py", "--work", work, "--tag", phase, "--out", f"{work}/{phase}",
                      "--phase-cap", str(phase_cap)], stdin=subprocess.DEVNULL,
                     stdout=open(f"{work}/watch_{phase}.stdout", "ab"), stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(4)
if w.poll() is not None: print("REFUSED: watcher died", w.returncode); sys.exit(4)
print("watcher pid", w.pid)
if phase == "train":
    mono, wall = time.monotonic(), time.time()
    json.dump({"mono": mono, "wall": wall, "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(wall)),
               "clock": "time.monotonic()"}, open(f"{work}/T0.json", "x"))
p = subprocess.Popen(cmd, cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=open(f"{work}/supervise_{phase}.stdout", "xb"),
                     stderr=open(f"{work}/supervise_{phase}.stderr", "xb"), start_new_session=True)
json.dump({"pid": p.pid, "mono": time.monotonic(), "wall": time.time(), "cmd": cmd}, open(f"{work}/{phase}.pid.json", "x"))
subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True)
print("supervisor launched, pid", p.pid, "watcher pid", w.pid)
