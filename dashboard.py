"""
Control panel for the Trackmania AI.   python dashboard.py [--port 8765]

Shows live training stats and starts/stops training from the browser. Needs no extra packages.
"""
import argparse
import json
import os
import subprocess

import numpy as np
import sys
import threading
import time
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config as cfgmod

ROOT = cfgmod.ROOT
PROC = {"p": None, "kind": None}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def alive():
    return PROC["p"] is not None and PROC["p"].poll() is None


def kill_proc(p=None):
    """Kill a training process (and its car workers). Pass the process to kill a specific one."""
    p = PROC["p"] if p is None else p
    if p is None or p.poll() is not None:
        return
    if sys.platform == "win32":       # also kills the car worker processes
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        p.terminate()


def downsample(items, n=2500):
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


class Store:
    """Reads runs/<run>/events.jsonl incrementally."""

    def __init__(self):
        self.path, self.off = None, 0
        self.clear()

    def clear(self):
        self.off = 0
        self.laps, self.tests, self.upds = [], [], []     # laps: (lap number, ms, greedy, t) of full laps only
        self.glaps = []                                      # IQN exploitation laps (no random actions): [n, s]
        self.n_all, self.n_mid = 0, 0
        self.rbs, self.ends = [], []
        self.targets = []                                    # reward target changes: ghost rebuilds, section switches
        self.sess_t0, self.sess_lap0, self.last_t = None, 0, None
        self.prof = None
        self.map_laps, self.map_n, self.map_tests = {}, {}, {}
        self.notice = None

    def refresh(self, path):
        if path != self.path:
            self.path = path
            self.clear()
        try:
            size = path.stat().st_size
        except OSError:
            self.clear()
            return
        if size < self.off:
            self.clear()
        with open(path, "rb") as f:
            f.seek(self.off)
            data = f.read()
        end = data.rfind(b"\n")
        if end < 0:
            return
        self.off += end + 1
        for line in data[:end].splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            self.last_t = e.get("t")
            k = e.get("k")
            if k == "start":
                self.sess_t0, self.sess_lap0 = e["t"], self.n_all
                self.notice = e.get("notice")
            elif k == "lap":
                if e.get("pool"):                      # pool cars of a map run: not this map's laps
                    continue
                if e.get("map"):                       # base run: per-map stats
                    ml = self.map_laps.setdefault(e["map"], [])
                    ml.append(e.get("ms"))
                    del ml[:-100]
                    self.map_n[e["map"]] = self.map_n.get(e["map"], 0) + 1
                self.n_all += 1
                if e.get("x") is not None and e.get("end") != "finished":
                    self.ends.append([e["x"], e["z"], e.get("end", ""), 1 if e.get("mid") else 0, e.get("t") or 0])
                    del self.ends[:-3000]
                if e.get("mid"):
                    self.n_mid += 1
                else:
                    self.laps.append((self.n_all, e.get("ms"), 1 if e.get("greedy") else 0, e.get("t") or 0))
                    if e.get("greedy") and e.get("ms"):
                        self.glaps.append([self.n_all, e["ms"] / 1000.0])
                        del self.glaps[:-2000]
            elif k == "test":
                self.tests.append(e)
                if e.get("map"):
                    self.map_tests[e["map"]] = self.map_tests.get(e["map"], 0) + 1
            elif k in ("target", "section") and not e.get("map"):
                self.targets.append({"lap": self.n_all, "k": k, "what": e.get("what") or e.get("mode")})
                del self.targets[:-1000]
            elif k == "upd":
                self.upds.append(e)
            elif k == "prof":
                self.prof = e
            elif k == "rb":
                self.rbs.append(e.get("r", {}))
                del self.rbs[:-10]


STORE = Store()


def _tail(path, n):
    """Last n lines of a text file without reading all of it (train.log grows for days)."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200 * n))
        data = f.read().decode("utf-8", errors="replace")
    return data.splitlines()[-n:]


_MARKS = {"key": None, "marks": []}


def map_marks(rd):
    """Start, checkpoints and finish for the map card: where the PB (else the champion / best lap) trace's checkpoint
    counter went up; start = the trace's first point. Cached by file times."""
    for name in ("pb_trace.npy", "champ_trace.npy", "best_trace.npy"):
        f = rd / name
        if f.exists():
            break
    else:
        return []
    key = (str(f), f.stat().st_mtime)
    if _MARKS["key"] == key:
        return _MARKS["marks"]
    out = []
    try:
        import numpy as np
        tr = np.load(f)
        out.append({"k": "start", "name": "Start", "x": round(float(tr[0, 0]), 1), "z": round(float(tr[0, 2]), 1)})
        # every trace here ends at the finish: checkpoints = counter increments before the last row
        ups = [i for i in range(1, len(tr) - 1) if tr[i, 4] > tr[i - 1, 4]]
        for j, i in enumerate(ups):
            out.append({"k": "cp", "name": f"CP {j + 1}", "x": round(float(tr[i, 0]), 1), "z": round(float(tr[i, 2]), 1)})
        out.append({"k": "finish", "name": "Finish", "x": round(float(tr[-1, 0]), 1), "z": round(float(tr[-1, 2]), 1)})
    except Exception:
        out = []
    _MARKS.update(key=key, marks=out)
    return out


def track_outline(rd):
    try:
        import numpy as np
        pts = np.load(rd / "centerline.npy")
        step = max(1, len(pts) // 600)
        return [[round(float(p[0]), 1), round(float(p[2]), 1)] for p in pts[::step]]
    except Exception:
        return []


_SEC = {"key": None, "rows": []}


def _cuts(tr):
    tk = tr[:, 4]
    c = [0] + [i for i in range(1, len(tk)) if tk[i] > tk[i - 1]]
    if c[-1] != len(tr) - 1:
        c.append(len(tr) - 1)
    return c


def _lap_limit_warn(cfg, rd):
    """max_lap_seconds shorter than 1.3 x champion / PB: tests and good laps end 'timeout' (caelid_v2)."""
    try:                                             # same rule as rewards.lap_limit_warning (no torch here)
        pb = float(cfg.get("pb_seconds") or 0) * 1000 or (read_json(rd / "pb_trace.json").get("ms")
                                                         if cfg.get("pb_file") else None)
        ms = [x for x in (read_json(rd / "best.json").get("champion_ms"), pb) if x]
        need = 1.3 * max(ms) / 1000.0 if ms else 0.0
        if need and float(cfg["max_lap_seconds"]) < need:
            return [f"Max lap {cfg['max_lap_seconds']:g} s is less than 1.3 x the champion / PB time ({need:.0f} s): "
                    "champion tests and good laps get cut off as 'timeout'. Raise 'Max lap (s)'."]
    except Exception:
        pass
    return []


def section_rows(rd):
    """Per checkpoint section: your PB/WR split, the AI's best split, failure rate, target in use."""
    sj = read_json(rd / "sections.json")
    files = ([rd / "pb_trace.npy", rd / "best_trace.npy", rd / "centerline.json", rd / "own_line.json"]
             + sorted(rd.glob("sec_best_*.npy")))
    key = (sj.get("t"), tuple((f.name, f.stat().st_mtime) for f in files if f.exists()))
    if key == _SEC["key"]:
        return _SEC["rows"]
    n = int(sj.get("n") or 0)
    pb = ai = None
    try:
        pb = np.load(rd / "pb_trace.npy")
    except (OSError, ValueError):
        pass
    try:
        ai = np.load(rd / "best_trace.npy")
    except (OSError, ValueError):
        pass

    def splits(tr):
        if tr is None or tr.ndim != 2 or tr.shape[1] < 5:
            return []
        c = _cuts(tr)
        return [(tr[c[j + 1], 3] - (tr[c[j], 3] if j else 0.0)) / 100.0 for j in range(len(c) - 1)]

    pbs, ais = splits(pb), splits(ai)
    n = max(n, len(pbs))
    guide = pb is not None
    own = bool(read_json(rd / "own_line.json").get("on"))
    chosen = read_json(rd / "centerline.json").get("guide") or []      # per section: PB / AI / AI* (build_guide)
    full = sj.get("full") or []
    rows = []
    for k in range(n):
        best = ais[k] if k < len(ais) and len(ais) == n else None
        try:
            seg = np.load(rd / f"sec_best_{k}.npy")
            d = (seg[-1, 3] - seg[0, 3]) / 100.0
            best = d if best is None else min(best, d)
        except (OSError, ValueError, IndexError):
            pass
        modes, rates, laps = sj.get("mode") or [], sj.get("rate") or [], sj.get("laps") or []
        m = modes[k] if k < len(modes) else "pb"
        rate = rates[k] if k < len(rates) else None
        if own or not guide:
            tgt = "its own line (champion)"
        elif m == "ai":
            tgt = f"AI route (fallback: {round((rate or 0) * 100)}% fail)"
        elif m == "try":
            tgt = "PB/WR (retrying)"
        elif k < len(chosen) and str(chosen[k]).startswith("AI"):
            tgt = "AI route (faster)"
        else:
            tgt = "PB/WR"
        nl = laps[k] if k < len(laps) else 0
        rows.append({"k": k + 1, "pb": round(pbs[k], 2) if k < len(pbs) else None,
                     "ai": round(best, 2) if best is not None else None,
                     "rate": rate, "laps": nl, "target": tgt,
                     # only mid-track (practice) starts got here so far, no lap from the start line
                     "practice_only": bool(nl) and k < len(full) and not full[k]})
    _SEC.update(key=key, rows=rows)
    return rows


HUB_CACHE = cfgmod.ROOT / "runs" / "_hub" / "index.json"


def hub_state(refresh=False):
    """The hub index (cached in runs/_hub/index.json, like tm_ai.py hub), my lineage, hub log tail."""
    cfg = cfgmod.load()
    err = None
    if refresh or not HUB_CACHE.exists():
        try:
            import urllib.request
            req = urllib.request.Request(cfg["hub_index_url"], headers={"User-Agent": "tm_ai-dashboard"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read(50 * 1024 * 1024).decode("utf-8"))
            HUB_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = HUB_CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, HUB_CACHE)
        except Exception as e:
            err = f"could not load {cfg['hub_index_url']}: {str(e)[:120]}"
    idx = read_json(HUB_CACHE)
    mine = set()
    for f in list((cfgmod.ROOT / "runs").glob("*/base.json")) + list((cfgmod.ROOT / "runs" / "_base").glob("*/base.json")):
        b = read_json(f).get("base_id")
        if b:
            mine.add(b)
    try:
        log = (cfgmod.ROOT / "runs" / "_hub" / "hub.log").read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
    except OSError:
        log = []
    user = cfg.get("github_user", "")
    return {"index": idx, "error": err, "age_s": round(time.time() - HUB_CACHE.stat().st_mtime) if HUB_CACHE.exists() else None,
            "lineage": sorted(mine), "user": user, "repo": cfg["hub_repo"],
            "verifier": bool(user) and user.lower() in [v.lower() for v in idx.get("verifiers") or []],
            "busy": alive(), "log": log}


_RUNS = {"t": 0.0, "v": []}


def runs_cached():
    """Existing runs for the run picker (refreshed every 10 s)."""
    if time.time() - _RUNS["t"] > 10:
        _RUNS.update(t=time.time(), v=safe(cfgmod.list_runs))
    return _RUNS["v"]


def tests_can_run(cfg, rd, best):
    """Champion tests start after the first finished full lap (base: on any pool map)."""
    if cfg["run_type"] == "base":
        return any((rd / "maps" / cfgmod.map_slug(e["map"]) / "best_run.json").exists() for e in cfg["map_pool"])
    # same rule as the trainers (common.full_lap_done): a champion, or a finished FULL lap in this run's events
    return bool(best.get("champion_ms")) or any(lp[1] for lp in STORE.laps)


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return []


def map_rows(cfg, rd):
    """Base run: one row per pool map - champion vs PB, tests, finish rate."""
    rows = []
    for e in cfg["map_pool"]:
        slug = cfgmod.map_slug(e["map"])
        md = rd / "maps" / slug
        best = read_json(md / "best.json")
        pb = e.get("pb_seconds") or ((read_json(md / "pb_trace.json").get("ms") or 0) / 1000.0) or None
        ml = STORE.map_laps.get(slug, [])
        rows.append({"name": Path(e["map"]).name, "slug": slug, "pb": pb,
                     "champ": best.get("champion_ms") / 1000.0 if best.get("champion_ms") else None,
                     "tests": STORE.map_tests.get(slug, 0), "laps": STORE.map_n.get(slug, 0),
                     "fin": round(100 * sum(1 for x in ml if x) / len(ml)) if ml else None})
    return rows


def build_state():
    cfg = cfgmod.load()
    rd = cfgmod.run_dir(cfg)
    STORE.refresh(rd / "events.jsonl")
    status = read_json(rd / "status.json")
    best = read_json(rd / "best.json")
    now = time.time()
    train_alive = alive() and PROC["kind"] == "train"
    fresh = status.get("state") == "running" and now - status.get("heartbeat", 0) < 150
    running = train_alive or fresh
    playing = alive() and PROC["kind"] == "play"
    benching = alive() and PROC["kind"] in ("bench", "transfer", "diagnose")
    if running:
        state, reason = "running", ""
    elif playing:
        state, reason = "playing", "champion is driving"
    elif benching:
        state, reason = "benchmarking", ({"bench": "testing car counts", "diagnose": "PPO diagnosis trials"}.get(PROC["kind"], "transfer test")) + " (see the log)"
    elif status.get("state") == "running":
        state, reason = "stopped", "stopped unexpectedly"
    else:
        state, reason = status.get("state", "idle"), status.get("reason", "")

    S = STORE
    fin = [(lp[0], lp[1] / 1000.0) for lp in S.laps if lp[1]]
    last50 = [lp[1] for lp in S.laps[-50:]]
    last20 = [s for _, s in fin][-20:]
    last_test_lap = S.tests[-1]["lap_n"] if S.tests else 0
    sess_end = now if running else (S.last_t or now)
    upd = S.upds[-1] if S.upds else {}
    log_path = rd / "train.log"
    try:
        log = _tail(log_path, 200)
    except OSError:
        log = []
    return {
        "config": {**cfg, "github_token": "********" if cfg.get("github_token") else ""},
        "state": state, "reason": reason,
        "can_play": champion_path(cfg).exists(),
        "pb_ms": cfg["pb_seconds"] * 1000 if cfg["pb_seconds"] else None,
        "champion_ms": best.get("champion_ms"), "first_test_ms": best.get("first_test_ms"),
        "laps_total": S.n_all, "mid_laps": S.n_mid,
        "session_laps": S.n_all - S.sess_lap0,
        "session_min": round((sess_end - S.sess_t0) / 60, 1) if S.sess_t0 else 0,
        "finished_pct": round(100 * sum(1 for ms in last50 if ms) / len(last50)) if last50 else None,
        "recent_avg": sum(last20) / len(last20) if last20 else None,
        "best_lap": min((s for _, s in fin), default=None),
        "next_test_in": max(0, cfg["test_every_laps"] - (S.n_all - last_test_lap)),
        "tests_wait": not tests_can_run(cfg, rd, best),
        "hit_ramp": read_json(rd / "hit_ramp.json") or None,
        "notice": S.notice, "run_warnings": list(cfgmod.RUN_WARNINGS) + _lap_limit_warn(cfg, rd),
        "runs": runs_cached(), "run_folder": rd.name,
        "tests_total": len(S.tests), "promotions": sum(1 for t in S.tests if t["promoted"]),
        "learn": {k: upd.get(k) for k in ("aux", "sil", "sil_act", "sil_n")},
        "updates": upd.get("u", 0), "steps": upd.get("steps", 0), "speed": upd.get("speed"),
        # the last 2000 finished full laps: [lap number, s, no-random lap, event time] (the page draws at most these)
        "laps": [[lp[0], round(lp[1] / 1000.0, 3), lp[2], round(lp[3], 1)] for lp in S.laps if lp[1]][-2000:],
        "glaps": S.glaps[-1000:], "marks": map_marks(rd),
        "tests": [{"lap_n": t["lap_n"], "ms": t["ms"], "champ_ms": t["champ_ms"], "t": t.get("t"),
                   "promoted": t["promoted"], "rolled_back": t["rolled_back"]} for t in S.tests[-400:]],
        "v": downsample([[i + 1, u["v"]] for i, u in enumerate(S.upds)]),
        "vl": downsample([[i + 1, u["vl"]] for i, u in enumerate(S.upds)]),
        "log": log,
        "rb": {k: sum(r.get(k, 0.0) for r in S.rbs[-5:]) / max(1, len(S.rbs[-5:]))
               for k in sorted({k for r in S.rbs[-5:] for k in r})},
        "ends": S.ends[-1500:], "track": track_outline(rd), "linesight": cfgmod.LINESIGHT,
        "run_preset": read_json(rd / "rewards.json").get("preset"), "own_line": read_json(rd / "own_line.json"), "targets": S.targets[-300:],
        "weights": [{"k": k, "default": d, "label": l, "hint": h} for k, (d, l, h) in cfgmod.WEIGHTS.items()],
        "defaults": {**cfgmod.DEFAULTS, "github_token": ""},
        "best_run": read_json(rd / "best_run.json") or None,
        "last_play": read_json(rd / "play_result.json") or None,
        "prof": S.prof, "bench": read_json(rd / "bench.json") or None,
        "sections": safe(section_rows, rd),
        "maps": safe(map_rows, cfg, rd) if cfg["run_type"] == "base" else None,
        "transfers": list(read_json(rd / "transfers.json").values()) if cfg["run_type"] == "base" else None,
        "base": read_json(rd / "base.json") or None,
    }


# --- Discord updates (webhook: no bot needed) --------------------------------------------------------
def _fmt(ms):
    return "–" if ms is None else f"{ms / 1000:.3f} s"


def discord_post(url, embed):
    import urllib.request
    data = json.dumps({"username": "Trackmania AI", "embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "tm_ai-dashboard/1.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def discord_embed(st, title, color):
    c = st["config"]
    pb, ch = st.get("pb_ms"), st.get("champion_ms")
    vs = ""
    if ch and pb:
        d = (ch - pb) / 1000
        vs = f"  ({'+' if d >= 0 else ''}{d:.3f} vs PB)"
    last = st["tests"][-1] if st.get("tests") else None
    last_txt = "–" if not last else (_fmt(last["ms"]) if last["ms"] else "no finish") + (" (new champion)" if last.get("promoted") else "")
    br = st.get("best_run") or {}
    f = [("Champion", _fmt(ch) + vs, False),
         ("Your PB", _fmt(pb), True),
         ("Fastest run saved", _fmt(br.get("ms")), True),
         ("Best training lap", f"{st['best_lap']:.3f} s" if st.get("best_lap") else "–", True),
         ("Recent average", f"{st['recent_avg']:.3f} s" if st.get("recent_avg") else "–", True),
         ("Full laps finished", f"{st['finished_pct']}%" if st.get("finished_pct") is not None else "–", True),
         ("Last champion test", last_txt, True),
         ("Laps", f"{st['session_laps']:,} this session · {st['laps_total']:,} total", True),
         ("Speed", f"{st['speed']:.0f}x real time" if st.get("speed") else "–", True),
         ("Session", f"{st['session_min']:.0f} min · {st.get('steps', 0):,} steps", True)]
    map_name = c["map"].replace(chr(92), "/").split("/")[-1].replace(".Map.Gbx", "")
    return {"title": title, "color": color,
            "description": f"**{map_name}** · run `{c['run']}` · {c.get('algo', 'ppo').upper()} · {c['envs']} cars",
            "fields": [{"name": n, "value": v, "inline": i} for n, v, i in f],
            "footer": {"text": "state: " + st["state"] + (f" ({st['reason']})" if st.get("reason") else "")},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def discord_loop():
    last_post, last_champ, last_state = 0.0, None, None
    while True:
        time.sleep(20)
        try:
            cfg = cfgmod.load()
            url = cfg.get("discord_webhook", "").strip()
            if not url:
                continue
            st = build_state()
            ch, state = st.get("champion_ms"), st["state"]
            if last_champ is None:
                last_champ = ch or 0
            if ch and last_champ and ch < last_champ:
                discord_post(url, discord_embed(st, f"🏁 New champion: {_fmt(ch)}", 0x39C46A))
                last_post = time.time()
            last_champ = ch or last_champ
            if last_state == "running" and state != "running":
                discord_post(url, discord_embed(st, "⏹ Training stopped", 0xE5484D if "crash" in st.get("reason", "")
                                                or "unexpected" in st.get("reason", "") else 0x8B93A0))
            elif state == "running" and time.time() - last_post >= max(1, cfg.get("discord_every_min", 15)) * 60:
                discord_post(url, discord_embed(st, "📊 Training update", 0x3B8BEA))
                last_post = time.time()
            last_state = state
        except Exception:
            pass


def pid_alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:
        pass
    if sys.platform == "win32":              # (os.kill would terminate the process on Windows)
        r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}"], capture_output=True, text=True)
        return str(int(pid)) in r.stdout
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def kill_pid(pid):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))], capture_output=True)
        else:
            os.kill(int(pid), 15)
    except Exception:
        pass


def frozen_guard():
    """Training that froze is restarted from its last checkpoint: the trainer marks status.json "froze" itself
    (its freeze guard), or its heartbeat stops for 150 s. Also for runs this dashboard did not start (the pid comes
    from status.json)."""
    restarts = []
    while True:
        time.sleep(15)
        try:
            if alive() and PROC["kind"] != "train":
                continue
            cfg = cfgmod.load()
            rd = cfgmod.run_dir(cfg)
            st = read_json(rd / "status.json")
            now = time.time()
            if (rd / "STOP").exists() or (alive() and now - PROC.get("t0", 0) < 180):
                continue
            hb, pid = st.get("heartbeat", now), st.get("pid")
            froze = st.get("state") == "froze" and now - hb < 600
            stale = st.get("state") == "running" and 150 < now - hb < 1800 and (alive() or pid_alive(pid))
            if not (froze or stale):
                continue
            restarts[:] = [t for t in restarts if now - t < 3600] + [now]
            with open(rd / "train.log", "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + ("Training froze; " if froze else "Training froze for 150 s; ")
                        + "restarting from the last checkpoint (stacks in hang.txt).\n")
            if alive():
                kill_proc()
            elif pid_alive(pid) and pid != os.getpid():
                kill_pid(pid)
            st["state"] = "stopped"
            (rd / "status.json").write_text(json.dumps(st), encoding="utf-8")
            time.sleep(5)
            if len(restarts) <= 5:
                start("train", ["--new"])          # the same run again: never the "trained run exists" question
            else:
                with open(rd / "train.log", "a", encoding="utf-8") as f:
                    f.write(time.strftime("%H:%M:%S ") + "Froze more than 5 times in an hour: not restarting.\n")
        except Exception:
            pass


def champion_path(cfg):
    rd = cfgmod.run_dir(cfg)
    if cfg["run_type"] == "base":
        return rd / "maps" / cfgmod.map_slug(cfg["map_pool"][0]["map"]) / "champion.zip" if cfg["map_pool"] else rd / "-"
    return rd / ("champion.pt" if cfg["algo"] == "iqn" else "champion.zip")


def start(kind, extra=()):
    cfg = cfgmod.load()
    if alive():
        return 409, "Something is already running."
    base = cfg["run_type"] == "base"
    if kind == "hub":
        pass
    elif base and not cfg["map_pool"]:
        return 400, "Add maps to the map pool in Settings first (base run)."
    elif not base and not cfg["map"]:
        return 400, "Set the map path in Settings first."
    if base and cfg["algo"] != "ppo" and kind == "train":
        return 400, "Base runs are PPO only."
    if kind == "play" and not champion_path(cfg).exists():
        return 400, "No champion yet."
    if kind == "hub":
        rd = cfgmod.ROOT / "runs" / "_hub"
        rd.mkdir(parents=True, exist_ok=True)
        logf = open(rd / "hub.log", "a", encoding="utf-8")
        PROC["p"] = subprocess.Popen([sys.executable, "-u", "tm_ai.py", "hub", *extra], cwd=str(ROOT), stdout=logf,
                                     stderr=subprocess.STDOUT)
        PROC["kind"], PROC["t0"] = kind, time.time()
        return 200, "started - see the Hub tab's log"
    if kind == "transfer" and not (cfgmod.base_dir(cfg["base_name"]) / "latest.zip").exists():
        return 400, f"Base '{cfg['base_name']}' has no model yet."
    rd = cfgmod.run_dir(cfg)
    logf = open(rd / "train.log", "a", encoding="utf-8")
    cmd = {"train": "train", "bench": "bench", "transfer": "transfer", "hub": "hub", "diagnose": "diagnose"}.get(kind, "drive")
    PROC["p"] = subprocess.Popen([sys.executable, "-u", "tm_ai.py", cmd, *extra],
                                 cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    PROC["kind"], PROC["t0"] = kind, time.time()
    return 200, "ok"


def stop():
    cfg = cfgmod.load()
    rd = cfgmod.run_dir(cfg)
    if PROC["kind"] in ("play", "bench", "transfer", "hub", "diagnose") or not (alive() or read_json(rd / "status.json").get("state") == "running"):
        kill_proc()
        return
    (rd / "STOP").write_text("stop")                       # graceful: saves the model first
    # hard stop if it hangs - only THIS process: a training started later must never be killed by this timer
    threading.Timer(90, kill_proc, args=(PROC["p"],)).start()


def open_folder(path):
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    except OSError:
        pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, (ROOT / "dashboard.html").read_bytes(), "text/html")
        elif self.path.startswith("/api/state"):
            self.send(200, json.dumps(build_state()))
        elif self.path.startswith("/api/hub"):
            self.send(200, json.dumps(hub_state("refresh=1" in self.path)))
        else:
            self.send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        code, msg = 200, "ok"
        if self.path == "/api/config":
            if body.get("github_token") == "********":       # masked in the page: keep the saved token
                body["github_token"] = cfgmod.load().get("github_token", "")
            merged = cfgmod.load()                            # merge: a partial save never resets other settings
            merged.update({k: v for k, v in body.items() if k in cfgmod.DEFAULTS})
            cfgmod.save(merged)
        elif self.path == "/api/start":
            cfg = cfgmod.load()
            others = [] if body.get("new") or alive() else safe(cfgmod.start_conflict, cfg)
            if others:                                   # never start a trained map from scratch by accident
                self.send(409, json.dumps({"message": "A trained run for this map exists", "choice": others,
                                           "run": cfg["run"]}))
                return
            code, msg = start("train", ["--new"] if body.get("new") else [])
        elif self.path == "/api/hub/import":
            pid = "".join(c for c in str(body.get("id") or "") if c.isalnum() or c == "-")
            name = "".join(c for c in str(body.get("name") or pid) if c.isalnum() or c in "-_")
            code, msg = (400, "no pack id") if not pid else start(
                "hub", ["import", pid, "--as", name] + (["--base"] if body.get("base") else []))
        elif self.path == "/api/hub/verify_pending":
            code, msg = start("hub", ["verify-pending"])
        elif self.path == "/api/transfer":
            m = str(body.get("map") or "").strip().strip('"')
            if not m:
                code, msg = 400, "Give the path of a map that is not in the pool."
            else:
                code, msg = start("transfer", ["--map", m] + (["--pb", str(body["pb"])] if body.get("pb") else []))
        elif self.path == "/api/bench":
            code, msg = start("bench")
        elif self.path == "/api/diagnose":
            code, msg = start("diagnose")
        elif self.path == "/api/play":
            code, msg = start("play")
        elif self.path == "/api/stop":
            stop()
        elif self.path == "/api/discord_test":
            url = cfgmod.load().get("discord_webhook", "").strip()
            if not url:
                code, msg = 400, "Save a Discord webhook URL in Settings first."
            else:
                try:
                    discord_post(url, discord_embed(build_state(), "✅ Test message from your Trackmania AI", 0x3B8BEA))
                    msg = "Sent - check your Discord channel."
                except Exception as e:
                    code, msg = 502, f"Discord refused it: {str(e)[:120]}"
        elif self.path == "/api/open":
            open_folder(cfgmod.run_dir(cfgmod.load()))
        else:
            code, msg = 404, "unknown"
        self.send(code, json.dumps({"message": msg}))


def open_window(url):
    """Own lightweight window (pywebview: WebView2 on Windows) if installed. Returns False -> use the browser.
    Closing the window never stops training: it runs in its own process, and this server keeps running."""
    try:
        import webview
    except ImportError:
        print("pywebview is not installed (pip install pywebview): opening the dashboard in the browser.")
        return False
    try:
        webview.create_window("Trackmania AI", url, width=1440, height=920, background_color="#0e1013")
        webview.start()                                  # blocks until the window is closed
    except Exception as e:
        print(f"Could not open the dashboard window ({str(e)[:100]}): opening the browser.")
        return False
    print("Dashboard window closed. Training (if running) keeps going; this server keeps watching it.")
    print(f"Reopen {url} in a browser, or press Ctrl+C here to quit the dashboard.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--window", action="store_true", help="open in its own window (pywebview) instead of the browser")
    args = ap.parse_args()
    port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print("Trackmania AI dashboard:", url, "(Ctrl+C to quit)")
    threading.Thread(target=frozen_guard, daemon=True).start()
    threading.Thread(target=discord_loop, daemon=True).start()
    if args.window:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        if not open_window(url):
            webbrowser.open(url)
        try:
            while True:                                  # keep serving (and guarding training) after the window
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
