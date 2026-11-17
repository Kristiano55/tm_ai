"""
Control panel for the Trackmania AI.   python dashboard.py [--port 8765]

Shows live training stats and starts/stops training from the browser. Needs no extra packages.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
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


def kill_proc():
    p = PROC["p"]
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
        self.laps, self.tests, self.upds = [], [], []     # laps: (lap number, ms) of full laps only
        self.n_all, self.n_mid = 0, 0
        self.sess_t0, self.sess_lap0, self.last_t = None, 0, None

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
            elif k == "lap":
                self.n_all += 1
                if e.get("mid"):
                    self.n_mid += 1
                else:
                    self.laps.append((self.n_all, e.get("ms")))
            elif k == "test":
                self.tests.append(e)
            elif k == "upd":
                self.upds.append(e)


STORE = Store()


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
    if running:
        state, reason = "running", ""
    elif playing:
        state, reason = "playing", "champion is driving"
    elif status.get("state") == "running":
        state, reason = "stopped", "stopped unexpectedly"
    else:
        state, reason = status.get("state", "idle"), status.get("reason", "")

    S = STORE
    fin = [(n, ms / 1000.0) for n, ms in S.laps if ms]
    last50 = [ms for _, ms in S.laps[-50:]]
    last20 = [s for _, s in fin][-20:]
    last_test_lap = S.tests[-1]["lap_n"] if S.tests else 0
    sess_end = now if running else (S.last_t or now)
    upd = S.upds[-1] if S.upds else {}
    log_path = rd / "train.log"
    try:
        log = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-14:]
    except OSError:
        log = []
    return {
        "config": cfg, "state": state, "reason": reason,
        "can_play": (rd / "champion.zip").exists(),
        "pb_ms": cfg["pb_seconds"] * 1000 if cfg["pb_seconds"] else None,
        "champion_ms": best.get("champion_ms"), "first_test_ms": best.get("first_test_ms"),
        "laps_total": S.n_all, "mid_laps": S.n_mid,
        "session_laps": S.n_all - S.sess_lap0,
        "session_min": round((sess_end - S.sess_t0) / 60, 1) if S.sess_t0 else 0,
        "finished_pct": round(100 * sum(1 for ms in last50 if ms) / len(last50)) if last50 else None,
        "recent_avg": sum(last20) / len(last20) if last20 else None,
        "best_lap": min((s for _, s in fin), default=None),
        "next_test_in": max(0, cfg["test_every_laps"] - (S.n_all - last_test_lap)),
        "tests_total": len(S.tests), "promotions": sum(1 for t in S.tests if t["promoted"]),
        "updates": upd.get("u", 0), "steps": upd.get("steps", 0), "speed": upd.get("speed"),
        "laps": downsample([[n, s] for n, s in fin]),
        "tests": [{"lap_n": t["lap_n"], "ms": t["ms"], "champ_ms": t["champ_ms"],
                   "promoted": t["promoted"], "rolled_back": t["rolled_back"]} for t in S.tests[-400:]],
        "v": downsample([[i + 1, u["v"]] for i, u in enumerate(S.upds)]),
        "vl": downsample([[i + 1, u["vl"]] for i, u in enumerate(S.upds)]),
        "log": log,
        "weights": [{"k": k, "default": d, "label": l, "hint": h} for k, (d, l, h) in cfgmod.WEIGHTS.items()],
        "best_run": read_json(rd / "best_run.json") or None,
        "last_play": read_json(rd / "play_result.json") or None,
    }


def start(kind):
    cfg = cfgmod.load()
    if alive():
        return 409, "Something is already running."
    if not cfg["map"]:
        return 400, "Set the map path in Settings first."
    if kind == "play" and not (cfgmod.run_dir(cfg) / "champion.zip").exists():
        return 400, "No champion yet."
    rd = cfgmod.run_dir(cfg)
    logf = open(rd / "train.log", "a", encoding="utf-8")
    PROC["p"] = subprocess.Popen([sys.executable, "-u", "tm_ai.py", "train" if kind == "train" else "drive"],
                                 cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    PROC["kind"] = kind
    return 200, "ok"


def stop():
    cfg = cfgmod.load()
    rd = cfgmod.run_dir(cfg)
    if PROC["kind"] == "play" or not (alive() or read_json(rd / "status.json").get("state") == "running"):
        kill_proc()
        return
    (rd / "STOP").write_text("stop")                       # graceful: saves the model first
    threading.Timer(90, kill_proc).start()                 # hard stop if it hangs


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
            cfgmod.save(body)
        elif self.path == "/api/start":
            code, msg = start("train")
        elif self.path == "/api/play":
            code, msg = start("play")
        elif self.path == "/api/stop":
            stop()
        elif self.path == "/api/open":
            open_folder(cfgmod.run_dir(cfgmod.load()))
        else:
            code, msg = 404, "unknown"
        self.send(code, json.dumps({"message": msg}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print("Trackmania AI dashboard:", url, "(Ctrl+C to quit)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
