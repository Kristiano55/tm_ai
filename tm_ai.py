"""
Trackmania 2020 RL agent on tickgym (PPO, runs on CPU).

    python tm_ai.py selftest --map "C:\\Maps\\X.Map.Gbx"
    python tm_ai.py train                (settings come from config.json / the dashboard)
    python tm_ai.py drive                (plays runs/<run>/champion.zip, writes play.txt for TICK)

Normally you start/stop all of this from the dashboard: python dashboard.py
"""
import argparse
import faulthandler
import json
import os
import warnings
warnings.filterwarnings("ignore", message=r".*(get_schedule_fn|constant_fn).*")   # SB3 deprecation noise
import sys
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tickgym import Action, Client, InputText, Lidar, TelemetryGroup, TransportError

import config as cfgmod

GROUPS = (TelemetryGroup.RACE | TelemetryGroup.VEHICLE
          | TelemetryGroup.DRIVETRAIN | TelemetryGroup.WHEELS)
GROUPS_EXTRA = GROUPS | TelemetryGroup.EFFECTS | TelemetryGroup.RUNTIME

# Extra observations, in the order they are appended after the 38 base features.
EXTRA_NAMES = (
    ["accel_x", "accel_y", "accel_z"] + [f"wheel_spin_{w}" for w in ("fl", "fr", "rl", "rr")] + ["icing"]
    + ["motion_alignment", "wall_hit", "roof_contact", "wetness", "surface_grip", "soft_surface", "on_ice", "in_water"]
    + ["accel_coef", "control_coef", "gravity_coef", "adherence_coef"]
    + ["turbo_level", "turbo_left", "reactor", "reactor_dir", "engine_off", "no_brakes", "no_steering", "no_grip",
       "fragile", "slow_motion", "cruise"]
    + ["ghost_delta", "ghost_speed", "ghost_speed_25m", "ghost_speed_60m", "speed_vs_ghost"]
    + ["track_fraction", "line_offset", "line_angle", "recent_hit", "airtime"]
)
N_BASE0 = 38                                  # features before the extras

# Steering levels (-127..127). "progressive" is fine near the centre (small corrections at speed)
# and coarse at the edges (hairpins only need "a lot"). Both have 21 levels.
_P = np.round(np.array([4, 8, 13, 19, 26, 35, 46, 60, 78, 100]) * 1.27).astype(int)
STEER_LEVELS = {
    "uniform": np.round(np.arange(-10, 11) * 12.7).astype(int),
    "progressive": np.concatenate([-_P[::-1], [0], _P]),
}


def steer_levels(n, mode="progressive"):
    """n steering levels (odd: 21, 31, 41) on the same curve as the 21-level sets (n = 21 gives them exactly)."""
    half = (int(n) - 1) // 2
    base = STEER_LEVELS[mode][11:].astype(np.float64)              # the 10 positive levels
    pos = np.round(np.interp(np.linspace(1, 10, half), np.arange(1, 11), base)).astype(int)
    return np.concatenate([-pos[::-1], [0], pos])


_HANG = {"file": None, "pet": time.time(), "guard": None}


def action_space(nvec, analog):
    """PPO/IQN action space: MultiDiscrete [steer level, mode], or with analog steering Box [steer -1..1, mode]."""
    if analog:
        return spaces.Box(np.array([-1.0, 0.0], np.float32), np.array([1.0, nvec[1] - 1], np.float32), dtype=np.float32)
    return spaces.MultiDiscrete(nvec)


def steer_value(a0, levels, analog):
    """Policy steering output -> TM steering -127..127 (integer)."""
    if analog:
        return int(np.clip(np.round(float(a0) * 127.0), -127, 127))
    return int(levels[int(a0)])


def _hang_file():
    """runs/<run>/hang.txt, opened once and kept open (faulthandler writes to it from C, even when Python hangs)."""
    if _HANG["file"] is None:
        try:
            import config as _c
            d = _c.run_dir(_c.load())
            _HANG["file"] = open(d / "hang.txt", "a", encoding="utf-8")
        except Exception:
            _HANG["file"] = sys.stderr
    return _HANG["file"]


GAS, BRAKE = (0, 3), (2, 3)                      # throttle modes: 0 gas, 1 coast, 2 brake, 3 gas + brake
MODE_VEC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float64)   # last-action inputs (gas, coast, brake)
from ppo_ext import N_PRIV  # noqa: E402  (privileged critic inputs, appended after the lidar)


def watchdog(seconds=90):
    """Re-armed while things move. If this process is stuck this long, all its threads' stacks go to hang.txt."""
    _HANG["pet"] = time.time()
    try:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(seconds, repeat=False, file=_hang_file())
    except Exception:
        pass


def freeze_guard(run_dir, seconds=180):
    """Trainer only: a thread that notices when the main loop stops calling watchdog() for `seconds`. It writes all
    stacks to hang.txt, marks status.json "froze" (the dashboard restarts the run), stops the child processes and
    exits. Works on Windows (no signals needed)."""
    if _HANG["guard"] is not None:
        return
    import threading
    run_dir = Path(run_dir)
    _HANG["pet"] = time.time()

    def loop():
        while True:
            time.sleep(5)
            idle = time.time() - _HANG["pet"]
            if idle < seconds:
                continue
            try:
                f = _hang_file()
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} trainer froze ({idle:.0f} s without progress) ===\n")
                f.flush()
                faulthandler.dump_traceback(file=f, all_threads=True)
                f.flush()
            except Exception:
                pass
            try:
                st = read_json(run_dir / "status.json")
                write_json(run_dir / "status.json", {**st, "state": "froze", "heartbeat": time.time(),
                                                     "reason": f"froze ({idle:.0f} s without progress)",
                                                     "pid": os.getpid()})
                say(f"Training froze ({idle:.0f} s without progress); stacks in hang.txt. Exiting so it can restart.")
            except Exception:
                pass
            try:
                import multiprocessing as mp
                for c in mp.active_children():
                    c.terminate()
            except Exception:
                pass
            os._exit(3)

    _HANG["guard"] = threading.Thread(target=loop, daemon=True, name="freeze-guard")
    _HANG["guard"].start()


# --- surface vocabulary: material / gameplay ids -> fixed one-hot slots, shared by ALL runs -------------------
N_SURF = 16                                   # slots per vocabulary; slot 16 = "other" (any id beyond the first 16)
VOCAB_FILE = cfgmod.ROOT / "runs" / "_materials.json"
_VOCAB = {"material": {}, "gameplay": {}, "m": None}


def _vocab_load():
    try:
        m = os.path.getmtime(VOCAB_FILE)
    except OSError:
        return
    if m == _VOCAB["m"]:
        return
    d = read_json(VOCAB_FILE)
    for kind in ("material", "gameplay"):
        _VOCAB[kind] = {int(k): int(v) for k, v in (d.get(kind) or {}).items()}
    _VOCAB["m"] = m


def surface_slot(kind, sid):
    """Slot (0..16) of a surface id. New ids get the next free slot, in order of first appearance, written to
    runs/_materials.json under a lock file so every car, run and later model maps an id to the same input."""
    sid = int(sid)
    voc = _VOCAB[kind]
    if sid in voc:
        return voc[sid]
    _vocab_load()
    if sid in _VOCAB[kind]:
        return _VOCAB[kind][sid]
    lock = VOCAB_FILE.with_suffix(".lock")
    try:
        VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return N_SURF
    for _ in range(200):                                   # ~2 s
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 10:     # stale lock (crashed process)
                    os.unlink(lock)
            except OSError:
                pass
            time.sleep(0.01)
    else:
        return N_SURF                                      # could not lock: "other" this once
    try:
        d = read_json(VOCAB_FILE)
        d.setdefault("material", {})
        d.setdefault("gameplay", {})
        cur = {int(k): int(v) for k, v in d[kind].items()}
        if sid not in cur:
            cur[sid] = len(cur) if len(cur) < N_SURF else N_SURF
            d[kind] = {str(k): v for k, v in cur.items()}
            d["note"] = "Surface ids -> input slots (0-15; 16 = other). Shared by all runs: never edit or reorder."
            write_json(VOCAB_FILE, d)
            say(f"New {kind} surface id {sid} -> input slot "
                + (str(cur[sid] + 1) if cur[sid] < N_SURF else "'other' (all 16 slots taken)"))
        _VOCAB[kind] = cur
        _VOCAB["m"] = os.path.getmtime(VOCAB_FILE)
    finally:
        os.close(fd)
        try:
            os.unlink(lock)
        except OSError:
            pass
    return _VOCAB[kind][sid]


MAPS_FILE = cfgmod.ROOT / "runs" / "_maps.json"


def remember_map(uid, path, sha=None, run_dir=None):
    """Map UID -> local file (runs/_maps.json, used by verify / benchmark) and the run folder's map_info.json."""
    try:
        rec = {"uid": uid, "path": str(path), "sha256": sha, "name": Path(str(path)).name}
        if run_dir is not None and read_json(Path(run_dir) / "map_info.json").get("uid") != uid:
            write_json(Path(run_dir) / "map_info.json", rec)
        d = read_json(MAPS_FILE)
        if d.get(uid, {}).get("path") != str(path):
            MAPS_FILE.parent.mkdir(parents=True, exist_ok=True)
            d[uid] = rec
            write_json(MAPS_FILE, d)
    except Exception:                                      # file trouble must never stop training
        pass


def onehot_ids(kind, ids, valid=None):
    """[len(ids) * 17] one-hots; rows where valid is False stay all zero (e.g. a ray that hit nothing)."""
    out = np.zeros((len(ids), N_SURF + 1), np.float32)
    for i, sid in enumerate(ids):
        if valid is None or valid[i]:
            out[i, surface_slot(kind, sid)] = 1.0
    return out.ravel()


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def write_json(path, obj):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except OSError:
        pass


def atomic_save(path, write, backup=True):
    """write(tmp_path) then rename over path; the previous file is kept as <name>.prev<ext> (one backup).
    A crash mid-write never leaves a half-written checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    if backup and path.exists():
        os.replace(path, path.with_name(path.stem + ".prev" + path.suffix))
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _f(obj, name, default=0.0):
    """Read one telemetry scalar; a field that is missing in this tickgym version reads as `default`."""
    try:
        return float(getattr(obj, name))
    except Exception:
        return default


def to_car_frame(q, v):
    """Rotate world vector v into the car frame. q = world orientation (x, y, z, w)."""
    x, y, z, w = (float(c) for c in q)
    qv = np.array([-x, -y, -z])                      # conjugate = inverse rotation
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def to_car_frame_many(q, vs):
    """to_car_frame for a batch of world vectors vs [N, 3] (same numbers, one numpy call)."""
    x, y, z, w = (float(c) for c in q)
    qv = np.array([-x, -y, -z])
    vs = np.asarray(vs, dtype=np.float64)
    t = 2.0 * np.cross(qv, vs)
    return vs + w * t + np.cross(qv, t)


# --- reference line (the driven path of a finished lap) ---------------------------------------
def make_line(xyz):
    """Smooth a driven path and thin it to ~1 m spacing."""
    p = np.asarray(xyz, dtype=np.float64)
    sm = np.array([p[max(0, i - 2):i + 3].mean(axis=0) for i in range(len(p))])
    sm[0], sm[-1] = p[0], p[-1]
    keep = [0]
    for i in range(1, len(sm)):
        if np.linalg.norm(sm[i] - sm[keep[-1]]) >= 1.0:
            keep.append(i)
    if keep[-1] != len(sm) - 1:
        keep.append(len(sm) - 1)
    return sm[keep]


def save_npy(path, arr):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def line_progress(pts, seg, cum, pos, di):
    """Nearest point on a polyline, searched forward from index di. Returns (metres along line, index)."""
    lo, hi = max(di - 10, 0), min(di + 150, len(pts) - 1)
    j = lo + int(np.argmin(np.linalg.norm(pts[lo:hi + 1] - pos, axis=1)))
    if j >= len(pts) - 1:
        return float(cum[-1]), j
    a = float(np.clip(np.dot(pos - pts[j], pts[j + 1] - pts[j]) / max(seg[j], 1e-3) ** 2, 0.0, 1.0))
    return float(cum[j] + a * seg[j]), j


def dense_line(pts):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, seg, np.concatenate([[0.0], np.cumsum(seg)])


def trace_to_profile(dense, trace):
    """Where a driven lap was on the line at each tick: rows (metres along line, race tick)."""
    pts, seg, cum = dense
    di, ss = 0, []
    for p in np.asarray(trace)[:, :3]:
        s, di = line_progress(pts, seg, cum, p, di)
        ss.append(s)
    ss = np.maximum.accumulate(np.array(ss))
    keep = np.concatenate([[True], np.diff(ss) > 0.05])
    return np.stack([ss[keep], np.asarray(trace)[keep, 3]], axis=1)


def trace_record(g, actions, ticks=5, max_ticks=60000):
    """Play a recorded race (e.g. your PB) and log where the car was every `ticks` ticks.
    Rows: x, y, z, race tick, checkpoints taken. The inputs are only used to know where the run drove."""
    acts = sorted((Action(max(0, int(a.tick)), a.kind, a.value) for a in actions), key=lambda a: a.tick)
    t = g.reset(seed=0)
    hist, i = [], 0
    while True:
        hi = g.next_tick + ticks
        win = []
        while i < len(acts) and acts[i].tick < hi:
            win.append(acts[i])
            i += 1
        t = g.step(ticks, win)
        hist.append([*np.asarray(t.vehicle.position, dtype=np.float64), int(t.race.race_tick),
                     int(t.race.checkpoints_taken)])
        if t.finished or t.race.race_tick > max_ticks:
            break
    return np.array(hist), (int(t.race.race_time_ms) if t.finished else None)


def _cp_cuts(trace):
    tk = trace[:, 4]
    cuts = [0] + [i for i in range(1, len(tk)) if tk[i] > tk[i - 1]]
    if cuts[-1] != len(trace) - 1:
        cuts.append(len(trace) - 1)
    return cuts


def freeze_preset(cfg, run_dir):
    """A NEW run (no model yet) with reward_preset "minimal" gets rewards.json with the MINIMAL weights; the cars use
    them instead of the settings. Existing runs are never changed."""
    run_dir = Path(run_dir)
    f = run_dir / "rewards.json"
    if f.exists():
        d = read_json(f)
        say(f"This run uses the '{d.get('preset')}' reward preset (rewards.json); weight changes in Settings don't "
            "apply to it.")
        return
    has_model = any((run_dir / n).exists() for n in ("latest.zip", "champion.zip", "latest.pt", "latest.prev.pt",
                                                     "champion.pt"))
    if cfg.get("reward_preset") == "minimal" and cfg.get("run_type") != "base" and not has_model:
        write_json(f, {"preset": "minimal", "weights": dict(cfgmod.MINIMAL), "t": time.time()})
        say("New run: minimal reward preset (progress, time, finish, checkpoint, give-up cost) frozen in rewards.json.")


def champion_ghost(env, run_dir, ms, log=None):
    """Rebuild the ghost (reward target) from the champion lap: called only when the champion improves (or no ghost
    exists). PB-guide runs rebuild the guide (faster of PB vs AI best per section). Logs event k=target."""
    run_dir = Path(run_dir)
    try:
        if env.guide:
            r = build_guide(run_dir)
            if not r:
                return False
            say(f"Champion {ms / 1000:.3f} s -> target line sections: {' '.join(r[1])} (target pace {r[0] / 1000:.3f} s)")
            gms = r[0]
        else:
            old = read_json(run_dir / "profile.json").get("ms")
            env._offer_profile(ms)
            gms = read_json(run_dir / "profile.json").get("ms")
            if gms == old:
                return False
        if log:
            log(k="target", what="ghost", ms=int(ms), ghost_ms=int(gms))
        return True
    except Exception as e:                      # file trouble must never stop training
        say(f"Ghost rebuild failed ({str(e)[:80]})")
        return False


def build_guide(run_dir):
    """Target line = per checkpoint section, the faster of your PB and the AI's best finished run.
    So the line starts as your PB and switches section by section wherever the AI finds something faster.
    Sections marked "ai" in sections.json (laps keep failing there on the PB/WR route) use the AI's own fastest
    drive through that section instead (best finished lap or best single section drive, sec_best_<k>.npy)."""
    run_dir = Path(run_dir)
    try:
        pb = np.load(run_dir / "pb_trace.npy")
    except (OSError, ValueError):
        return None
    srcs = {"PB": pb}
    try:
        ai = np.load(run_dir / "best_trace.npy")
        if ai.ndim == 2 and ai.shape[1] >= 5:
            srcs["AI"] = ai
    except (OSError, ValueError):
        pass
    cuts = {k: _cp_cuts(v) for k, v in srcs.items()}
    n = len(cuts["PB"]) - 1
    use = [k for k in srcs if len(cuts[k]) - 1 == n]          # same checkpoint count only
    modes = read_json(run_dir / "sections.json").get("mode", [])
    rows, t0, chosen = [], 0.0, []
    for j in range(n):
        # candidates: (duration, label, rows of the section, base tick)
        cand = []
        for k in use:
            tr, a, b = srcs[k], cuts[k][j], cuts[k][j + 1]
            dur = tr[b, 3] - (tr[a, 3] if j else 0.0)
            cand.append((dur, k, tr[a:b + 1] if j == n - 1 else tr[a:b], tr[a, 3] if j else 0.0))
        if j < len(modes) and modes[j] == "ai":
            ai_c = [c for c in cand if c[1] != "PB"]
            try:
                seg = np.load(run_dir / f"sec_best_{j}.npy")
                if seg.ndim == 2 and len(seg) >= 3:
                    ai_c.append((seg[-1, 3] - seg[0, 3], "AI*", seg if j == n - 1 else seg[:-1], seg[0, 3]))
            except (OSError, ValueError):
                pass
            cand = ai_c or cand                                # no AI route through it yet: keep the PB
        dur, k, part, base = min(cand, key=lambda c: c[0])
        for r in part:
            rows.append([r[0], r[1], r[2], t0 + r[3] - base])
        t0 += dur
        chosen.append(k)
    comp = np.array(rows)
    if len(comp) < 10:
        return None
    line = make_line(comp[:, :3])
    prof = trace_to_profile(dense_line(line), comp)
    ms = int(round(t0 * 10))
    save_npy(run_dir / "profile.npy", prof)
    save_npy(run_dir / "centerline.npy", line)
    write_json(run_dir / "centerline.json", {"ms": ms, "guide": chosen, "t": time.time()})
    write_json(run_dir / "profile.json", {"ms": ms})
    return ms, chosen


FAIL_ENDS = ("stuck", "off-zone", "far off line", "behind ghost", "timeout")


class HitRamp:
    """Wall-hit penalty ramp-in (setting w_hit_ramp): while the run's full-lap finish rate over the last 100 laps is
    below 30 % the penalty counts FLOOR (25 %), growing linearly to 100 % at a 60 % finish rate. Once 60 % has been
    reached it stays at 100 % for good. Written to hit_ramp.json in the run (per map) folder; the cars read it."""

    LO, HI, FLOOR, N = 0.30, 0.60, 0.25, 100

    def __init__(self, run_dir, cfg):
        self.f, self.on = Path(run_dir) / "hit_ramp.json", bool(cfg.get("w_hit_ramp", True))
        d = read_json(self.f)
        self.laps, self.reached = list(d.get("laps", []))[-self.N:], bool(d.get("reached"))
        self.last, self.t = None, 0.0

    def add(self, info):
        if info.get("mid") or info.get("pool") or info.get("end") in ("car restarted", "gym lost"):
            return
        self.laps = (self.laps + [1 if info.get("finished") else 0])[-self.N:]

    def rate(self):
        return sum(self.laps) / len(self.laps) if self.laps else 0.0

    def factor(self):
        if not self.on:
            return 1.0
        r = self.rate()
        if len(self.laps) >= 20 and r >= self.HI:
            self.reached = True
        if self.reached:
            return 1.0
        return float(self.FLOOR + (1.0 - self.FLOOR) * np.clip((r - self.LO) / (self.HI - self.LO), 0.0, 1.0))

    def update(self, log=None):
        f = round(self.factor(), 3)
        if f != self.last or time.time() - self.t > 30:
            if self.last is not None and f != self.last and log and (f == 1.0 or abs(f - self.last) >= 0.1):
                log(k="hit_ramp", factor=f, rate=round(self.rate(), 3))
            self.last, self.t = f, time.time()
            write_json(self.f, {"factor": f, "rate": round(self.rate(), 3), "reached": self.reached, "on": self.on,
                                "laps": self.laps, "t": time.time()})


class SectionTracker:
    """Per checkpoint section: how often laps that reach it fail there. With a PB/WR target line, a section where
    most laps fail switches its target to the AI's own best route through it ("ai"), and back to the PB/WR route
    when the AI gets through on that route (or on a periodic retry). Writes sections.json (read by the cars:
    difficulty input + target mode) and rebuilds the guide on every switch."""

    WINDOW, MIN_N, FAIL, RETRY_S, TRY_N = 200, 50, 0.7, 600.0, 50
    # hysteresis (a section used to flip PB <-> AI every ~30 s): after a switch the target is kept >= HOLD_S, and
    # the rate must cross with a margin: to AI above FAIL + MARGIN, back to PB at <= 0.5 - MARGIN / WR_N laps on PB
    HOLD_S, MARGIN, WR_N, LOG_S = 600.0, 0.1, 5, 600.0

    def __init__(self, run_dir, cfg, n_sections):
        self.run_dir, self.cfg = Path(run_dir), cfg
        self.f = self.run_dir / "sections.json"
        self.guide = bool(cfg.get("pb_file")) and (self.run_dir / "pb_trace.npy").exists()
        self.on = self.guide and cfg.get("sec_fallback", True)
        d = read_json(self.f)
        n = max(1, int(n_sections))
        if d.get("n") != n:
            d = {}
        self.n = n
        self.out = [list(x)[-self.WINDOW:] for x in d.get("out", [[] for _ in range(n)])]
        self.mode = list(d.get("mode", ["pb"] * n)) if self.on else ["pb"] * n
        self.since = list(d.get("since", [0.0] * n))
        self.retry = list(d.get("retry", [self.RETRY_S] * n))
        self.wr = [0] * n
        self.last_write, self.last_log = 0.0, [0.0] * n
        if not self.on and "ai" in d.get("mode", []):          # fallback switched off: back to the PB/WR line
            self.write()
            if self.guide:
                build_guide(self.run_dir)
        self.write()

    def add(self, info):
        """One finished training lap (any end)."""
        if info.get("end") in ("car restarted", "gym lost"):
            return
        cp0, j = int(info.get("cp0", 0)), int(info.get("cp", 0))
        for k in range(max(cp0, 0), min(j, self.n)):
            self.out[k].append(0)
        if j < self.n and info.get("end") in FAIL_ENDS:
            self.out[j].append(1)
        for k in range(self.n):
            del self.out[k][:-self.WINDOW]
        for k in info.get("secs_wr") or []:
            if 0 <= k < self.n:
                self.wr[k] += 1

    def rate(self, k):
        o = self.out[k]
        return sum(o) / len(o) if len(o) >= 10 else None

    def _ai_route(self, k):
        return (self.run_dir / f"sec_best_{k}.npy").exists() or (self.run_dir / "best_trace.npy").exists()

    def _set(self, k, mode, why, log):
        old, self.mode[k] = self.mode[k], mode
        now = time.time()
        self.out[k], self.wr[k], self.since[k] = [], 0, now
        tgt = {"pb": "your PB/WR route", "try": "your PB/WR route (trial)", "ai": "the AI's own best route"}[mode]
        if now - self.last_log[k] >= self.LOG_S:              # at most one line per section per 10 min
            self.last_log[k] = now
            say(f"Section {k + 1} (checkpoint {k} -> {k + 1}): target is now {tgt} - {why}")
        if log:
            log(k="section", sec=k, mode=mode, old=old, why=why)

    def update(self, log=None):
        """Decide switches; call every few seconds."""
        changed, now = False, time.time()
        for k in range(self.n if self.on else 0):
            r, m, n_out = self.rate(k), self.mode[k], len(self.out[k])
            if now - float(self.since[k] or 0.0) < self.HOLD_S:     # just switched: keep the target for now
                continue
            if m == "pb" and r is not None and n_out >= self.MIN_N and r > self.FAIL + self.MARGIN and self._ai_route(k):
                self._set(k, "ai", f"{r:.0%} of the laps reaching it fail there", log)
                changed = True
            elif m == "ai" and self.wr[k] >= self.WR_N:
                self._set(k, "pb", "the AI got through on the PB/WR route", log)
                self.retry[k] = self.RETRY_S
                changed = True
            elif m == "ai" and now - self.since[k] > self.retry[k]:
                self._set(k, "try", f"periodic retry after {self.retry[k] / 60:.0f} min", log)
                changed = True
            elif m == "try" and n_out >= self.TRY_N:
                if r is not None and r <= 0.5 - self.MARGIN:
                    self._set(k, "pb", f"retry worked ({r:.0%} fail)", log)
                    self.retry[k] = self.RETRY_S
                else:
                    self.retry[k] = min(self.retry[k] * 2, 3600.0)
                    self._set(k, "ai", f"retry failed ({r:.0%} fail); next retry in {self.retry[k] / 60:.0f} min", log)
                changed = True
        if changed:
            self.write()
            try:
                r = build_guide(self.run_dir)
                if r:
                    say(f"Target line sections: {' '.join(r[1])} (target pace {r[0] / 1000:.3f} s)")
            except Exception as e:
                say(f"Could not rebuild the target line ({str(e)[:80]})")
        elif now - self.last_write > 30:
            self.write()

    def write(self):
        self.last_write = time.time()
        rates = [self.rate(k) for k in range(self.n)]
        write_json(self.f, {"n": self.n, "mode": self.mode, "rate": [round(x, 3) if x is not None else None for x in rates],
                            "laps": [len(o) for o in self.out], "out": self.out, "since": self.since,
                            "retry": self.retry, "t": time.time()})


def n_sections(run_dir, env_route_len=None):
    """Number of checkpoint sections (checkpoints + finish): from the PB trace if there is one."""
    try:
        return len(_cp_cuts(np.load(Path(run_dir) / "pb_trace.npy"))) - 1
    except (OSError, ValueError):
        return env_route_len


class TMEnv(gym.Env):
    """One car on one map. Action = (steer level, throttle mode)."""

    LOOK = (10, 25, 50, 90, 140)      # route lookahead in metres
    PREVIEW = (15, 30, 45, 60, 80, 100, 130, 160, 200, 250)    # track memory: where the line goes, far ahead
    N_HIST = 5                            # short memory: the last 5 decisions' slide / rotation / speed
    N_MEM = 2 * len(PREVIEW) + 3 * N_HIST
    PATH3D = 40                           # Linesight-style: 40 line points, every 10 m, x/y/z in the car frame
    N_PATH = 3 * PATH3D + 12              # + 3 more previous actions (5 in total)
    NEAR_LINE = 15.0                      # m (horizontal): closer than this the car is "on" the reference line
    FAR_LINE, FAR_TICKS = 40.0, 800       # end a lap after 8 s continuously further than 40 m off the line
    STALL_TICKS = 400                     # on the line, moving, but no progress along it: 4 s grace
    _ARCS = np.array(LOOK + PREVIEW + tuple(10.0 * k for k in range(1, PATH3D + 1)))
    N_DIFF = 1                            # how often laps fail in the current section (0..1)
    WHEELS4 = ("fl", "fr", "rl", "rr")
    SURF_RAYS = (32, 33, 34, 35, 14, 15, 16, 17, 13, 18)   # 4 downward rays + 6 centre fan rays: surface ahead
    N_ICE = 11                            # per wheel icing state + blend, front/rear icing grip, any wheel on ice/snow
    N_SURF_IN = 4 * (N_SURF + 1) * 2 + len(SURF_RAYS) * (N_SURF + 1) + N_ICE   # surface block (last non-lidar inputs)
    N_BASE = N_BASE0 + len(EXTRA_NAMES) + N_MEM + N_PATH + N_DIFF + N_SURF_IN   # non-lidar features
    DIFF_COL = N_BASE0 + len(EXTRA_NAMES) + N_MEM + N_PATH                       # where the difficulty input sits
    LOOK_OFF = 28                         # where the lookahead features start

    def __init__(self, cfg, run_dir=None, shorten=True, role="training"):
        self.W = {k: float(cfg[k]) for k in cfgmod.WEIGHTS}
        fw = read_json(Path(run_dir) / "rewards.json").get("weights") if run_dir else None
        if fw:                                        # run started with a preset (minimal): its own weights
            self.W.update({k: float(fw.get(k, 0.0)) for k in cfgmod.WEIGHTS})
        self.iqn = cfg.get("algo") == "iqn"          # IQN charges failure itself: no give-up penalty here
        self.STEER = STEER_LEVELS[cfg["steer_mode"]]
        if self.iqn and int(cfg.get("_iqn_levels") or 21) != len(self.STEER):     # finer IQN steering (new runs)
            self.STEER = steer_levels(int(cfg["_iqn_levels"]), cfg["steer_mode"])
        self.slug, self.pool_car = cfg.get("_slug") or None, bool(cfg.get("_pool"))   # base / pool map tags
        self.pnorm, self.pscale = float(cfg.get("_pnorm") or 0.0), 1.0     # map-neutral progress (base runs)
        self.cut_behind = float(cfg["cut_behind_s"])
        self.ghost_window = float(cfg["ghost_window_s"])
        self.vcp_every = float(cfg["vcp_every_s"])
        self.vcp_s, self.vcp_t, self.vcp_i = np.zeros(0), np.zeros(0), 0
        self.tf = cfg["ticks_per_step"] / 5.0          # per-step costs are scaled so they mean the same per second
        self.run_dir, self.role, self.shorten = (Path(run_dir) if run_dir else None), role, shorten
        self.dense, self._line_m, self.prof, self._prof_m, self.champ_ms = None, None, None, None, None
        self.phi, self._prev_lv, self.since_hit, self.air = 0.0, None, 20, 0
        self.hist = []
        self.extra = bool(cfg["extra_obs"])
        self.lidar_every = max(1, int(cfg.get("lidar_every", 1)))    # >1: scan every N steps, reuse it in between
        self._scan, self._scan_n = None, 0
        self.tm = {"tick": 0.0, "obs": 0.0, "env": 0.0, "n": 0}        # time spent per part, this lap (profiling)
        self.groups = GROUPS_EXTRA if self.extra else GROUPS
        self.gspd = None
        self.mid_p = float(cfg["iqn_mid_start_p" if self.iqn else "mid_start_p"]) if role == "training" else 0.0
        self.mid_every = max(20, int(cfg["mid_start_every_s"] * 100 / cfg["ticks_per_step"]))
        self.starts, self.fails, self._ca_m, self.start_i, self.acts = [], np.zeros(0), "unset", None, []
        self._starts_line = "unset"
        self.guide = bool(cfg.get("pb_file")) and self.run_dir is not None and (self.run_dir / "pb_trace.npy").exists()
        self.cprof, self._cprof_key = None, None
        self.cp_arc, self.off_line, self.far, self.pbuf, self.nbuf = None, 0.0, 0, [], []
        self.hit_factor, self._ramp_m = 1.0, None
        self.stall, self.cp0, self.seg0, self.sec_wr, self._wr_max = 0, 0, None, [], 0.0
        self.sec_rate, self.sec_mode, self._sec_m, self.pb_dense, self.pb_cut = [], [], None, None, None
        # coverage (novelty -> speed focus): per cov_len-m stretch of the line, how well the AI knows it (0..1),
        # sent by the main process (Coverage in CarPool). Training cars of map runs only.
        self.cov_on = role == "training" and not self.pool_car and not self.slug
        self.cov_len = max(5.0, float(cfg.get("cov_len", 25)))
        self.cov_known, self.cov_wks = np.zeros(0), np.zeros(0)
        self.cov_s0, self.cov_max, self.cov_spd = 0.0, 0.0, {}

        self.session = Client(timeout=60).open_map(cfg["map"])
        mi = getattr(getattr(self.session, "info", None), "map", None)
        self.map_uid = getattr(mi, "map_uid", None)
        if self.map_uid:
            remember_map(self.map_uid, cfg["map"], getattr(mi, "sha256", None), self.run_dir)
        # 32-ray level fan, 4 downward rays (drops/gaps), 5 forward-up rays (ceilings, overhangs)
        fan = Lidar.fan(rays=32, fov_degrees=180, range_m=120)
        down = Lidar.directions([0, 0, 0, 0], [-4, -12, -25, -45], range_m=40)
        up = Lidar.directions([0, 0, 20, -20, 0], [10, 25, 15, 15, 45], range_m=40)
        self.lidar = Lidar(fan.rays + down.rays + up.rays)
        self.ray_range = np.array([r.range_m for r in self.lidar.rays], dtype=np.float32)
        self._connect()

        # Fallback route (until a reference line exists): start -> centre of each checkpoint.
        groups = {}
        for wp in self.g.map.waypoints:
            if wp.group_id is not None:
                groups.setdefault(wp.group_id, []).append(np.asarray(wp.vertices).mean(axis=0))
        targets = [np.mean(c, axis=0) for c in groups.values()]
        start = np.asarray(self.g.map.start_position, dtype=np.float64)
        self.route = np.vstack([start] + targets)
        seg = np.diff(self.route, axis=0)
        self.seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-3)
        self.seg_dir = seg / self.seg_len[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.lap_len = float(self.cum[-1])
        self.n_seg = len(self.seg_len)
        self.multilap = self.g.map.lap_count > 1

        self.ticks = cfg["ticks_per_step"]
        self.max_ticks = int(cfg["max_lap_seconds"] * 100)
        # throttle: gas / coast / brake (+ gas+brake with gas_brake; PPO only, IQN keeps 3 modes)
        self.n_modes = (int(cfg.get("_iqn_modes") or 3) if self.iqn else (4 if cfg.get("gas_brake", True) else 3))
        # analog steering (PPO): action [steer -1..1, mode]; else [steer level, mode]
        self.analog = bool(cfg.get("analog_steer", True)) and not self.iqn
        self.nvec = [len(self.STEER), self.n_modes]
        self.action_space = action_space(self.nvec, self.analog)
        self.priv = not self.iqn                     # privileged inputs for PPO's value net (after the lidar)
        self.pb_ms = float(cfg.get("pb_seconds") or 0) * 1000.0
        self.observation_space = spaces.Box(-10.0, 10.0, (self.N_BASE + len(self.lidar.rays)
                                                          + (N_PRIV if self.priv else 0),), np.float32)
        self._last_obs = np.zeros(self.observation_space.shape, np.float32)

    # --- mid-track starts (curriculum) ----------------------------------------------------------
    def _global_progress(self, pos, taken):
        """Progress at an arbitrary point (no forward search window), also resets the search index."""
        if self.dense is not None:
            pts = self.dense[0]
            self.di = max(0, int(np.argmin(np.linalg.norm(pts - pos, axis=1))) - 5)
        else:
            self.di = 0
        return self._progress(pos, taken)

    def _build_starts(self):
        """Replay the champion's inputs once and snapshot the car every few seconds along its lap.
        The physics is deterministic, so every snapshot is exactly the champion's state at that tick."""
        if not self.run_dir or self.mid_p <= 0:
            return
        f = self.run_dir / "champion_actions.json"
        try:
            m = os.path.getmtime(f)
        except OSError:
            m = None
        done = bool(self.champ_ms) or (self.run_dir / "best_run.json").exists()   # the AI has finished a lap
        if not done:                            # no mid-track starts until then: learn the start first
            return
        if m == self._ca_m:
            return
        self._ca_m = m
        acts = read_json(f).get("acts") or [] if m else []
        old = [st["snap"] for st in self.starts]
        self.starts, self.fails = [], np.zeros(0)
        try:
            if old:
                self.g.discard(*old)
        except Exception:
            pass
        try:
            limit = max(4, getattr(self.g.info, "max_snapshots_per_session", 64) - 2)
            if len(acts) < 2 * self.mid_every:
                acts = []
            self.g.reset(seed=0)
            pa = np.zeros(8)
            pb_only = self.iqn and self.guide       # IQN: start on the WR/PB route (where it cuts), not the AI's own
            if pb_only:
                acts = []
            else:
                limit //= (2 if self.guide else 1)
            for i, (steer, mode) in enumerate(acts[:-self.mid_every // 2]):
                t = self.g.step(self.ticks, steer=int(steer), accelerate=int(mode) in GAS, brake=int(mode) in BRAKE)
                pa = np.concatenate([[steer / 127.0], MODE_VEC[int(mode)], pa[:4]])
                if t.finished:
                    break
                if (i + 1) % self.mid_every == 0 and len(self.starts) < limit:
                    self.starts.append({"snap": self.g.snapshot(), "taken": int(t.race.checkpoints_taken),
                                        "speed": float(t.vehicle.speed), "pa": pa.copy(),
                                        "pos": np.asarray(t.vehicle.position, dtype=np.float64)})
            if self.guide:                      # plus start points along your PB (its states, not its inputs)
                pb = self.run_dir / "pb_trace.json"
                rec = InputText.read(read_json(pb).get("file", "")) if pb.exists() else None
                if rec:
                    self._starts_from_record(rec, limit)
            self.fails = np.zeros(len(self.starts))
            self._starts_line = "unset"
            mk = self.run_dir / "starts_logged.json"          # log once per rebuild, not once per car
            if self.starts and (self.role != "training" or read_json(mk).get("m") != m):
                if self.role == "training":
                    write_json(mk, {"m": m})
                say(f"[{self.role}] {len(self.starts)} mid-track start points built "
                    + ("(along your PB/WR)" if pb_only else "(champion lap + your PB)"))
        except Exception:                       # snapshots unsupported / limit reached: train without
            self.starts, self.fails = self.starts[:len(self.fails)] if len(self.fails) else [], self.fails

    def _starts_from_record(self, rec, limit):
        acts = sorted((Action(max(0, int(a.tick)), a.kind, a.value) for a in rec), key=lambda a: a.tick)
        self.g.reset(seed=0)
        i, k, n0 = 0, 0, len(self.starts)
        while len(self.starts) - n0 < limit:
            hi = self.g.next_tick + self.ticks
            win = []
            while i < len(acts) and acts[i].tick < hi:
                win.append(acts[i])
                i += 1
            t = self.g.step(self.ticks, win)
            k += 1
            if t.finished or t.race.race_tick > 60000:
                break
            if k % self.mid_every == 0:
                self.starts.append({"snap": self.g.snapshot(), "taken": int(t.race.checkpoints_taken),
                                    "speed": float(t.vehicle.speed), "pa": np.zeros(8),
                                    "pos": np.asarray(t.vehicle.position, dtype=np.float64)})
        if self.starts and self.starts[-1].get("pos") is not None and len(self.starts) - n0 > 1:
            self.starts.pop()                    # skip the one right before the finish

    # --- per-section bookkeeping (failure rates, target fallback, best segment per section) ----------------
    def _read_sections(self):
        """sections.json (written by the trainer): failure rate and target mode ("pb" / "ai" / "try") per section."""
        if not self.run_dir:
            return
        f = self.run_dir / "sections.json"
        try:
            m = os.path.getmtime(f)
        except OSError:
            return
        if m == self._sec_m:
            return
        d = read_json(f)
        self._sec_m = m
        self.sec_rate = [float(x or 0.0) for x in d.get("rate", [])]
        self.sec_mode = list(d.get("mode", []))
        if self.guide and self.pb_dense is None and "ai" in self.sec_mode:
            try:
                pb = np.load(self.run_dir / "pb_trace.npy")
                self.pb_dense, self.pb_cut = pb[:, :3], _cp_cuts(pb)
            except (OSError, ValueError):
                pass

    def _difficulty(self):
        k = int(self.taken)
        return float(self.sec_rate[k]) if 0 <= k < len(self.sec_rate) else 0.0

    def _pb_off(self, pos, k):
        """Horizontal distance from the PB/WR route within section k."""
        c = self.pb_cut
        if c is None or k + 1 >= len(c):
            return 0.0
        seg = self.pb_dense[c[k]:c[k + 1] + 1]
        return float(np.min(np.hypot(seg[:, 0] - pos[0], seg[:, 2] - pos[2])))

    def _section_done(self, k, tick):
        """The car just completed section k (checkpoint k -> k+1)."""
        if (self.seg0 is not None and self.sec_mode and k < len(self.sec_mode) and self.sec_mode[k] == "ai"
                and self._wr_max < self.NEAR_LINE):           # drove the whole section on the PB/WR route
            self.sec_wr.append(k)
        self._wr_max = 0.0
        if self.seg0 is not None and self.run_dir is not None and self.role == "training":
            try:
                self._offer_segment(k, np.asarray(self.hist[self.seg0:], dtype=np.float64))
            except Exception:                                  # file trouble must never stop training
                pass
        self.seg0 = len(self.hist) - 1

    def _offer_segment(self, k, seg):
        """Keep the fastest drive through each section (any lap, also unfinished ones): fallback target route."""
        if len(seg) < 3:
            return
        f = self.run_dir / f"sec_best_{k}.npy"
        dur = seg[-1, 3] - seg[0, 3]
        try:
            old = np.load(f)
            if old[-1, 3] - old[0, 3] <= dur:
                return
        except (OSError, ValueError, IndexError):
            pass
        save_npy(f, seg)

    def _pick_start(self):
        """Mostly the sections where laps fail; every section keeps some chance."""
        w = 1.0 + self.fails
        i = int(self.np_random.choice(len(self.starts), p=w / w.sum()))
        self.fails *= 0.995
        return i

    def _blame(self, s, failed):
        """Credit a failed lap to the start point just before where it ended."""
        if not len(self.starts):
            return
        if self.start_i is not None and not failed:
            self.fails[self.start_i] *= 0.7
            return
        if failed:
            ss = [st.get("s", 0.0) for st in self.starts]
            j = int(np.searchsorted(ss, s)) - 1
            if j >= 0:
                self.fails[j] = min(self.fails[j] + 1.0, 30.0)

    # --- plumbing --------------------------------------------------------------------------
    def _connect(self):
        old = getattr(self, "g", None)
        try:
            if old is not None:
                old.close()
        except Exception:
            pass
        self.g = self.session.connect(self.groups, self.lidar if self.lidar_every == 1 else None, timeout=20)
        self._scan = None

    def _read_lidar(self, t, fresh=False):
        """(hit, distance) of the lidar rays: from the step's own scan, or with lidar_every > 1 from a separate
        raycast every N steps (reused in between)."""
        if self.lidar_every == 1:
            sc = t.lidar
        else:
            if fresh or self._scan is None or self._scan_n % self.lidar_every == 0:
                self._scan = self.g.raycast(self.lidar.rays)
            self._scan_n += 1
            sc = self._scan
        hit = np.asarray(sc.hit, dtype=bool)
        self._lid = (hit, np.where(hit, np.asarray(sc.distance, dtype=np.float32), self.ray_range))
        mat = getattr(sc, "material_id", None)
        self._lid_mat = np.asarray(mat, dtype=np.int64) if mat is not None else np.zeros(len(hit), np.int64)
        return self._lid

    def _run_weights(self):
        """rewards.json of a run started with a reward preset: its frozen weights win over the settings."""
        return read_json(self.run_dir / "rewards.json").get("weights") if self.run_dir else None

    def _read_weights(self):
        """Pick up weights saved from the dashboard (applies from the next lap on)."""
        d = self._run_weights() or read_json(cfgmod.CONFIG_PATH)
        for k in cfgmod.WEIGHTS:
            try:
                self.W[k] = float(d[k])
            except (KeyError, TypeError, ValueError):
                pass
        for k, a in (("cut_behind_s", "cut_behind"), ("ghost_window_s", "ghost_window")):
            try:
                setattr(self, a, float(d[k]))
            except (KeyError, TypeError, ValueError):
                pass

    def _read_best(self):
        self.champ_ms = read_json(self.run_dir / "best.json").get("champion_ms") if self.run_dir else None

    def _load_line(self):
        if not self.run_dir:
            return
        f = self.run_dir / "centerline.npy"
        try:
            m = os.path.getmtime(f)
            if m == self._line_m:
                return
            pts = np.load(f).astype(np.float64)
        except (OSError, ValueError):
            return
        if len(pts) < 3:
            return
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.dense, self._line_m = (pts, seg, np.concatenate([[0.0], np.cumsum(seg)])), m
        self.cp_arc = self._map_checkpoints()

    def _map_checkpoints(self):
        """Arc length of every checkpoint (and the finish) on the reference line, in driving order.
        Used as anchors: progress stays between the last taken and the next checkpoint. None = unreliable."""
        if self.dense is None or self.multilap:
            return None
        pts, _, cum = self.dense
        arcs = []
        for c in self.route[1:]:
            d = np.linalg.norm(pts - c, axis=1)
            j = int(np.argmin(d))
            if d[j] > 60.0:                      # line misses a checkpoint: no anchors
                return None
            arcs.append(float(cum[j]))
        return np.sort(np.array(arcs))

    def _load_profile(self):
        """The ghost: when the fastest run so far was at each point of the reference line."""
        if not self.run_dir or self.dense is None:
            return
        f = self.run_dir / "profile.npy"
        try:
            m = os.path.getmtime(f)
            if m == self._prof_m:
                return
            prof = np.load(f).astype(np.float64)
        except (OSError, ValueError):
            return
        if len(prof) >= 5:
            ps, pt = prof[:, 0], prof[:, 1]
            gs = np.gradient(ps) / np.maximum(np.gradient(pt / 100.0), 1e-3)       # ghost speed along the line, m/s
            gs = np.convolve(np.pad(gs, 4, mode="edge"), np.ones(9) / 9, mode="valid")
            self.prof, self.gspd, self._prof_m = (ps, pt), np.clip(gs, 0.0, 200.0), m
            self._make_vcps()

    def _load_champ(self):
        """The champion's pace on the current line: laps are cut when far behind it (not the fastest ghost)."""
        if not self.run_dir or self.dense is None:
            return
        f = self.run_dir / "champ_trace.npy"
        try:
            key = (os.path.getmtime(f), self._line_m)
        except OSError:
            return
        if key == self._cprof_key:
            return
        try:
            pr = trace_to_profile(self.dense, np.load(f))
            self.cprof, self._cprof_key = (pr[:, 0], pr[:, 1]), key
        except (OSError, ValueError, IndexError):
            pass

    def _make_vcps(self):
        """Virtual checkpoints every few seconds of the ghost's (your PB's) pace: where it was at 4 s, 8 s, ..."""
        self.vcp_s, self.vcp_t = np.zeros(0), np.zeros(0)
        if self.prof is None or self.vcp_every <= 0:
            return
        ps, pt = self.prof
        step = self.vcp_every * 100.0
        ts = np.arange(step, pt[-1] - step * 0.5, step)
        if len(ts):
            self.vcp_s, self.vcp_t = np.interp(ts, pt, ps), ts

    def _phi(self, s, tick):
        """Seconds ahead of the ghost (negative = behind)."""
        if self.prof is None:
            return 0.0
        return float((np.interp(s, self.prof[0], self.prof[1]) - tick) / 100.0)

    def _finished(self, ms):
        """Called on every finished lap: keep the fastest run ever (best_run.txt). The ghost is NOT touched here: it
        follows the champion only (champion_ghost), so the reward target doesn't move on every lucky lap."""
        try:
            self._offer_best(ms)
        except Exception:                         # file trouble must never stop training
            pass

    def _offer_best(self, ms):
        if self.run_dir is None:
            return
        f = self.run_dir / "best_run.json"
        cur = read_json(f).get("ms")
        if cur is not None and ms >= cur:
            return
        rec = self.g.export()
        ms = int(rec.finish_ms)
        if cur is not None and ms >= cur:
            return
        tmp = self.run_dir / f"best_run.{os.getpid()}.tmp"
        InputText.write(str(tmp), rec.actions)
        os.replace(tmp, self.run_dir / "best_run.txt")
        write_json(f, {"ms": ms, "source": self.role, "t": time.time()})
        save_npy(self.run_dir / "best_trace.npy", np.asarray(self.hist, dtype=np.float64))

    def _offer_profile(self, ms):
        if self.run_dir is None or self.dense is None:
            return
        f = self.run_dir / "profile.json"
        cur = read_json(f).get("ms")
        if cur is not None and ms >= cur:
            return
        if self.guide:                            # the guide builds the ghost (PB / best sections)
            return
        prof = trace_to_profile(self.dense, np.asarray(self.hist, dtype=np.float64))
        if len(prof) < 5:
            return
        save_npy(self.run_dir / "profile.npy", prof)
        write_json(f, {"ms": ms})

    def _hit_penalty(self, w, tf, hit, speed_loss):
        """Wall hit: costs in proportion to the speed it cost (a glancing touch little, >= 10 m/s lost the full
        amount), times the run's ramp-in factor (hit_ramp.json: lower while the run rarely finishes)."""
        if not hit:
            return 0.0
        return -w * tf * float(hit) * self.hit_factor * min(1.0, max(0.0, float(speed_loss)) / 10.0)

    def _read_ramp(self):
        if not self.run_dir:
            return
        f = self.run_dir / "hit_ramp.json"
        try:
            m = os.path.getmtime(f)
        except OSError:
            self.hit_factor = 1.0
            return
        if m != self._ramp_m:
            self._ramp_m = m
            self.hit_factor = float(read_json(f).get("factor", 1.0))

    def _limit_ticks(self):
        """Once a champion exists, cut hopeless laps at 1.6x its time: much faster training."""
        if self.shorten and self.champ_ms:
            return min(self.max_ticks, int(self.champ_ms / 10 * 1.6) + 300)
        return self.max_ticks

    def _progress(self, pos, taken):
        """Metres driven along the route (reference line if there is one)."""
        if self.dense is not None:
            pts, seg, cum = self.dense
            s, j = line_progress(pts, seg, cum, pos, self.di)
            lo, hi = 0.0, float(cum[-1])
            if self.cp_arc is not None:           # stay inside the current checkpoint section
                n = len(self.cp_arc)
                lo = float(self.cp_arc[taken - 1]) - 20.0 if 0 < taken <= n else (0.0 if taken <= 0 else hi)
                hi = float(self.cp_arc[taken]) + 5.0 if taken < n else hi
            off = float(np.hypot(*(pos - pts[j])[[0, 2]]))
            if off > self.NEAR_LINE:              # off the line: re-acquire it anywhere in this section
                i0 = int(np.searchsorted(cum, max(lo, 0.0)))
                i1 = int(np.searchsorted(cum, hi, side="right"))
                if i1 > i0:
                    dd = np.hypot(pts[i0:i1, 0] - pos[0], pts[i0:i1, 2] - pos[2])
                    k = int(np.argmin(dd))
                    if dd[k] < self.NEAR_LINE:
                        s, j = line_progress(pts, seg, cum, pos, i0 + k + 10)
                        off = float(dd[k])
            s = float(np.clip(s, max(lo, 0.0), hi))
            self.di = min(j, int(np.searchsorted(cum, s)) + 1)
            self.off_line = off
            return s
        self.off_line = 0.0
        if self.multilap:
            laps, i = divmod(taken, self.n_seg)
        else:
            laps, i = 0, min(taken, self.n_seg - 1)
        proj = float(np.clip(np.dot(pos - self.route[i], self.seg_dir[i]), 0.0, self.seg_len[i]))
        return laps * self.lap_len + self.cum[i] + proj

    def _route_point(self, arc):
        if self.dense is not None:
            pts, _, cum = self.dense
            arc = min(max(arc, 0.0), cum[-1])
            return np.array([np.interp(arc, cum, pts[:, k]) for k in range(3)])
        arc = arc % self.lap_len if self.multilap else min(arc, self.lap_len)
        return np.array([np.interp(arc, self.cum, self.route[:, k]) for k in range(3)])

    def _route_points(self, arcs):
        """_route_point for many arc lengths at once: [N] -> [N, 3]."""
        arcs = np.asarray(arcs, dtype=np.float64)
        if self.dense is not None:
            pts, _, cum = self.dense
            a = np.minimum(np.maximum(arcs, 0.0), cum[-1])
            return np.stack([np.interp(a, cum, pts[:, k]) for k in range(3)], 1)
        a = arcs % self.lap_len if self.multilap else np.minimum(arcs, self.lap_len)
        return np.stack([np.interp(a, self.cum, self.route[:, k]) for k in range(3)], 1)

    def _behind_champ(self, s, tick, phi):
        if self.cprof is not None:                       # only the AI's own champion; no champion yet = no cut
            return float((np.interp(s, self.cprof[0], self.cprof[1]) - tick) / 100.0)
        return 0.0

    def _pointless_release(self, t, pos, s, speed):
        """0..1: how clearly releasing is pointless here. 1 = straight road ahead and nothing in front;
        0 in corners, near obstacles, in the air or at low speed, so the agent stays free to release there."""
        if speed < 15.0 or not np.any(np.asarray(t.wheels.contact, dtype=bool)):
            return 0.0
        q = t.vehicle.rotation
        ang = 0.0
        ds = np.array([25.0, 50.0, max(60.0, speed * 1.5)])     # how much the route bends in the next ~1.5 s
        loc = to_car_frame_many(q, self._route_points(s + ds) - pos)
        ang = max(ang, float(np.max(np.abs(np.arctan2(loc[:, 0], loc[:, 2])))))
        straight = float(np.clip(1.0 - ang / 0.12, 0.0, 1.0))   # fades out from ~0 to ~7 degrees of bend
        hit, dist = self._lid
        fwd = float(np.min(dist[13:19]))                         # centre of the 32-ray fan
        clear = float(np.clip((fwd - speed * 1.2) / (speed * 0.8 + 1.0), 0.0, 1.0))
        return straight * clear

    def _privileged(self, s):
        """Value-net-only inputs (the actor sees 0): ghost / PB time left, metres to the next checkpoint and to the
        finish, gap to the ghost, failure rate of this section."""
        tick = float(getattr(self, "_tick", 0))
        length = self.dense[2][-1] if self.dense is not None else self.lap_len
        g_left = 0.0
        if self.prof is not None:
            g_left = (self.prof[1][-1] - float(np.interp(s, self.prof[0], self.prof[1]))) / 100.0 / 60.0
        pb_left = (self.pb_ms / 1000.0 - tick / 100.0) / 60.0 if self.pb_ms else 0.0
        d_cp = 0.0
        if self.cp_arc is not None and 0 <= self.taken < len(self.cp_arc):
            d_cp = (float(self.cp_arc[self.taken]) - s) / 500.0
        return [float(np.clip(g_left, -5, 5)), float(np.clip(pb_left, -5, 5)), float(np.clip(d_cp, -2, 5)),
                float(np.clip((length - s) / 1000.0, -1, 10)), float(np.clip(self.phi / 5.0, -2, 2)), self._difficulty()]

    def _surface(self, t):
        """Surface inputs: per wheel material + gameplay surface (one-hot over the shared vocabulary), material of
        the road ahead seen by the downward / centre lidar rays, and the ice state. Gameplay + ice need RUNTIME
        (extra observations); without them those parts are 0."""
        w, rt = t.wheels, getattr(t, "runtime", None)
        mats = getattr(w, "material_id", None)
        out = [onehot_ids("material", mats) if mats is not None else np.zeros(4 * (N_SURF + 1), np.float32)]
        if self.extra and rt is not None:
            gp = [int(_f(rt, f"{k}_surface_gameplay_id")) for k in self.WHEELS4]
            out.append(onehot_ids("gameplay", gp))
        else:
            out.append(np.zeros(4 * (N_SURF + 1), np.float32))
        hit = self._lid[0]
        rays = [i for i in self.SURF_RAYS if i < len(hit)]
        ahead = onehot_ids("material", self._lid_mat[rays], hit[rays])
        out.append(np.concatenate([ahead, np.zeros(len(self.SURF_RAYS) * (N_SURF + 1) - len(ahead), np.float32)]))
        if self.extra and rt is not None:
            ice = [float(np.clip(_f(rt, f"{k}_icing_state"), -5, 5)) for k in self.WHEELS4]
            ice += [float(np.clip(_f(rt, f"{k}_icing_blend"), -5, 5)) for k in self.WHEELS4]
            ice += [_f(rt, "front_wheels_icing_grip_scale", 1.0) - 1.0, _f(rt, "rear_wheels_icing_grip_scale", 1.0) - 1.0,
                    float(_f(rt, "is_any_wheel_on_ice_or_snow") > 0.5)]
            out.append(np.clip(np.asarray(ice, np.float32), -5, 5))
        else:
            out.append(np.zeros(self.N_ICE, np.float32))
        return np.concatenate(out)

    def _extras(self, t, q, pos, s):
        """41 extra features (see EXTRA_NAMES). All zero when extra observations are switched off."""
        n = len(EXTRA_NAMES)
        if not self.extra:
            return [0.0] * n
        v, w, rt, ef = t.vehicle, t.wheels, t.runtime, t.effects
        lv = to_car_frame(q, v.linear_velocity)
        acc = np.zeros(3) if self._prev_lv is None else (lv - self._prev_lv) / (self.ticks * 0.01)
        self._prev_lv = lv
        out = list(np.clip(acc / 20.0, -5.0, 5.0))                                       # acceleration (collisions, traction)
        out += list(np.asarray(w.angular_velocity, dtype=np.float32) / 300.0)             # wheel spin
        out += [0.0]          # old "icing" slot: kept 0 (trained checkpoints expect it); real ice inputs: _surface()
        hit = _f(rt, "wall_hit_this_step")
        out += [_f(rt, "forward_motion_alignment_scale", 1.0), hit, _f(rt, "roof_contact_this_step"),
                _f(rt, "wetness"), _f(rt, "surface_grip_scale", 1.0), _f(rt, "soft_surface_wheel_count") / 4.0,
                0.0, _f(v, "in_water")]     # old "on_ice" slot: kept 0, see _surface()
        out += [_f(rt, "accel_coef", 1.0) - 1.0, _f(rt, "control_coef", 1.0) - 1.0,
                _f(rt, "gravity_coef", 1.0) - 1.0, _f(rt, "adherence_coef", 1.0) - 1.0]
        rdir = _f(ef, "reactor_direction")
        out += [_f(ef, "turbo") / 5.0, min(_f(ef, "turbo_remaining_ms") / 6000.0, 1.0), _f(ef, "reactor") / 2.0,
                float(rdir == 1) - float(rdir == 2), float(_f(ef, "engine") == 1), _f(ef, "no_brakes"),
                _f(ef, "no_steering"), _f(ef, "no_grip"), _f(ef, "fragile"), _f(ef, "slow_motion", 1.0) - 1.0,
                _f(ef, "cruise_speed") / 100.0]
        if self.prof is not None and self.gspd is not None:                              # the ghost's pace here and ahead
            ps = self.prof[0]
            g0, g1, g2 = (float(np.interp(s + d, ps, self.gspd)) for d in (0.0, 25.0, 60.0))
            out += [float(np.clip(self.phi / 5.0, -2.0, 2.0)), g0 / 100.0, g1 / 100.0, g2 / 100.0,
                    float(np.clip((float(v.speed) - g0) / 50.0, -2.0, 2.0))]
        else:
            out += [0.0] * 5
        length = self.dense[2][-1] if self.dense is not None else self.lap_len
        cp, ahead = self._route_points([s, s + 5.0])
        off, tan = to_car_frame(q, cp - pos), to_car_frame(q, ahead - cp)
        out += [min(s / max(length, 1.0), 2.0), float(np.clip(off[0] / 10.0, -3.0, 3.0)),
                float(np.arctan2(tan[0], tan[2]) / np.pi)]
        self.since_hit = 0 if hit else min(self.since_hit + 1, 20)
        self.air = 0 if np.any(np.asarray(w.contact, dtype=bool)) else min(self.air + 1, 20)
        out += [1.0 - self.since_hit / 20.0, self.air / 20.0]
        return out

    def _obs(self, t, s):
        self._last_t = t
        v, w, d = t.vehicle, t.wheels, t.drivetrain
        q, pos = v.rotation, np.asarray(v.position, dtype=np.float64)
        f = [v.speed / 100.0]
        f += list(to_car_frame(q, v.linear_velocity) / 100.0)
        f += list(to_car_frame(q, v.angular_velocity) / 10.0)
        f += list(to_car_frame(q, [0.0, 1.0, 0.0]))                 # world "up" in the car frame (roll/pitch)
        f += list(np.asarray(w.contact, dtype=np.float32))
        f += list(np.asarray(w.slipping, dtype=np.float32))
        f += [d.steer / 127.0, d.gear / 5.0]
        f += list(self.pa)                                          # last two actions (input latency)
        nl, npv = len(self.LOOK), len(self.PREVIEW)                 # all route points of this step in one go
        locs = to_car_frame_many(q, self._route_points(s + self._ARCS) - pos)
        for loc in locs[:nl]:                                       # where the route goes next
            n = float(np.linalg.norm(loc)) + 1e-6
            f += [float(np.arctan2(loc[0], loc[2]) / np.pi), float(loc[1] / n)]
        f += self._extras(t, q, pos, s)
        for loc in locs[nl:nl + npv]:                               # track memory: line ahead, far beyond the lidar
            n = float(np.linalg.norm(loc)) + 1e-6
            f += [float(np.arctan2(loc[0], loc[2]) / np.pi), float(loc[1] / n)]
        lv, av = to_car_frame(q, v.linear_velocity), to_car_frame(q, v.angular_velocity)
        self.mem = self.mem[1:] + [[float(np.clip(lv[0] / 30.0, -3, 3)), float(np.clip(av[1] / 5.0, -3, 3)),
                                    float(v.speed) / 100.0]]
        f += [x for row in self.mem for x in row]                    # oldest first
        f += list(np.clip(locs[nl + npv:] / 100.0, -5.0, 5.0).ravel())   # the line ahead in 3D, up to 400 m
        f += list(self.pa_old)
        f.append(self._difficulty())                                 # section failure rate (exploration focus)
        f += list(self._surface(t))                                  # what the wheels are on, what is ahead, ice
        hit, dist = self._lid
        f += list(np.minimum(dist, self.ray_range) / self.ray_range)
        if self.priv:
            f += self._privileged(s)
        self._last_obs = np.clip(np.nan_to_num(np.asarray(f, dtype=np.float32)), -10, 10)
        return self._last_obs

    # --- gym API ---------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._read_weights()
        self._read_best()
        self._load_line()
        self._load_profile()
        self._load_champ()
        try:
            self._build_starts()                # replays runs to take snapshots: must happen BEFORE the reset
        except TransportError:
            self._connect()
        try:
            t = self.g.reset(seed=0)            # physics is deterministic
        except TransportError:
            self._connect()
            t = self.g.reset(seed=0)
        self.taken, self.slow, self.prev_speed, self.di = 0, 0, 0.0, 0
        self._prev_lv, self.since_hit, self.air = None, 20, 0
        self.pa = np.zeros(8)
        self.mem = [[0.0, 0.0, 0.0]] * self.N_HIST
        self.pa_old = np.zeros(12)
        self.sbuf, self.pbuf, self.nbuf, self.far, self.stall = [], [], [], 0, 0
        self.limit = self._limit_ticks()
        if self.pnorm > 0:                     # every map's full lap pays like a pnorm-metre lap (clipped 0.2..5 so
            length = self.dense[2][-1] if self.dense is not None else self.lap_len   # driving always beats waiting)
            self.pscale = float(np.clip(self.pnorm / max(length, 1.0), 0.2, 5.0))
        self.start_i, self.acts = None, []
        self.rsum, self.rsteps = {}, 0
        self.cov_spd = {}
        self.ast = np.zeros(9)      # this lap: steps per throttle mode (4), sum |steer|/127, full-lock steps, speed, n, max m
        if self.starts and self.np_random.random() < self.mid_p:
            try:
                i = self._pick_start()
                st = self.starts[i]
                t = self.g.restore(st["snap"])
                self.start_i, self.taken, self.prev_speed, self.pa = i, st["taken"], st["speed"], st["pa"].copy()
            except Exception:                   # snapshot lost: fall back to a normal start
                self.starts, self.fails, self._ca_m = [], np.zeros(0), None
                t = self.g.reset(seed=0)
        if self.starts and self._starts_line != self._line_m:    # start points' track position (for blame)
            for st in self.starts:
                st["s"] = self._global_progress(st["pos"], st["taken"])
            self._starts_line = self._line_m
        if self.start_i is None and int(t.race.race_tick) > 0:     # never hand a car a half-driven lap
            t = self.g.reset(seed=0)
        pos = np.asarray(t.vehicle.position, dtype=np.float64)
        tick = int(t.race.race_tick) if self.start_i is not None else 0
        self._tick = tick
        self.di = 0
        self._s = self._global_progress(pos, self.taken) if self.start_i is not None else self._progress(pos, 0)
        self.phi = self._phi(self._s, tick)
        self.cov_s0 = self.cov_max = float(self._s)
        self.vcp_i = int(np.searchsorted(self.vcp_s, self._s, side="right"))
        self.hist = [[*pos, tick, self.taken]]
        self.cp0, self.sec_wr, self._wr_max = self.taken, [], 0.0
        self.seg0 = 0 if self.start_i is None else None     # section segments only from a checkpoint crossing
        self._read_sections()
        self._read_ramp()
        self._read_lidar(t, fresh=True)
        return self._obs(t, self._s), {}

    def step(self, action):
        watchdog()
        mode = int(np.clip(np.round(float(action[1])), 0, self.n_modes - 1))
        steer = steer_value(action[0], self.STEER, self.analog)
        self.acts.append((steer, mode))
        c0 = time.perf_counter()
        try:
            t = self.g.step(self.ticks, steer=steer, accelerate=mode in GAS, brake=mode in BRAKE)
            self._read_lidar(t)
            c1 = time.perf_counter()
            self.tm["tick"] += c1 - c0                           # TICK round trips (step + any raycast)
        except (TransportError, TimeoutError, ConnectionError):   # game hiccup / no answer in 20 s: new gym, end lap
            self._connect()
            info = {"finished": False, "race_ms": 0, "cp": self.taken, "tick": 0, "end": "gym lost",
                    "cp0": self.cp0}
            return self._last_obs, 0.0, False, True, info

        W, r = self.W, {}
        pos = np.asarray(t.vehicle.position, dtype=np.float64)
        taken, tick = int(t.race.checkpoints_taken), int(t.race.race_tick)
        self._tick = tick
        s = self._progress(pos, taken)
        speed = float(t.vehicle.speed)
        dp = s - self._s                      # _s = progress already paid; a skipped stretch is paid at
        dp = min(dp, 10.0) if dp >= 0 else max(dp, -10.0)   # <= 10 m per step, a backward jump costs once
        r["progress"] = W["w_progress"] * dp * self.pscale
        kn, ck = self._known(self._s)
        if kn is not None and dp > 0 and (W["w_novel"] or W["w_known"]):    # new stretch pays more, known less
            r["novel"] = r["progress"] * (W["w_novel"] * (1.0 - kn) - min(W["w_known"], 0.5) * self._wks(ck) * kn)
        tf = self.tf
        r["time"] = -W["w_time"] * tf
        r["steer"] = -W["w_steer_smooth"] * abs(steer / 127.0 - self.pa[0])
        if taken > self.taken:
            r["checkpoint"] = W["w_checkpoint"]
            self.taken = taken
        phi = self._phi(s, tick)                                                 # seconds ahead of the ghost
        while self.vcp_i < len(self.vcp_s) and s >= self.vcp_s[self.vcp_i]:   # virtual checkpoint passed
            split = (self.vcp_t[self.vcp_i] - tick) / 100.0                      # seconds ahead of the PB/ghost here
            r["vcp"] = r.get("vcp", 0.0) + W["w_vcp"] + float(np.clip(W["w_vsplit"] * split, -3.0, 3.0))
            self.vcp_i += 1
        if self.prof is not None and W["ghost_weight"]:
            # time gained on the ghost this step, but only inside a window behind it: once a lap is far
            # behind, losing more time costs nothing extra, so ending the lap early can never pay off
            lo = -max(self.ghost_window, 0.1)
            r["ghost"] = float(np.clip(W["ghost_weight"] * (max(phi, lo) - max(self.phi, lo)), -5.0, 5.0))
            if kn is not None and W["w_focus"]:                              # known stretch: time gained counts more
                r["focus"] = r["ghost"] * W["w_focus"] * kn
        self.phi = phi
        if self.guide and W["w_guide"] and self.dense is not None:            # pulled toward the target line,
            off = pos - self._route_point(s)                                     # but only while slower than it
            d = float(np.hypot(off[0], off[2]))
            behind = float(np.clip(-phi / 0.3, 0.0, 1.0))
            r["guide"] = -W["w_guide"] * tf * min(d / 8.0, 1.0) * behind
        r["crash"] = -W["w_crash"] * max(0.0, self.prev_speed - speed - 3.0)    # sudden speed loss = wall hit
        if mode in BRAKE:
            r["brake"] = -W["w_brake"] * tf
        elif mode == 1:
            r["coast"] = -W["w_coast"] * tf
            if W["w_release"]:
                r["release"] = -W["w_release"] * tf * self._pointless_release(t, pos, s, speed)
        if W["w_slip"]:
            r["slip"] = -W["w_slip"] * tf * float(np.mean(np.asarray(t.wheels.slipping, dtype=np.float32)))
        if W["w_air"] and not np.any(np.asarray(t.wheels.contact, dtype=bool)):
            r["air"] = -W["w_air"] * tf
        if W["w_speed"]:
            r["speed"] = W["w_speed"] * tf * speed / 100.0
        if W["w_hit"] and self.extra:
            r["hit"] = self._hit_penalty(W["w_hit"], tf, _f(t.runtime, "wall_hit_this_step"), self.prev_speed - speed)
        if W["w_wall"]:                                                          # side rays of the fan
            hit, dist = self._lid
            near = float(np.min(np.concatenate([dist[:4], dist[28:32]])))
            r["wall"] = -W["w_wall"] * tf * speed / 100.0 * max(0.0, 1.0 - near / 2.0)
        self.prev_speed = speed
        self._s = self._s + dp if s >= self._s else s
        a = self.ast
        a[mode] += 1
        a[4] += abs(steer) / 127.0
        a[5] += abs(steer) >= 127
        a[6] += speed
        a[7] += 1
        a[8] = max(a[8], self._s)
        if self.cov_on:
            self.cov_max = max(self.cov_max, self._s)
            a = self.cov_spd.setdefault(ck, [0.0, 0])
            a[0] += speed
            a[1] += 1
        self.pa_old = np.concatenate([self.pa[4:8], self.pa_old[:8]])
        self.pa = np.concatenate([[steer / 127.0], MODE_VEC[mode], self.pa[:4]])
        prev_taken = self.hist[-1][4] if self.hist else taken
        self.hist.append([*pos, tick, taken])
        if self.sec_mode and taken < len(self.sec_mode) and self.sec_mode[taken] == "ai" and self.pb_dense is not None:
            self._wr_max = max(self._wr_max, self._pb_off(pos, taken))   # is it driving the PB/WR route here?
        if taken > prev_taken:
            self._section_done(int(prev_taken), tick)

        end, terminated, truncated = "", False, False
        length = self.dense[2][-1] if self.dense is not None else self.lap_len
        # Ending a lap early (stuck / off the track) costs what driving the rest of it would have paid,
        # so giving up is never cheaper than pushing on through a bad section.
        give_up = W["w_progress"] * max(0.0, length - s) * self.pscale
        if self.iqn:                            # IQN charges failure itself (time cost for the rest of the mini-race)
            give_up = 0.0
        if t.finished:
            terminated, end = True, "finished"
            ref = self.champ_ms if self.champ_ms else self.max_ticks * 10
            gain = (ref - int(t.race.race_time_ms)) / 1000.0                     # seconds faster than the champion
            r["finish"] = W["w_finish"] + float(np.clip(W["w_finish_time"] * gain, -20.0, 20.0))
            if self.start_i is None:            # mid-track laps are not full runs: not saved as best run / ghost
                self._finished(int(t.race.race_time_ms))
        elif t.race.entered_off_zone:
            terminated, end = True, "off-zone"
            r["end"] = -(W["w_offzone"] + give_up)
        else:
            self.slow = self.slow + self.ticks if (speed < 1.5 and tick > 300) else 0
            nb = max(2, 200 // self.ticks)
            self.sbuf = (self.sbuf + [s])[-nb:]
            self.pbuf = (self.pbuf + [pos])[-nb:]
            self.nbuf = (self.nbuf + [self.off_line < self.NEAR_LINE])[-nb:]
            full = len(self.sbuf) >= nb and tick > 300
            moved = float(np.sum(np.linalg.norm(np.diff(np.asarray(self.pbuf), axis=0), axis=1)))
            if all(self.nbuf) and full and self.sbuf[-1] - self.sbuf[0] < 2.0:
                # on the line but not getting along it: if the car still moves (e.g. the line jumps a gap or
                # goes through a wall it can't take), give it 4 s more before calling it stuck
                self.stall = self.stall + self.ticks if moved >= 3.0 else self.STALL_TICKS
            else:
                self.stall = 0
            no_progress = full and (moved < 3.0 or self.stall >= self.STALL_TICKS)   # distance actually driven
            self.far = self.far + self.ticks if self.off_line > self.FAR_LINE else 0
            if self.slow >= 200 or no_progress:                                 # stuck
                terminated, end = True, "stuck"
                r["end"] = -(W["w_stuck"] + give_up)
            elif self.far >= self.FAR_TICKS:                                    # lost, far off the line
                terminated, end = True, "far off line"
                r["end"] = -(W["w_stuck"] + give_up)
            elif (self.role == "training" and self.cut_behind and tick > 500
                  and self._behind_champ(s, tick, phi) < -self.cut_behind):      # tests always drive the full lap
                truncated, end = True, "behind ghost"                            # hopeless lap: save the time
            elif tick >= self.limit:
                truncated, end = True, "timeout"
        for k, v in r.items():
            if v or k in self.rsum:                  # terms whose weight is 0 never show up in the breakdown
                self.rsum[k] = self.rsum.get(k, 0.0) + v
        self.rsteps += 1
        reward = float(sum(r.values()))
        if terminated or truncated:
            self._blame(s, end not in ("finished",))
        info = {"finished": bool(t.finished), "race_ms": int(t.race.race_time_ms),
                "cp": taken, "tick": tick, "end": end, "mid": self.start_i is not None, "cp0": self.cp0,
                "secs_wr": list(self.sec_wr), "map": self.slug, "pool": 1 if self.pool_car else 0,
                "n_seg": int(self.n_seg),
                "x": round(float(pos[0]), 1), "z": round(float(pos[2]), 1)}
        c2 = time.perf_counter()
        obs = self._obs(t, s)
        c3 = time.perf_counter()
        self.tm["obs"] += c3 - c2
        self.tm["env"] += c2 - c1                                # rewards, progress, stuck rules
        self.tm["n"] += 1
        info["aux"] = (float(_f(t.runtime, "wall_hit_this_step") if self.extra else 0.0) or float(r["crash"] < 0),
                       float(np.any(np.asarray(t.wheels.contact, dtype=bool))), tick, taken)
        if terminated or truncated:
            info["rb"], info["rsteps"] = dict(self.rsum), self.rsteps
            a, n = self.ast, max(self.ast[7], 1)
            info["ast"] = {"m": [round(float(x) / n, 3) for x in a[:self.n_modes]], "sa": round(float(a[4] / n), 3),
                           "fl": round(float(a[5] / n), 3), "v": round(float(a[6] / n), 1), "s": round(float(a[8]), 1)}
            if self.cov_on:
                info["cov"] = self._cov_info(length, bool(t.finished))
            info["tm"], self.tm = self.tm, {"tick": 0.0, "obs": 0.0, "env": 0.0, "n": 0}
        return obs, float(reward), terminated, truncated, info

    # --- coverage (novelty -> speed focus) ---------------------------------------------------------------
    def _known(self, s):
        """(known 0..1 or None when coverage is off, stretch index) at progress s."""
        k = int(max(s, 0.0) // self.cov_len)
        if not self.cov_on:
            return None, k
        return (float(self.cov_known[k]) if k < len(self.cov_known) else 0.0), k

    def _wks(self, k):
        """w_known scale of stretch k (< 1 where the reduced progress pay would not beat the time cost)."""
        return float(self.cov_wks[k]) if k < len(self.cov_wks) else 1.0

    def set_cov(self, d):
        self.cov_known = np.asarray(d.get("known", []), dtype=np.float64)
        self.cov_wks = np.asarray(d.get("wks", []), dtype=np.float64)

    def _cov_info(self, length, finished):
        """Stretches this lap drove through completely: [k0, k1). A lap failing inside a stretch doesn't count it."""
        L = self.cov_len
        k0 = int(np.ceil(self.cov_s0 / L - 1e-9))
        k1 = int(np.ceil(length / L)) if finished else int(self.cov_max // L)
        spd = {int(k): round(a[0] / a[1], 2) for k, a in self.cov_spd.items() if a[1] and k0 <= k < k1}
        cpa = [round(float(x), 1) for x in self.cp_arc] if self.cp_arc is not None else []
        return {"k0": k0, "k1": max(k0, k1), "spd": spd, "cp_arc": cpa, "ps": round(self.pscale, 4),
                "n": int(np.ceil(length / L))}

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


class StuckWatch:
    """Notices when most laps end "stuck" at the same spot and says where (once every 10 minutes at most)."""

    def __init__(self, n=200):
        self.n, self.ends, self.last = n, [], 0.0

    def add(self, info):
        if info.get("x") is None:
            return
        self.ends = (self.ends + [(info.get("end", ""), float(info["x"]), float(info["z"]), int(info.get("cp", 0)))])[-self.n:]
        if len(self.ends) < self.n or time.time() - self.last < 600:
            return
        st = [e for e in self.ends if e[0] == "stuck"]
        if len(st) < 0.8 * self.n:
            return
        xz = np.array([[e[1], e[2]] for e in st])
        mid = np.median(xz, axis=0)
        near = np.linalg.norm(xz - mid, axis=1) < 30.0
        if near.sum() < 0.8 * self.n:
            return
        cps = [e[3] for e, k in zip(st, near) if k]
        self.last = time.time()
        say(f"Hint: {int(near.sum()) * 100 // self.n}% of the last {self.n} laps ended 'stuck' within 30 m of "
            f"x={mid[0]:.0f} z={mid[1]:.0f} (after checkpoint {max(set(cps), key=cps.count)}). Something there "
            "blocks progress (e.g. the AI's route leaves the target line).")


def _make_env(cfg, run_dir):
    from stable_baselines3.common.monitor import Monitor
    raise_priority()                                  # runs inside the car's own process
    return Monitor(TMEnv(cfg, run_dir))


# --- speed: process priority, asynchronous cars, benchmark ------------------------------------------
def raise_priority(tick_too=False):
    """Windows: "above normal" priority for this process (and TICK's, if found). Needs psutil; skipped quietly."""
    if os.name != "nt":
        return
    try:
        import psutil
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        if tick_too:
            for p in psutil.process_iter(["name"]):
                n = (p.info.get("name") or "").lower()
                if "tick" in n or "trackmania" in n:
                    try:
                        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                    except Exception:
                        pass
    except Exception:
        pass


def car_worker(cfg, run_dir, conn, rot=None):
    """One car in its own process. Steps as soon as it gets an action; resets by itself when a lap ends.
    Replies (obs, reward, done, info, timing) where timing = (TICK, observation, rest of env, waited for action).
    rot (map pool): {"maps": [(cfg, run_dir), ...], "start": i, "stride": k, "every": N}: the car drives
    maps[start] and moves on by `stride` maps every N laps (only when there are more maps than such cars)."""
    raise_priority()
    try:
        idx, laps = (rot["start"] % len(rot["maps"]), 0) if rot else (0, 0)
        if rot:
            cfg, run_dir = rot["maps"][idx]
        env = TMEnv(cfg, run_dir)
        obs, _ = env.reset()
        conn.send(("ready", obs, int(env.observation_space.shape[0]), [int(x) for x in env.nvec],
                   int(env.n_seg), bool(env.analog)))
        waited = 0.0
        while True:
            w0 = time.perf_counter()
            msg = conn.recv()
            waited = time.perf_counter() - w0
            if msg is None:
                break
            if isinstance(msg, tuple):                 # (action, coverage update from the main process)
                msg, cov = msg
                env.set_cov(cov)
            tm0 = dict(env.tm)
            obs, r, term, trunc, info = env.step(msg)
            done = term or trunc
            tm1 = info.get("tm", env.tm)
            tm = tuple(tm1[k] - tm0[k] for k in ("tick", "obs", "env")) + (waited,)
            if done:
                info["terminal_observation"] = obs
                info["TimeLimit.truncated"] = bool(trunc and not term)
                r0 = time.perf_counter()
                laps += 1
                if rot and len(rot["maps"]) > rot["stride"] and laps % rot["every"] == 0:   # next pool map
                    env.close()
                    idx = (idx + rot["stride"]) % len(rot["maps"])
                    env = TMEnv(*rot["maps"][idx])
                obs, _ = env.reset()
                tm = (tm[0] + time.perf_counter() - r0,) + tm[1:]     # reset = TICK work
            conn.send(("step", obs, r, done, info, tm))
        env.close()
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception as e:
        try:
            conn.send(("error", f"{type(e).__name__}: {str(e)[:200]}"))
        except Exception:
            pass


class Coverage:
    """How well the AI knows each cov_len-m stretch of the target line (main process; saved as coverage.json).
    passes = laps that drove through the whole stretch; known = 1 - exp(-passes / cov_passes).
    A section's stretches reset when its target route swaps (PB <-> AI, sections.json)."""

    def __init__(self, cfg, run_dir):
        self.cfg, self.dir = cfg, Path(run_dir)
        self.file = self.dir / "coverage.json"
        self.L, self.tau = max(5.0, float(cfg.get("cov_len", 25))), max(1.0, float(cfg.get("cov_passes", 100)))
        d = read_json(self.file)
        if d and abs(float(d.get("len", 0)) - self.L) > 1e-6:
            say(f"Coverage: stretch length changed ({d.get('len')} -> {self.L:g} m), starting a new count.")
            d = {}
        self.passes = list(d.get("passes", []))
        self.spd = {int(k): list(v) for k, v in (d.get("spd") or {}).items()}
        self.modes, self.cp_arc = list(d.get("modes", [])), list(d.get("cp_arc", []))
        self.capped = set(int(x) for x in d.get("capped", []))
        self.ps, self.ver, self._sec_m, self._saved = 1.0, 0, None, time.time()
        self._payload = None

    def known(self):
        return 1.0 - np.exp(-np.asarray(self.passes, np.float64) / self.tau)

    def wks(self):
        """Per stretch: scale for w_known so that on a fully known stretch, progress pay at the AI's own median speed
        there still beats the time cost (with 10 % margin). Logged once per stretch."""
        W = read_json(cfgmod.CONFIG_PATH)
        def w(k):
            try:
                return float(W.get(k, self.cfg.get(k, 0.0)))
            except (TypeError, ValueError):
                return float(self.cfg.get(k, 0.0))
        wp, wt, wk = w("w_progress"), w("w_time"), min(w("w_known"), 0.5)
        ticks = self.cfg["ticks_per_step"]
        out = np.ones(len(self.passes))
        if wk <= 0:
            return out
        for k, v in self.spd.items():
            if k >= len(out) or not v:
                continue
            pay = wp * self.ps * float(np.median(v)) * ticks / 100.0        # progress pay per step at median speed
            need = 1.1 * wt * ticks / 5.0
            if pay * (1.0 - wk) < need:
                out[k] = float(np.clip((1.0 - need / pay) / wk, 0.0, 1.0)) if pay > 0 else 0.0
                if k not in self.capped:
                    self.capped.add(k)
                    say(f"Coverage: stretch {k} ({k * self.L:.0f}-{(k + 1) * self.L:.0f} m) is slow for the AI "
                        f"({np.median(v):.0f} m/s): known-stretch progress cut limited to {wk * out[k]:.2f} there "
                        "(so it never makes driving there pay less than the time cost).")
        return out

    def payload(self):
        if self._payload is None:
            self._payload = {"known": self.known().round(4).tolist(), "wks": self.wks().round(4).tolist()}
        return self._payload

    def _check_sections(self):
        """Reset the stretches of a section whose target route swapped (PB <-> AI)."""
        f = self.dir / "sections.json"
        try:
            m = os.path.getmtime(f)
        except OSError:
            return
        if m == self._sec_m:
            return
        self._sec_m = m
        modes = ["ai" if x == "ai" else "pb" for x in read_json(f).get("mode", [])]
        if self.modes and self.cp_arc:
            bounds = [0.0] + list(self.cp_arc) + [1e12]
            for k, (a, b) in enumerate(zip(self.modes, modes)):
                if a != b and k + 1 < len(bounds):
                    lo, hi = int(bounds[k] // self.L), int(np.ceil(bounds[k + 1] / self.L))
                    for j in range(lo, min(hi, len(self.passes))):
                        self.passes[j] = 0
                        self.spd.pop(j, None)
                        self.capped.discard(j)
                    say(f"Coverage: section {k + 1} now follows the {'AI' if b == 'ai' else 'PB'} route, "
                        "its stretches count as new again.")
        self.modes = modes

    def add(self, c):
        """One finished training lap's coverage info from a car (TMEnv._cov_info)."""
        n = int(c.get("n", 0))
        if n > len(self.passes):
            self.passes += [0] * (n - len(self.passes))
        if c.get("cp_arc"):
            self.cp_arc = list(c["cp_arc"])
        self.ps = float(c.get("ps", self.ps) or 1.0)
        self._check_sections()
        for k in range(int(c["k0"]), min(int(c["k1"]), len(self.passes))):
            self.passes[k] += 1
        for k, v in (c.get("spd") or {}).items():
            self.spd[int(k)] = (self.spd.get(int(k), []) + [float(v)])[-21:]
        self.ver += 1
        self._payload = None
        if time.time() - self._saved > 30:
            self.save()

    def save(self):
        try:
            write_json(self.file, {"len": self.L, "passes": self.passes, "spd": {str(k): v for k, v in self.spd.items()},
                                   "modes": self.modes, "cp_arc": self.cp_arc, "capped": sorted(self.capped)})
            self._saved = time.time()
        except Exception:
            pass


class CarPool:
    """Cars in their own processes, NOT in lock-step: each car gets its next action as soon as its own step is back,
    so nobody waits for the slowest car. Per-car order is kept (results of a car arrive in its step order)."""

    DEADLINE = float(os.environ.get("TMAI_CAR_DEADLINE", 120))   # a car step taking longer = hung car, replaced
    # (a reset that builds mid-track starts replays a whole lap: on long maps the deadline grows, see __init__)

    def __init__(self, cfg, run_dir, n, specs=None, coverage=True):
        """specs: per car (cfg, run_dir, rot) to give cars their own maps (base runs, pool cars); None = all alike."""
        import multiprocessing as mp
        from multiprocessing.connection import wait
        self._wait, self.cfg, self.run_dir = wait, cfg, str(run_dir)
        self.specs = specs or [(cfg, str(run_dir), None)] * n
        if "TMAI_CAR_DEADLINE" not in os.environ:
            self.DEADLINE = max(120.0, 3.0 * time_budget(cfg))
        self.ctx = mp.get_context("spawn")
        self.n, self.conns, self.procs = n, [None] * n, [None] * n
        for i in range(n):
            self._spawn(i)
        self.first_obs = [None] * n
        self.sent = [None] * n                       # when each car got its action (None = not stepping)
        self.prof = np.zeros(5)                      # summed seconds: tick, obs, env, waited for action; + steps
        self.replaced, self.fails = 0, {}
        self.cov = Coverage(cfg, run_dir) if coverage and cfg.get("run_type") != "base" else None   # base runs: no coverage
        self.cov_sent = [-1] * n
        for i in range(n):
            self.first_obs[i] = self._ready(i)
        self.idx = {id(c): i for i, c in enumerate(self.conns)}

    def _spawn(self, i):
        a, b = self.ctx.Pipe()
        c_cfg, c_dir, rot = self.specs[i]
        p = self.ctx.Process(target=car_worker, args=(c_cfg, str(c_dir), b, rot), daemon=True)
        p.start()
        b.close()
        self.conns[i], self.procs[i] = a, p

    def _ready(self, i, timeout=600, close_on_error=True):
        c, t0 = self.conns[i], time.time()
        timeout = max(timeout, 5.0 * time_budget(self.cfg))
        while not c.poll(5.0):                     # keep the freeze guard fed while a car starts up
            watchdog()
            if time.time() - t0 > timeout:
                raise RuntimeError(f"car {i + 1} did not start within {timeout:.0f} s")
        m = c.recv()
        if m[0] == "error":
            if close_on_error:
                self.close()
            raise RuntimeError(f"car {i + 1} could not start: {m[1]}")
        self.dim, self.nvec, self.n_seg = m[2], m[3], m[4]
        self.analog = bool(m[5]) if len(m) > 5 else False
        return m[1]

    def replace(self, i, quiet=False):
        """Kill a hung car and start a fresh one in its place; returns the new car's first observation."""
        if not quiet:
            say(f"Car {i + 1} did not answer for {self.DEADLINE:.0f} s: replacing it with a fresh one.")
        try:
            self.procs[i].terminate()
            self.procs[i].join(timeout=10)
        except Exception:
            pass
        while True:                                # a fresh car can fail to start too (TICK timeout): retry
            self._spawn(i)
            try:
                obs = self._ready(i, close_on_error=False)
                break
            except RuntimeError as e:
                now = time.time()
                f = [t for t in self.fails.get(i, []) if now - t < self.FAIL_WINDOW] + [now]
                self.fails[i] = f
                if len(f) >= self.FAIL_MAX:
                    self.close()
                    raise RuntimeError(f"car {i + 1} failed {len(f)} times in {self.FAIL_WINDOW / 60:.0f} min "
                                       f"(last: {e})")
                say(f"Replacement car {i + 1} did not start ({str(e)[:80]}); trying again.")
                try:
                    self.procs[i].terminate()
                except Exception:
                    pass
                time.sleep(5)
        self.idx = {id(c): j for j, c in enumerate(self.conns)}
        self.sent[i] = None
        self.cov_sent[i] = -1
        self.replaced += 1
        return obs

    FAIL_MAX, FAIL_WINDOW = 3, 300.0                  # the same car failing 3 times in 5 min stops the run

    def _car_failed(self, i, why):
        now = time.time()
        f = [t for t in self.fails.get(i, []) if now - t < self.FAIL_WINDOW] + [now]
        self.fails[i] = f
        if len(f) >= self.FAIL_MAX:
            self.close()
            raise RuntimeError(f"car {i + 1} failed {len(f)} times in {self.FAIL_WINDOW / 60:.0f} min (last: {why})")
        say(f"Car {i + 1} failed ({why}): replacing it with a fresh one, training goes on.")
        obs = self.replace(i, quiet=True)
        return (i, obs, 0.0, True, {"end": "car restarted", "finished": False, "race_ms": 0, "cp": 0, "cp0": 0,
                                    "TimeLimit.truncated": True})

    def hung(self):
        """Results for cars stuck past the deadline: (car, fresh obs, 0 reward, done, info) after replacing them."""
        now, out = time.time(), []
        for i, t in enumerate(self.sent):
            if t is not None and now - t > self.DEADLINE:
                obs = self.replace(i)
                out.append((i, obs, 0.0, True, {"end": "car restarted", "finished": False, "race_ms": 0, "cp": 0,
                                                "cp0": 0, "TimeLimit.truncated": True}))
        return out

    def send(self, i, action):
        self.sent[i] = time.time()
        if self.cov is not None and self.cov_sent[i] != self.cov.ver:      # coverage changed: send it along
            self.cov_sent[i] = self.cov.ver
            self.conns[i].send((np.asarray(action), self.cov.payload()))
            return
        self.conns[i].send(np.asarray(action))

    def poll(self, timeout=1.0):
        """Results that are back: list of (car, obs, reward, done, info)."""
        out = []
        for c in self._wait(self.conns, timeout):
            i = self.idx[id(c)]
            try:
                m = c.recv()
            except (EOFError, OSError):
                m = ("error", "the car process stopped unexpectedly")
            if m[0] == "error":                       # e.g. TransportError timeout: replace the car, keep training
                out.append(self._car_failed(i, m[1]))
                continue
            _, obs, r, done, info, tm = m
            self.sent[i] = None
            if done and self.cov is not None and info.get("cov") and not info.get("pool"):
                try:
                    self.cov.add(info["cov"])
                except Exception as e:                                       # never stops training
                    say(f"Coverage update failed: {str(e)[:80]}")
            self.prof[:4] += tm
            self.prof[4] += 1
            out.append((i, obs, r, done, info))
        return out

    def take_prof(self):
        p, self.prof = self.prof, np.zeros(5)
        return p

    def close(self):
        if self.cov is not None:
            self.cov.save()
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def make_pool_vecenv(cfg, run_dir, specs=None):
    """PPO's cars: lock-step like SubprocVecEnv (PPO's rollouts need it) but a hung car is replaced instead of
    freezing the whole training."""
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class PoolVecEnv(VecEnv):
        def __init__(self):
            self.pool = CarPool(cfg, run_dir, cfg["envs"], specs)
            self.cur = list(self.pool.first_obs)
            self.n_seg = self.pool.n_seg
            self.render_mode = None
            obs_space = spaces.Box(-10.0, 10.0, (self.pool.dim,), np.float32)
            VecEnv.__init__(self, cfg["envs"], obs_space, action_space(self.pool.nvec, self.pool.analog))

        def reset(self):
            return np.stack(self.cur)

        def step_async(self, actions):
            for i, a in enumerate(actions):
                self.pool.send(i, a)

        def step_wait(self):
            n = self.num_envs
            res = [None] * n
            while any(r is None for r in res):
                watchdog()
                for i, obs, r, done, info in self.pool.poll(1.0) + self.pool.hung():
                    if done and "terminal_observation" not in info:          # replaced car: last known state
                        info["terminal_observation"] = self.cur[i]
                    res[i] = (obs, r, done, info)
            self.cur = [r[0] for r in res]
            return (np.stack(self.cur), np.array([r[1] for r in res], np.float32), np.array([r[2] for r in res]),
                    [r[3] for r in res])

        def close(self):
            self.pool.close()

        def get_attr(self, attr_name, indices=None):
            if attr_name == "render_mode":
                return [None] * self.num_envs
            raise AttributeError(attr_name)

        def set_attr(self, attr_name, value, indices=None):
            pass

        def env_method(self, method_name, *a, indices=None, **k):
            return [None] * self.num_envs

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * self.num_envs

        def seed(self, seed=None):
            return [None] * self.num_envs

    return PoolVecEnv()


def prof_summary(car, wall, main=None, learner=None, tester=None, ticks=5):
    """One profiling record: where a car-step's time goes (ms per step) plus main / learner / tester numbers."""
    n = max(car[4], 1)
    rec = {"k": "prof", "steps_s": round(car[4] / max(wall, 1e-6), 1),
           "x": round(car[4] * ticks / 100.0 / max(wall, 1e-6), 1),
           "car_ms": {"tick": round(1000 * car[0] / n, 2), "obs": round(1000 * car[1] / n, 2),
                      "env": round(1000 * car[2] / n, 2), "wait": round(1000 * car[3] / n, 2)}}
    if main:
        rec["main_ms"] = {k: round(1000 * v / n, 3) for k, v in main.items()}
    if learner:
        rec["learner"] = learner
    if tester:
        rec["tester"] = tester
    return rec


def prof_line(rec):
    c = rec["car_ms"]
    s = (f"Speed {rec['x']:.1f}x real time ({rec['steps_s']:.0f} steps/s). Per car step: TICK {c['tick']:.1f} ms, "
         f"inputs {c['obs']:.1f} ms, rewards {c['env']:.1f} ms, waiting for an action {c['wait']:.1f} ms.")
    if rec.get("learner"):
        L = rec["learner"]
        s += f" Learner: {L['ups']:.1f} updates/s ({L['device']}, busy {L['busy']:.0f}%, reuse x{L['reuse']:.1f})."
    if rec.get("tester"):
        T_ = rec["tester"]
        s += f" Tests: {T_['n']} in this period, {T_['avg_s']:.1f} s each."
    return s


def bench(cfg):
    """Drive 6/8/10/12 cars (gas + random steering, no learning) ~30 s each; print the speed; suggest a car count."""
    run_dir = cfgmod.run_dir(cfg)
    secs = float(os.environ.get("TMAI_BENCH_S", 30))
    counts = [int(x) for x in os.environ.get("TMAI_BENCH_CARS", "6,8,10,12").split(",")]
    raise_priority(tick_too=True)
    results, rng = [], np.random.default_rng(0)

    def run(n, lidar_every):
        pool = CarPool(dict(cfg, mid_start_p=0.0, iqn_mid_start_p=0.0, lidar_every=lidar_every), run_dir, n, coverage=False)
        try:
            for i in range(n):
                pool.send(i, [rng.uniform(-0.2, 0.2) if pool.analog else rng.integers(8, 13), 0])
            t0, warm = time.time(), True
            while True:
                for i, *_ in pool.poll(1.0):
                    pool.send(i, [rng.uniform(-0.2, 0.2) if pool.analog else rng.integers(8, 13), 0])
                now = time.time()
                if warm and now - t0 > min(5.0, secs / 5):           # skip the start-up seconds
                    pool.take_prof()
                    t0, warm = now, False
                if not warm and now - t0 >= secs:
                    return prof_summary(pool.take_prof(), now - t0, ticks=cfg["ticks_per_step"])
        finally:
            pool.close()

    say(f"Benchmark: {', '.join(map(str, counts))} cars, {secs:.0f} s each (no learning) ...")
    for n in counts:
        try:
            rec = run(n, cfg["lidar_every"])
        except Exception as e:
            say(f"  {n} cars: failed ({str(e)[:100]})")
            continue
        results.append({"cars": n, **rec})
        say(f"  {n:2d} cars: {rec['steps_s']:7.0f} steps/s = {rec['x']:5.1f}x real time "
            f"(TICK {rec['car_ms']['tick']:.1f} ms per step)")
    best = max(results, key=lambda r: r["x"]) if results else None
    other = None
    if best:
        le = 2 if cfg["lidar_every"] == 1 else 1
        try:
            other = run(best["cars"], le)
            say(f"  {best['cars']} cars with lidar every {le} step(s): {other['x']:.1f}x real time "
                f"(vs {best['x']:.1f}x with lidar every {cfg['lidar_every']})")
            other = {"lidar_every": le, **other}
        except Exception as e:
            say(f"  lidar comparison failed ({str(e)[:100]})")
        say(f"Suggestion: {best['cars']} cars ({best['x']:.1f}x real time). Leave ~3 CPU threads for the learner, "
            "the tester and Windows.")
    write_json(run_dir / "bench.json", {"t": time.time(), "results": results, "best": best and best["cars"],
                                        "lidar_every": cfg["lidar_every"], "lidar_other": other})
    return best


def run_episode(env, model, seed=None):
    """seed None: the policy's single most likely action every step. A seed: slight, exactly reproducible
    sampling (same weights + same seed = same lap), which gets through spots where the argmax stalls."""
    import torch
    if seed is not None:
        torch.manual_seed(int(seed))
    env.analog = isinstance(model.action_space, spaces.Box)    # drive the way the model was trained
    obs, _ = env.reset()
    done, info = False, {}
    while not done:
        act, _ = model.predict(obs, deterministic=seed is None)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
    return info


# --- models: creation and warm-start migration between steering layouts ------------------------
GAMMA = [0.995]


def set_gamma(cfg):
    """Discount from a planning horizon in seconds, independent of the decision interval."""
    GAMMA[0] = float(np.clip(1.0 - cfg["ticks_per_step"] * 0.01 / max(cfg["horizon_s"], 1.0), 0.9, 0.9999))
    return GAMMA[0]


MODE_BIAS = (1.0, 0.0, -1.0, -2.0)      # fresh runs: gas ~64 %, coast 23 %, brake 9 %, gas+brake 3 % (3 modes: 69/25/6)


def make_model(env, levels=None, fresh=False, steer_t=None, steer_hard=None):
    """fresh=True: a brand-new run (not a conversion target): start NEUTRAL. With uniform throttle logits a new car
    held brake half the time (brake + gas+brake) and crawled into "stuck"; with analog steering the near-random level
    logits at temperature 0.01 made the steer mean jump between random levels (often full lock). So: throttle logits
    biased to gas, analog steer logits exactly equal (mean 0 = straight) at a smooth temperature of 1."""
    from ppo_ext import TMPPO, PrivPolicy, AnalogPolicy
    import torch
    if isinstance(env.action_space, spaces.Box):         # analog steering: the level logits feed the steer mean
        lv = list(levels if levels is not None else STEER_LEVELS["progressive"].tolist())
        t = float(steer_t if steer_t is not None else (1.0 if fresh else 0.1))
        hard = bool(steer_hard if steer_hard is not None else not fresh)     # conversion target: exact old argmax
        m = TMPPO(AnalogPolicy, env, device="cpu", verbose=0,
                  n_steps=512, batch_size=512, n_epochs=10, gamma=GAMMA[0], gae_lambda=0.95,
                  learning_rate=3e-4, ent_coef=0.005, target_kl=0.02,
                  policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), steer_levels=[int(x) for x in lv],
                                     steer_t=t, steer_hard=hard))
    else:
        lv = list(range(int(env.action_space.nvec[0])))
        m = TMPPO(PrivPolicy, env, device="cpu", verbose=0,                 # MLP policy -> CPU is the right device
                  n_steps=512, batch_size=512, n_epochs=10, gamma=GAMMA[0], gae_lambda=0.95,
                  learning_rate=3e-4, ent_coef=0.005, target_kl=0.02,    # target_kl: no destructive updates
                  policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    if fresh:
        an, n = m.policy.action_net, len(lv)
        with torch.no_grad():
            if isinstance(env.action_space, spaces.Box):
                an.weight[:n] = 0.0                       # all level logits equal -> steer mean 0 for every input
                an.bias[:n] = 0.0
            nm = an.bias.shape[0] - n
            an.bias[n:] = torch.tensor(MODE_BIAS[:nm], dtype=an.bias.dtype)
    return m


ENT_BOOST = 3.0          # entropy weight in a section where every lap fails: 1 + 3 = 4x the normal exploration


def focus_entropy(model):
    """More exploration where laps keep failing: each sample's entropy bonus is weighted by
    1 + ENT_BOOST * (failure rate of its section), read from the observation's difficulty input (DIFF_COL)."""
    pol = model.policy
    if getattr(pol, "_focus", False):
        return
    orig, col = pol.evaluate_actions, TMEnv.DIFF_COL

    def evaluate_actions(obs, actions):
        values, log_prob, entropy = orig(obs, actions)
        if entropy is not None:
            entropy = entropy * (1.0 + ENT_BOOST * obs[:, col].clamp(0.0, 1.0))
        return values, log_prob, entropy

    pol.evaluate_actions, pol._focus = evaluate_actions, True


def interp_rows(old_levels, rows, new_levels):
    """Linear interpolation over the steering-level axis."""
    old = np.asarray(old_levels, dtype=np.float64)
    out = np.empty((len(new_levels),) + rows.shape[1:], dtype=np.float64)
    for i, lvl in enumerate(new_levels):
        j = int(np.clip(np.searchsorted(old, lvl), 1, len(old) - 1))
        a = float(np.clip((lvl - old[j - 1]) / (old[j] - old[j - 1]), 0.0, 1.0))
        out[i] = (1 - a) * rows[j - 1] + a * rows[j]
    return out


def widen_columns(w_old, old_base, new_base, new_dim):
    """First-layer weights for a longer observation: old columns keep their place, new features start at 0
    (so the network behaves exactly as before until it learns to use them)."""
    out = np.zeros((w_old.shape[0], new_dim), dtype=w_old.dtype)
    out[:, :old_base] = w_old[:, :old_base]
    n_tail = w_old.shape[1] - old_base                       # the lidar block
    out[:, new_base:new_base + n_tail] = w_old[:, old_base:]
    return out


def transfer_policy(old, new, old_levels, new_levels, old_base=N_BASE0, new_base=TMEnv.N_BASE):
    """Copy a trained policy into a fresh model; the steering outputs are re-mapped by level value,
    so the new policy starts out behaving like the old one."""
    import torch
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    for k, v in sd_old.items():
        if k.startswith("action_net") or k not in sd_new:
            continue
        if sd_new[k].shape == v.shape:
            sd_new[k] = v.clone()
        elif v.ndim == 2 and k.endswith(".0.weight") and sd_new[k].shape[0] == v.shape[0]:
            wide = widen_columns(v.cpu().numpy(), old_base, new_base, int(sd_new[k].shape[1]))
            sd_new[k] = torch.tensor(wide, dtype=sd_new[k].dtype)
    n_old = len(old_levels)
    for part in ("weight", "bias"):
        key = f"action_net.{part}"
        o = sd_old[key].cpu().numpy().astype(np.float64)
        steer = interp_rows(old_levels, o[:n_old], new_levels)
        modes = o[n_old:]
        m_new = int(sd_new[key].shape[0]) - len(new_levels)
        if m_new > len(modes):                     # new gas+brake mode = the brake row, 3 lower: never the
            extra = modes[2:3].copy()               # most likely mode, so the first actions stay identical
            if part == "bias":
                extra -= 3.0
            modes = np.concatenate([modes, extra])
        sd_new[key] = torch.tensor(np.concatenate([steer, modes[:m_new]]), dtype=sd_new[key].dtype)
    new.policy.load_state_dict(sd_new)
    new.num_timesteps, new._n_updates = old.num_timesteps, old._n_updates


def saved_ticks(run_dir):
    """ticks_per_step the run's checkpoints were trained with: levels.json, else the last start event."""
    t = read_json(Path(run_dir) / "levels.json").get("ticks")
    if t:
        return int(t)
    try:
        last = None
        with open(Path(run_dir) / "events.jsonl", "rb") as f:
            for line in f:
                if b'"k": "start"' in line[:40]:
                    last = line
        if last:
            return int(json.loads(last)["cfg"]["ticks_per_step"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def n_modes_of(space):
    return int(space.high[1]) + 1 if isinstance(space, spaces.Box) else int(space.nvec[1])


def migrate(run_dir, cfg, env):
    """If the steering levels or the observation layout changed since the checkpoints were saved,
    convert them (latest + champion) so training continues instead of starting over."""
    from stable_baselines3 import PPO
    new_levels, new_base = STEER_LEVELS[cfg["steer_mode"]], TMEnv.N_BASE
    lv_file = run_dir / "levels.json"
    saved = read_json(lv_file)
    old_levels = np.array(saved.get("steer", STEER_LEVELS["uniform"].tolist()))
    old_base = int(saved.get("obs_base", N_BASE0))
    same_levels = len(old_levels) == len(new_levels) and np.array_equal(old_levels, new_levels)
    new_modes, new_dim = n_modes_of(env.action_space), int(env.observation_space.shape[0])
    old_modes, old_dim = int(saved.get("modes", 3)), int(saved.get("obs_dim", old_base + 41))
    analog = isinstance(env.action_space, spaces.Box)
    tag = "analog1" if analog else "priv1"           # analog1 = PrivPolicy + analog steering head (AnalogPolicy)
    if not same_levels or old_base != new_base or old_modes != new_modes or old_dim != new_dim \
            or saved.get("policy") != tag:
        for name in ("latest", "champion"):
            p = run_dir / f"{name}.zip"
            if not p.exists():
                continue
            try:
                old = PPO.load(str(p), device="cpu")
                op = old.policy                            # analog -> analog keeps its steer settings; levels -> hard
                new = make_model(env, new_levels, steer_t=getattr(op, "steer_t", None),
                                 steer_hard=getattr(op, "steer_hard", None) if hasattr(op, "steer_t") else True)
                transfer_policy(old, new, old_levels, new_levels, old_base, new_base)
                new.save(str(run_dir / name))
                say(f"Converted {name}.zip to the current layout (steering {len(old_levels)}->{len(new_levels)}, "
                    f"throttle modes {old_modes}->{new_modes}, inputs {old_dim}->{new_dim}, "
                    f"steering {saved.get('policy') == 'analog1' and 'analog' or 'levels'}->{analog and 'analog' or 'levels'}).")
            except Exception as e:     # never load weights under the wrong layout
                p.replace(run_dir / f"{name}.unconverted.zip")
                say(f"Could not convert {name}.zip ({str(e)[:80]}); moved aside, starting fresh.")
    write_json(lv_file, {"steer": new_levels.tolist(), "obs_base": new_base, "modes": new_modes, "obs_dim": new_dim,
                         "policy": tag, "ticks": int(cfg["ticks_per_step"])})


# --- champion tests, dashboard events, stop conditions ----------------------------------------
def pb_ms(cfg, run_dir):
    """Your PB in ms: the pb_seconds setting, else the traced PB file's time."""
    if cfg.get("pb_seconds"):
        return float(cfg["pb_seconds"]) * 1000.0
    ms = read_json(Path(run_dir) / "pb_trace.json").get("ms") if cfg.get("pb_file") else None
    return float(ms) if ms else None


def test_tries(cfg, run_dir):
    """Best-of-N for a champion test: test_tries normally, test_tries_close once the champion is within
    test_close_s of the PB (more seeded laps find the rare faster one; still exactly reproducible)."""
    champ, pb = read_json(Path(run_dir) / "best.json").get("champion_ms"), pb_ms(cfg, run_dir)
    if champ and pb and champ - pb <= cfg["test_close_s"] * 1000.0:
        return max(cfg["test_tries"], cfg["test_tries_close"])
    return cfg["test_tries"]


def time_budget(cfg):
    """Seconds a lap may take in the worst case (long maps need longer timeouts)."""
    return float(cfg["max_lap_seconds"])


def ppo_test_worker(cfg, run_dir, inbox, busy, maps=None):
    """Champion tests for PPO in their own process (spawn): best of N laps with the EMA weights it is sent.
    Writes champion.zip/.txt, champ_trace, champion_actions, the reference line; asks for a rollback via ROLLBACK.
    maps (base run): [(cfg, map folder, slug), ...] tested round-robin, one map per test, files in that map's
    folder; no rollbacks (one shared model)."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    raise_priority()
    run_dir = Path(run_dir)
    events = run_dir / "events.jsonl"
    rr, cur = 0, None

    def log(**kw):
        kw["t"] = round(time.time(), 2)
        with open(events, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")

    env, model, bad, tests = None, None, 0, 0
    while True:
        msg = inbox.get()
        if msg is None:
            break
        try:
            if maps:                                     # next pool map that has been finished at least once
                for _ in range(len(maps)):
                    mcfg, R, slug = maps[rr % len(maps)]
                    rr += 1
                    if (Path(R) / "best_run.json").exists():
                        break
                else:
                    continue
            else:
                mcfg, R, slug = cfg, run_dir, None
            R = Path(R)
            if maps and cur != slug and env is not None:          # next pool map: one test gym at a time
                env.close()
                env = None
            cur = slug
            best_file, line_file, line_meta = R / "best.json", R / "centerline.npy", R / "centerline.json"
            if env is None:
                try:                           # needs one extra gym slot besides the training cars
                    env = TMEnv(mcfg, R, shorten=False, role="champion test")
                except Exception as e:
                    say(f"Champion tests disabled: {e}. Lower 'envs' by one to enable them.")
                    busy.value = -1
                    continue
            if model is None:
                model = PPO.load(str(run_dir / "tester_base.zip"), device="cpu")   # layout only
            model.policy.load_state_dict(msg["sd"])
            best_try = None                    # best of N: argmax lap + seeded laps, keep the fastest
            n_try = test_tries(mcfg, R)
            for sd in [None] + [1000 + i for i in range(max(0, n_try - 1))]:
                inf = run_episode(env, model, sd)
                tr = {"info": inf, "seed": sd, "hist": list(env.hist), "acts": list(env.acts), "rec": env.g.export()}
                key = (0, inf["race_ms"]) if inf["finished"] else (1, -inf.get("cp", 0))
                if best_try is None or key < best_try[0]:
                    best_try = (key, tr)
            tr = best_try[1]
            info, env.hist, env.acts = tr["info"], tr["hist"], tr["acts"]
            fin, rec = bool(info["finished"]), tr["rec"]
            tests += 1
            try:                               # watch any test in TICK, also failed ones
                InputText.write(str(R / "last_test.txt"), rec.actions)
            except Exception:
                pass
            ms = int(rec.finish_ms) if fin else None
            best = read_json(best_file)
            champ = best.get("champion_ms")
            promoted = rolled = False
            if fin:
                best.setdefault("first_test_ms", ms)
            if fin and (champ is None or ms < champ):
                promoted, bad = True, 0
                model.save(R / "champion_tmp.zip")         # champion = the smoothed weights
                os.replace(R / "champion_tmp.zip", R / "champion.zip")
                best["champion_ms"] = champ = ms
                best["champion_seed"] = tr["seed"]
                save_npy(R / "champ_trace.npy", np.asarray(env.hist, dtype=np.float64))
                InputText.write(str(R / "champion.txt"), rec.actions)
            elif champ is not None and (not fin or ms > champ + cfg["rollback_margin_s"] * 1000):
                bad += 1
                if bad >= cfg["rollback_after"] and not maps:
                    if (R / "champion.zip").exists():
                        (R / "ROLLBACK").write_text("1")
                        rolled = True
                    bad = 0
            else:
                bad = 0
            ca = R / "champion_actions.json"
            if fin and (promoted or not ca.exists()):          # route for the mid-track starts
                write_json(ca, {"ms": ms, "acts": env.acts, "ticks": env.ticks})
            if fin and not (R / "champ_trace.npy").exists():
                save_npy(R / "champ_trace.npy", np.asarray(env.hist, dtype=np.float64))
            line_ms = read_json(line_meta).get("ms") or best.get("first_test_ms")   # old runs: line from the first test
            refresh = cfg["line_refresh_s"] and promoted and line_ms and ms <= line_ms - cfg["line_refresh_s"] * 1000
            if env.guide:
                refresh = False                                  # guide mode: the line is PB/best sections
            if fin and (not line_file.exists() or refresh):     # reference line follows the champion
                what = "rebuilt from the new champion" if line_file.exists() else "created"
                save_npy(line_file, make_line(np.asarray(env.hist)[:, :3]))
                env._load_line()
                for f in ("profile.json", "profile.npy"):
                    (R / f).unlink(missing_ok=True)
                write_json(line_meta, {"ms": ms, "t": time.time()})
                say(f"Reference line + ghost {what} ({ms / 1000:.3f} s lap). Expect a short dip while cars adapt.")
            if fin and (promoted or not (R / "profile.npy").exists()):   # ghost follows the champion only
                champion_ghost(env, R, ms, (lambda **kw: log(**kw, map=slug)) if slug else log)
            write_json(best_file, best)
            log(k="test", lap_n=msg["lap_n"], ms=ms, champ_ms=champ, promoted=promoted, rolled_back=rolled,
                tries=n_try, seed=tr["seed"], map=slug)
            say(f"Champion test {tests}{' on ' + slug if slug else ''} (best of {n_try}): "
                + (f"{ms / 1000:.3f} s" if fin else f"no finish ({info.get('end')})")
                + (" -> NEW CHAMPION" if promoted else "") + (" -> rolling back to champion" if rolled else ""))
        except Exception as e:
            say(f"Champion test failed: {str(e)[:120]}")
            if env is not None:
                env.close()
            env = None
        finally:
            if busy.value != -1:
                busy.value = 0
    if env is not None:
        env.close()


def _make_callback(cfg, run_dir, tracker=None, maps=None, notice=None):
    """maps (base run): [(cfg, folder, slug), ...]: per-map section trackers + stuck hints, round-robin tests."""
    by_slug = {m[2]: m for m in maps or []}
    from stable_baselines3.common.callbacks import BaseCallback

    events, best_file = run_dir / "events.jsonl", run_dir / "best.json"
    status_file, line_file = run_dir / "status.json", run_dir / "centerline.npy"
    line_meta = run_dir / "centerline.json"

    def log(**kw):
        kw["t"] = round(time.time(), 2)
        with open(events, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")

    class Champion(BaseCallback):
        def _on_training_start(self):
            self.t0, self.last_chk = time.time(), 0.0
            self.steps0 = self.model.num_timesteps
            self.laps = self.since = self.bad = self.tests = 0
            self.reason, self.last_v, self.tests_off = "step budget reached", 0.0, False
            self.ema_update(reset=True)
            self.rb, self.rb_steps, self.neg = {}, 0, 0
            self.stuckw = StuckWatch()
            self.trackers, self.stuckws = {}, {}
            self.ep = [[] for _ in range(cfg["envs"])]       # self-imitation: the lap each car is driving
            self.ramps = {}                                  # wall-hit ramp per run folder (per map in base runs)
            self.finished_once = (run_dir / "best_run.json").exists() or bool(read_json(best_file).get("champion_ms"))
            freeze_guard(run_dir, seconds=max(180.0, 2.0 * time_budget(cfg) + 60.0))
            self.ptm, self.last_prof, self.prof_steps = np.zeros(5), time.time(), self.model.num_timesteps
            log(k="start", cfg=cfg, **({"notice": notice} if notice else {}))
            self.status("running")
            say("Training started: %d cars, %s steering, CPU." % (cfg["envs"], cfg["steer_mode"]))
            if not self.tests_ready():
                say("Champion tests start after the first finished lap.")

        def status(self, state, reason=""):
            write_json(status_file, {"state": state, "reason": reason, "pid": os.getpid(),
                                     "heartbeat": time.time(), "started": self.t0})

        def _ret_scale(self):
            vn = self.model.get_vec_normalize_env()
            return 1.0 / float(np.sqrt(vn.ret_rms.var + vn.epsilon)) if vn is not None and vn.norm_reward else 1.0

        def _on_rollout_end(self):
            self.last_v = float(self.model.rollout_buffer.values.mean())
            self.ema_update()

        def ema_update(self, reset=False):
            """Smoothed copy of the weights. Champion tests drive with it: steadier than the latest update."""
            sd = self.model.policy.state_dict()
            if reset or getattr(self, "ema", None) is None:
                self.ema = {k: v.detach().clone() for k, v in sd.items()}
                return
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    self.ema[k].mul_(0.7).add_(v.detach(), alpha=0.3)
                else:
                    self.ema[k] = v.detach().clone()

        def _on_rollout_start(self):
            self.model.aux_rows = []                        # aux-head labels of this rollout, one row per step
            self.model.ret_scale = self._ret_scale()
            vn = self.model.get_vec_normalize_env()
            self.model.ret_clip = float(vn.clip_reward) if vn is not None else 10.0
            ds = cfg["decay_steps"]
            if ds:                                         # exploit more as training goes on
                f = 0.5 ** (self.model.num_timesteps / ds)
                self.model.lr_schedule = __import__('ppo_ext').ConstLR(max(3e-4 * f, 3e-5))
                self.model.ent_coef = max(0.005 * f, 0.0005)
            pol = self.model.policy
            if hasattr(pol, "steer_std"):                  # analog steering: exploration std decays with it
                import torch
                from ppo_ext import STEER_STD0
                cap = float(np.log(STEER_STD0 * max(0.5 ** (self.model.num_timesteps / ds) if ds else 1.0, 0.1)))
                with torch.no_grad():
                    pol.log_std.clamp_(max=cap, min=float(np.log(0.2 / 127.0)))
            nv = self.model.logger.name_to_value
            if "train/value_loss" in nv:
                steps = self.model.num_timesteps - self.steps0
                log(k="upd", u=int(self.model._n_updates), steps=int(self.model.num_timesteps),
                    v=self.last_v, vl=float(nv["train/value_loss"]),
                    ent=float(nv.get("train/entropy_loss", 0.0)), ev=float(nv.get("train/explained_variance", 0.0)),
                    speed=steps * cfg["ticks_per_step"] / 100.0 / max(time.time() - self.t0, 1e-6),
                    **{k2: round(float(nv[k1]), 5) for k1, k2 in (("train/aux_loss", "aux"), ("train/sil_loss", "sil"),
                       ("train/sil_active", "sil_act"), ("train/sil_steps", "sil_n")) if k1 in nv},
                    **({"sstd": round(pol.steer_std(), 3)} if hasattr(pol, "steer_std") else {}))
                self.save_all()
            if self.rb_steps:                                  # reward breakdown, per step, since the last update
                per = {k: v / self.rb_steps for k, v in self.rb.items()}
                log(k="rb", r={k: round(v, 5) for k, v in per.items()})
                running = sum(v for k, v in per.items() if k not in ("finish", "end", "checkpoint"))
                if running < 0:
                    self.neg += 1
                    if self.neg in (3, 20, 100):
                        worst = sorted(((v, k) for k, v in per.items() if k not in ("finish", "end", "checkpoint")
                                        and v < 0))[:3]
                        say(f"Warning: driving earns {running:+.4f} per step on average (before finish/checkpoint "
                            f"bonuses; at {cfg['ticks_per_step'] * 10} ms per step). Biggest costs: "
                            + ", ".join(f"{k} {v:+.4f}" for v, k in worst)
                            + ". Stopping can look attractive; lower penalties or raise Progress.")
                else:
                    self.neg = 0
                self.rb, self.rb_steps = {}, 0

        def _on_step(self):
            infos, dones = self.locals["infos"], self.locals["dones"]
            if self.model.aux_rows is not None:
                self.model.aux_rows.append([list(i.get("aux") or (0.0, 0.0, 0, 0)) for i in infos])
            if self.model.sil is not None and self.model.sil_weight > 0:
                vn = self.model.get_vec_normalize_env()
                raw = vn.get_original_reward() if vn is not None else self.locals["rewards"]
                obs, acts = self.model._last_obs, self.locals["actions"]
                for i, (info, done) in enumerate(zip(infos, dones)):
                    self.ep[i].append((obs[i].astype(np.float16), acts[i].astype(np.float32), float(raw[i])))
                    if done:
                        if info.get("finished") and not info.get("mid") and not info.get("pool") and len(self.ep[i]) > 5:
                            o, a, r = zip(*self.ep[i])
                            if self.model.sil.add(info.get("map"), info["race_ms"], np.stack(o), np.stack(a), np.array(r)):
                                say(f"Self-imitation: lap {info['race_ms'] / 1000:.3f} s kept"
                                    + (f" ({info['map']})" if info.get("map") else ""))
                        self.ep[i] = []
            for info, done in zip(infos, dones):
                if done:
                    fin, pool_lap = bool(info.get("finished")), bool(info.get("pool"))
                    log(k="lap", ms=int(info["race_ms"]) if fin else None, cp=int(info.get("cp", 0)),
                        end=info.get("end", ""), mid=1 if info.get("mid") else 0, x=info.get("x"), z=info.get("z"),
                        **({"ast": info["ast"]} if info.get("ast") else {}),
                        **({"map": info["map"]} if info.get("map") else {}), **({"pool": 1} if pool_lap else {}))
                    if not pool_lap:                     # pool cars of a map run: no effect on this map's stats
                        self.laps += 1
                        self.since += 1
                        slug = info.get("map")
                        if fin and not info.get("mid"):
                            self.finished_once = True
                        rd_ = Path(by_slug[slug][1]) if maps and slug in by_slug else run_dir
                        if rd_ not in self.ramps:
                            self.ramps[rd_] = HitRamp(rd_, cfg)
                        self.ramps[rd_].add(info)
                        if maps and slug in by_slug:     # base run: per-map trackers
                            if slug not in self.trackers:
                                mc, md, _ = by_slug[slug]
                                self.trackers[slug] = SectionTracker(md, mc, n_sections(md, info.get("n_seg") or 1))
                                self.stuckws[slug] = StuckWatch()
                            self.trackers[slug].add(info)
                            self.stuckws[slug].add(info)
                        else:
                            self.stuckw.add(info)
                            if tracker is not None:
                                tracker.add(info)
                    tm = info.get("tm")
                    if tm:
                        self.ptm += [tm["tick"], tm["obs"], tm["env"], 0.0, tm["n"]]
                    for kk, vv in (info.get("rb") or {}).items():
                        self.rb[kk] = self.rb.get(kk, 0.0) + vv
                    self.rb_steps += int(info.get("rsteps", 0))
            now = time.time()
            watchdog()
            if now - self.last_prof > float(os.environ.get("TMAI_PROF_S", 300)) and self.ptm[4]:
                wall, steps = now - self.last_prof, self.model.num_timesteps - self.prof_steps
                car = self.ptm.copy()                       # car-side times come from finished laps
                car[3] = max(0.0, wall * cfg["envs"] / max(steps, 1) * car[4] - car[:3].sum())   # idle: learning, lock-step
                rec = prof_summary(car, wall * car[4] / max(steps, 1), ticks=cfg["ticks_per_step"])
                rec["cars"] = cfg["envs"]
                log(**rec)
                say(prof_line(rec).replace("waiting for an action", "waiting (PPO learning + slowest car)"))
                self.ptm, self.last_prof, self.prof_steps = np.zeros(5), now, self.model.num_timesteps
            if now - self.last_chk > 2.0:
                self.last_chk = now
                if (run_dir / "STOP").exists():
                    self.reason = "stopped by user"
                    return False
                if cfg["time_limit_min"] and now - self.t0 > cfg["time_limit_min"] * 60:
                    self.reason = "time limit reached"
                    return False
                if not maps:
                    self.check_rollback()
                for hr in self.ramps.values():
                    hr.update(log)
                for tr_ in ([tracker] if tracker is not None else []) + list(self.trackers.values()):
                    try:
                        tr_.update(log)
                    except Exception as e:                      # never stop training over this
                        say(f"Section tracking error: {str(e)[:100]}")
                champ, pb = read_json(best_file).get("champion_ms"), cfg["pb_seconds"]
                if cfg["stop_at_pb"] and pb and champ and champ <= pb * 1000 and not maps:
                    self.reason = "reached your PB"
                    return False
                self.status("running")
            if self.since >= cfg["test_every_laps"] and not self.tests_off and self.tests_ready():
                self.champion_test()
            return True

        def tests_ready(self):
            """No champion tests (and so no rollbacks) before the first full lap has been finished in this run."""
            if maps:
                return any((Path(m[1]) / "best_run.json").exists() for m in maps)
            return self.finished_once

        def tester_start(self):
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self.inbox, self.busy = ctx.Queue(), ctx.Value("i", 0)
            self.tester = ctx.Process(target=ppo_test_worker, args=(cfg, str(run_dir), self.inbox, self.busy,
                                                                    maps), daemon=True)
            self.tester.start()

        def champion_test(self):
            """Hand the smoothed (EMA) weights to the tester process; training keeps driving meanwhile."""
            if getattr(self, "tester", None) is None:
                self.model.save(run_dir / "tester_base")         # network layout for the tester (weights replaced)
                self.tester_start()
            elif not self.tester.is_alive():
                say("Champion tester stopped unexpectedly, restarting it")
                self.tester_start()
            if self.busy.value == -1:                            # tester could not open a gym
                self.tests_off = True
                return
            if self.busy.value:
                limit = max(cfg["test_tries"], cfg["test_tries_close"]) * (1.3 * time_budget(cfg) + 20.0) + 180.0
                if time.time() - self.test_t0 > limit:          # a test that long is a hung tester
                    say(f"Champion test running for {limit / 60:.0f} min: restarting the tester")
                    try:
                        self.tester.terminate()
                    except Exception:
                        pass
                    self.tester_start()
                return
            if not maps:
                self.check_rollback()                            # never send a test before a pending rollback
            self.since = 0
            sd = self.ema if getattr(self, "ema", None) is not None else self.model.policy.state_dict()
            self.busy.value = 1
            self.test_t0 = time.time()
            self.inbox.put({"sd": {k: v.detach().clone() for k, v in sd.items()}, "lap_n": self.laps})

        def check_rollback(self):
            flag = run_dir / "ROLLBACK"
            if not flag.exists():
                return
            try:
                self.model.set_parameters(str(run_dir / "champion.zip"), exact_match=True, device="cpu")
                self.ema_update(reset=True)
            except Exception as e:
                say(f"Rollback failed ({str(e)[:60]})")
            flag.unlink(missing_ok=True)

        def save_all(self):
            atomic_save(run_dir / "latest.zip", lambda p: self.model.save(str(p)))
            if self.model.sil is not None:
                try:
                    self.model.sil.save(str(run_dir / "sil.npz"))      # the run's own laps; never exported in packs
                except OSError:
                    pass
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(str(run_dir / "vecnorm.pkl"))

        def _on_training_end(self):
            self.save_all()
            log(k="stop", reason=self.reason)
            self.status("stopped", self.reason)
            say("Stopped: " + self.reason)
            if getattr(self, "tester", None) is not None:
                try:
                    self.inbox.put(None)
                    self.tester.join(timeout=60)
                except Exception:
                    pass

    return Champion()


def prepare_guide(cfg, run_dir):
    """Trace your PB once (replaying its input file in the simulation) into pb_trace.npy, then build the target line."""
    f = cfg.get("pb_file", "").strip().strip('"')
    if not f:
        return
    src, out = Path(f), run_dir / "pb_trace.npy"
    if not src.exists():
        say(f"PB file not found: {src} (target line off)")
        return
    meta = read_json(run_dir / "pb_trace.json")
    if not (out.exists() and meta.get("file") == str(src) and meta.get("mtime") == src.stat().st_mtime):
        env = None
        try:
            env = TMEnv(dict(cfg, pb_file=""), None, shorten=False, role="pb trace")
            hist, ms = trace_record(env.g, InputText.read(str(src)), ticks=cfg["ticks_per_step"])
        except Exception as e:
            say(f"Could not trace the PB file ({str(e)[:100]}); target line off")
            return
        finally:
            if env is not None:
                env.close()
        if ms is None:
            say("The PB file did not finish the map in the simulation (wrong map or file?); target line off")
            return
        save_npy(out, hist)
        write_json(run_dir / "pb_trace.json", {"file": str(src), "mtime": src.stat().st_mtime, "ms": ms})
        say(f"Traced your PB: {ms / 1000:.3f} s, {int(hist[-1, 4])} checkpoints")
    r = build_guide(run_dir)
    if r:
        say(f"Target line: sections {' '.join(r[1])}, pace {r[0] / 1000:.3f} s. The AI is pulled toward it only "
            "where it is slower, and any section it drives faster replaces yours.")


def base_info(name, maps=None):
    """runs/_base/<name>/base.json: the base model's unique id (created once) and its pool maps."""
    d = cfgmod.base_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    info = read_json(d / "base.json")
    if not info.get("base_id"):
        import uuid
        info = {"base_id": uuid.uuid4().hex[:12], "created": time.time(), "name": name}
    if maps is not None:
        info["maps"] = maps
    write_json(d / "base.json", info)
    return info


def start_from_base(cfg, run_dir):
    """Map run with base_from set and no model of its own yet: copy the base model + reward scaling (the layout
    is converted by migrate() if it changed). Returns the base record for the run's base.json, or None."""
    name = cfg["base_from"]
    if not name or (run_dir / "latest.zip").exists():
        return None
    src = cfgmod.base_dir(name)
    if not (src / "latest.zip").exists():
        say(f"Base '{name}' has no model yet (runs/_base/{name}/latest.zip); starting from scratch")
        return None
    import shutil
    for f in ("latest.zip", "vecnorm.pkl", "levels.json"):
        if (src / f).exists():
            shutil.copyfile(src / f, run_dir / f)
    rec = {"base_name": name, "base_id": base_info(name).get("base_id"), "t": time.time()}
    write_json(run_dir / "base.json", rec)                 # also switches on map-neutral progress for this run
    return rec


def setup_cars(cfg, run_dir, pnorm):
    """Which map each car drives. Base run: cars spread over the pool (rotating every pool_rotate_laps laps when
    there are more maps than cars), each map with its own folder. Map run with a pool: pool_cars cars keep driving
    pool maps (no forgetting). Returns (specs for CarPool or None, maps for the tester / trackers or None)."""
    n, pool = cfg["envs"], cfg["map_pool"]
    if cfg["run_type"] == "base":
        if not pool:
            raise SystemExit("Base run: add maps to the map pool first (Settings).")
        maps = []
        for e in pool:
            c = cfgmod.map_cfg(cfg, e, pnorm)
            d = run_dir / "maps" / c["_slug"]
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / "map.json", e)
            prepare_guide(c, d)
            maps.append((c, str(d), c["_slug"]))
        base_info(cfg["base_name"], [m[2] for m in maps])
        pairs = [(m[0], m[1]) for m in maps]
        if len(maps) > n:
            specs = [(cfg, str(run_dir), {"maps": pairs, "start": i, "stride": n, "every": cfg["pool_rotate_laps"]})
                     for i in range(n)]
        else:
            specs = [(maps[i % len(maps)][0], maps[i % len(maps)][1], None) for i in range(n)]
        say(f"Base run '{cfg['base_name']}': {len(maps)} maps, {n} cars"
            + (f", each car changes map every {cfg['pool_rotate_laps']} laps" if len(maps) > n else ""))
        return specs, maps
    k = min(cfg["pool_cars"], n - 1) if pool else 0
    if k <= 0:
        return None, None
    main = dict(cfg, _pnorm=pnorm)
    pmaps = []
    for e in pool:
        c = cfgmod.map_cfg(cfg, e, pnorm, pool=True)
        d = run_dir / "pool" / c["_slug"]
        d.mkdir(parents=True, exist_ok=True)
        if cfg["base_from"]:                                  # start from the base's line / ghost for that map
            src = cfgmod.base_dir(cfg["base_from"]) / "maps" / c["_slug"]
            for f in ("centerline.npy", "centerline.json", "profile.npy", "profile.json"):
                if (src / f).exists() and not (d / f).exists():
                    import shutil
                    shutil.copyfile(src / f, d / f)
        prepare_guide(c, d)
        pmaps.append((c, str(d)))
    specs = [(main, str(run_dir), None)] * (n - k) + [
        (main, str(run_dir), {"maps": pmaps, "start": j, "stride": k, "every": cfg["pool_rotate_laps"]}) for j in range(k)]
    say(f"{k} of {n} cars keep driving the {len(pmaps)} pool maps (their laps don't count for this map)")
    return specs, None


def train(cfg):
    import torch
    from stable_baselines3 import PPO  # noqa: F401
    from ppo_ext import TMPPO, SILBuffer
    from stable_baselines3.common.vec_env import VecNormalize

    run_dir = cfgmod.run_dir(cfg)
    t_start = time.time()
    for w in cfgmod.RUN_WARNINGS:
        say("Warning: " + w)
    (run_dir / "STOP").unlink(missing_ok=True)
    (run_dir / "ROLLBACK").unlink(missing_ok=True)
    (run_dir / "starts_logged.json").unlink(missing_ok=True)   # mid-start log: once per rebuild
    torch.set_num_threads(2)                                   # tiny MLP: more threads only slow it down
    raise_priority(tick_too=True)
    base = cfg["run_type"] == "base"
    if not base:
        freeze_preset(cfg, run_dir)
    from_base = None if base else start_from_base(cfg, run_dir)
    pnorm = float(cfg["progress_norm_m"]) if (base or (run_dir / "base.json").exists()) else 0.0
    specs, maps = setup_cars(cfg, run_dir, pnorm)
    if not base:
        cfg = dict(cfg, _pnorm=pnorm)
        prepare_guide(cfg, run_dir)
    venv = make_pool_vecenv(cfg, str(run_dir), specs)       # a hung car is replaced, not the whole training frozen
    tracker = None if base else SectionTracker(run_dir, cfg, n_sections(run_dir, venv.n_seg))
    set_gamma(cfg)
    vn_file = run_dir / "vecnorm.pkl"
    env = None
    if cfg["resume"] and vn_file.exists():
        try:
            env = VecNormalize.load(str(vn_file), venv)
            env.training, env.gamma = True, GAMMA[0]
            env.action_space = venv.action_space          # the pickle holds the old action space (steering kind)
            say("Loaded reward scaling (vecnorm.pkl)")
        except Exception:                       # input size changed: keep just the reward scale
            try:
                import pickle
                with open(vn_file, "rb") as fh:
                    old = pickle.load(fh)
                env = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA[0])
                env.ret_rms = old.ret_rms
                say("Kept the reward scaling from vecnorm.pkl (input layout changed)")
            except Exception as e:
                env = None
                say(f"Could not load vecnorm.pkl ({str(e)[:60]}), relearning reward scale")
    if env is None:
        env = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA[0])
    else:
        old_ticks = saved_ticks(run_dir)
        if old_ticks and old_ticks != int(cfg["ticks_per_step"]):
            # the reward scale (running return variance) was learned at another decision interval: per-step rewards
            # and the number of steps per second both changed, so start the scale fresh (it relearns in minutes)
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            env.ret_rms = RunningMeanStd(shape=())
            env.returns = np.zeros(env.num_envs)
            say(f"Decision interval changed ({old_ticks * 10} -> {cfg['ticks_per_step'] * 10} ms): "
                "reward scaling (vecnorm.pkl) reset, relearning it.")
    migrate(run_dir, cfg, env)
    analog = isinstance(env.action_space, spaces.Box)

    model, latest, champ_zip = None, run_dir / "latest.zip", run_dir / "champion.zip"
    notice, errs = None, []
    if not (cfg["resume"] or from_base):
        if latest.exists() or champ_zip.exists():
            notice = "STARTING FROM SCRATCH because 'Resume from latest checkpoint' is off (the saved model is ignored)"
    else:
        for src in (latest, champ_zip):              # latest.zip, else the champion (missing / corrupt latest.zip)
            if not src.exists():
                continue
            try:
                model = TMPPO.load(str(src), env=env, device="cpu")
                model.target_kl = 0.02
                model.gamma = model.rollout_buffer.gamma = GAMMA[0]
                say(f"Planning horizon {cfg['horizon_s']:.0f} s -> gamma {GAMMA[0]:.4f} at {cfg['ticks_per_step'] * 10} ms")
                if src is latest:
                    say("Resumed from latest.zip")
                else:
                    why = "latest.zip is missing" if not latest.exists() else f"latest.zip could not be loaded ({errs[-1]})"
                    notice = f"Resumed from champion.zip because {why}"
                break
            except Exception as e:
                errs.append(str(e)[:100])
        if model is None:
            unconv = [p.name for p in run_dir.glob("*.unconverted.zip") if p.stat().st_mtime >= t_start - 5]
            if unconv:
                notice = ("STARTING FROM SCRATCH because the saved model could not be converted to the new input "
                          f"layout (moved aside as {', '.join(unconv)})")
            elif errs:
                notice = f"STARTING FROM SCRATCH because the saved model could not be loaded ({errs[-1]})"
    if notice:
        say("!" * 12 + " " + notice + " " + "!" * 12)
    if model is None:
        model = make_model(env, STEER_LEVELS[cfg["steer_mode"]], fresh=True)
    model.sil_weight, model.aux_weight = float(cfg["sil_weight"]), float(cfg["aux_weight"])
    model.ticks_per_step = cfg["ticks_per_step"]
    if model.sil_weight > 0:
        model.sil = SILBuffer(analog, STEER_LEVELS[cfg["steer_mode"]])
        if model.sil.load(str(run_dir / "sil.npz"), int(env.observation_space.shape[0])):
            say(f"Self-imitation memory: {sum(len(v) for v in model.sil.laps.values())} laps loaded")
    nm = n_modes_of(model.action_space)
    say(f"PPO: {'analog steering (-127..127), ' if analog else '21 steering levels, '}"
        f"{nm} throttle modes{' (gas+brake on)' if nm == 4 else ''}, "
        f"self-imitation {model.sil_weight:g}, aux heads {model.aux_weight:g}, critic with {N_PRIV} extra inputs")
    if from_base:
        write_json(run_dir / "base.json", {**from_base, "base_steps": int(model.num_timesteps)})
        say(f"Started from base '{from_base['base_name']}' (id {from_base['base_id']}, "
            f"{model.num_timesteps:,} steps of pool driving)")
    focus_entropy(model)
    cb = _make_callback(cfg, run_dir, tracker, maps, notice)
    try:
        model.learn(total_timesteps=10 ** 12, callback=cb, reset_num_timesteps=False)
    except KeyboardInterrupt:
        write_json(run_dir / "status.json", {"state": "stopped", "reason": "stopped by user", "heartbeat": time.time()})
    except Exception as e:
        write_json(run_dir / "status.json", {"state": "stopped", "reason": f"crashed: {str(e)[:120]}",
                                             "heartbeat": time.time()})
        raise
    finally:
        env.close()


def replay_champion(cfg, run_dir, env):
    """Play champion = the champion test's own lap, replayed from its recorded actions (champion_actions.json).
    Re-driving it with the network does NOT reproduce the time: its inputs (ghost, target line, section rates)
    have changed since the test and IQN samples its quantiles at random. Returns res or None (no recording)."""
    run_dir = Path(run_dir)
    ca, champ = read_json(run_dir / "champion_actions.json"), read_json(run_dir / "best.json").get("champion_ms")
    acts = ca.get("acts") or []
    if not acts:
        return None
    ticks = int(ca.get("ticks") or cfg["ticks_per_step"])
    if ca.get("ms") and champ and int(ca["ms"]) != int(champ):
        say(f"Warning: champion_actions.json is from a {int(ca['ms']) / 1000:.3f} s lap, the champion is "
            f"{champ / 1000:.3f} s")
    env.g.reset(seed=0)
    t = None
    for steer, mode in acts:
        t = env.g.step(ticks, steer=int(steer), accelerate=int(mode) in GAS, brake=int(mode) in BRAKE)
        if t.finished:
            break
    res = {"t": time.time(), "finished": bool(t is not None and t.finished), "end": "finished", "ms": None,
           "file": None, "champion_ms": champ, "source": "champion_actions.json"}
    if not res["finished"]:
        res["end"] = "no finish"
        say("Warning: replaying the champion's recorded actions did not finish (ticks per step changed?)")
        return res
    rec = env.g.export()
    InputText.write(str(run_dir / "play.txt"), rec.actions)
    res.update(ms=int(rec.finish_ms), file=str(run_dir / "play.txt"))
    say(f"Champion play finished in {rec.finish_ms / 1000:.3f} s -> {run_dir / 'play.txt'} (load it in TICK to watch it)")
    check_play(res["ms"], champ)
    return res


def check_play(ms, champ):
    if ms is not None and champ and abs(ms - champ) > 10:
        say(f"Warning: play ({ms / 1000:.3f} s) differs from the champion ({champ / 1000:.3f} s) by more than 0.01 s")


def drive(cfg, model_path=None):
    from stable_baselines3 import PPO

    run_dir = cfgmod.run_dir(cfg)
    if cfg["run_type"] == "base":                 # base run: the champion of the first pool map
        if not cfg["map_pool"]:
            raise SystemExit("Base run without pool maps.")
        cfg = cfgmod.map_cfg(cfg, cfg["map_pool"][0], cfg["progress_norm_m"])
        run_dir = run_dir / "maps" / cfg["_slug"]
    env = TMEnv(cfg, run_dir, shorten=False, role="play")
    if not model_path:
        res = replay_champion(cfg, run_dir, env)
        if res is not None:
            write_json(run_dir / "play_result.json", res)
            env.close()
            return
        say("No recorded champion lap (champion_actions.json): driving the champion network instead.")
    model = PPO.load(model_path or str(run_dir / "champion.zip"), device="cpu")
    say("Driving the champion (the lap is simulated in the background, the game window will not show it) ...")
    info = run_episode(env, model, read_json(run_dir / "best.json").get("champion_seed"))
    res = {"t": time.time(), "finished": bool(info["finished"]), "end": info["end"], "ms": None, "file": None}
    if info["finished"]:
        rec = env.g.export()
        InputText.write(str(run_dir / "play.txt"), rec.actions)
        res.update(ms=int(rec.finish_ms), file=str(run_dir / "play.txt"))
        say(f"Champion play finished in {rec.finish_ms / 1000:.3f} s -> {run_dir / 'play.txt'} (load it in TICK to watch it)")
    else:
        say(f"[play 1] no finish ({info['end']})")
    write_json(run_dir / "play_result.json", res)
    env.close()


FIELDS = {
    "runtime": ["wall_hit_this_step", "forward_motion_alignment_scale", "roof_contact_this_step", "wetness",
                "surface_grip_scale", "soft_surface_wheel_count", "is_any_wheel_on_ice_or_snow", "accel_coef",
                "control_coef", "gravity_coef", "adherence_coef"],
    "effects": ["reactor_direction", "turbo", "turbo_remaining_ms", "reactor", "engine", "no_brakes", "no_steering",
                "no_grip", "fragile", "slow_motion", "cruise_speed"],
    "vehicle": ["in_water"],
    "wheels": ["icing_percent", "angular_velocity"],
}


def transfer(cfg, map_path, pb_file=""):
    """Transfer score: drive the base model (no learning) on a map that is NOT in its pool, best of 3 (most likely
    actions + 2 seeds). Reports the time, or where it fails (checkpoint, position, share of the lap)."""
    from stable_baselines3 import PPO
    name = cfg["base_name"]
    bd = cfgmod.base_dir(name)
    if not (bd / "latest.zip").exists():
        raise SystemExit(f"Base '{name}' has no model yet ({bd / 'latest.zip'}).")
    entry = {"map": map_path, "pb_file": pb_file, "pb_seconds": 0}
    c = cfgmod.map_cfg(cfg, entry, cfg["progress_norm_m"])
    slug = c["_slug"]
    if slug in [cfgmod.map_slug(e["map"]) for e in cfg["map_pool"]]:
        say(f"Note: {map_path} is in the base's pool, so this is not a transfer test.")
    d = bd / "transfer" / slug
    d.mkdir(parents=True, exist_ok=True)
    prepare_guide(c, d)
    model = PPO.load(str(bd / "latest.zip"), device="cpu")
    env = TMEnv(c, d, shorten=False, role="transfer test")
    if model.observation_space.shape != env.observation_space.shape:
        env.close()
        raise SystemExit("The base model uses an older input layout: train the base once with this version first.")
    say(f"Transfer test: base '{name}' on {map_path} (not trained on it), best of 3 ...")
    tries = []
    for sd in (None, 1000, 1001):
        inf = run_episode(env, model, sd)
        length = env.dense[2][-1] if env.dense is not None else env.lap_len
        tries.append({"seed": sd, "finished": bool(inf["finished"]), "ms": int(inf["race_ms"]) if inf["finished"] else None,
                      "end": inf.get("end", ""), "cp": int(inf.get("cp", 0)), "x": inf.get("x"), "z": inf.get("z"),
                      "progress": round(min(float(getattr(env, "_s", 0.0)) / max(length, 1.0), 1.0), 3)})
        t = tries[-1]
        say(f"  try {len(tries)}: " + (f"finished in {t['ms'] / 1000:.3f} s" if t["finished"] else
            f"{t['end']} after checkpoint {t['cp']} at x={t['x']} z={t['z']} ({t['progress']:.0%} of the lap)"))
    env.close()
    best = min(tries, key=lambda t: (0, t["ms"]) if t["finished"] else (1, -t["progress"]))
    rec = {"t": time.time(), "map": map_path, "slug": slug, "base_id": base_info(name).get("base_id"),
           "best": best, "tries": tries}
    write_json(d / "transfer.json", rec)
    allr = read_json(bd / "transfers.json")
    allr[slug] = rec
    write_json(bd / "transfers.json", allr)
    with open(bd / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"k": "transfer", "t": round(time.time(), 2), "map": slug, "best": best}) + "\n")
    say("Transfer score: " + (f"{best['ms'] / 1000:.3f} s" if best["finished"] else
        f"no finish - best try got {best['progress']:.0%} of the lap ({best['end']} after checkpoint {best['cp']})"))
    return rec


def missing_fields(env):
    t = env._last_t
    out = []
    for grp, names in FIELDS.items():
        obj = getattr(t, grp, None)
        if obj is None:
            out.append(f"{grp} (whole group)")
            continue
        out += [f"{grp}.{n}" for n in names if not hasattr(obj, n)]
    raw = getattr(getattr(t, "wheels", None), "icing_percent", None)
    print("raw wheels.icing_percent:", raw, "| raw runtime.is_any_wheel_on_ice_or_snow:",
          getattr(getattr(t, "runtime", None), "is_any_wheel_on_ice_or_snow", None))
    return out


def selftest(cfg):
    from collections import Counter
    env = TMEnv(cfg, cfgmod.run_dir(cfg), shorten=False)
    o, _ = env.reset()
    seen = {"wheel material": Counter(), "wheel gameplay": Counter(), "ray material": Counter()}
    ice = {}
    for _ in range(40):
        o, r, term, trunc, info = env.step(np.array([0.0, 0]) if env.analog else np.array([10, 0]))   # straight, full gas
        t = env._last_t
        for m in getattr(t.wheels, "material_id", []):
            seen["wheel material"][int(m)] += 1
        hit = env._lid[0]
        for i in env.SURF_RAYS:
            if i < len(hit) and hit[i]:
                seen["ray material"][int(env._lid_mat[i])] += 1
        rt = getattr(t, "runtime", None)
        if rt is not None:
            for k in env.WHEELS4:
                seen["wheel gameplay"][int(_f(rt, f"{k}_surface_gameplay_id"))] += 1
            for name in [f"{k}_icing_{x}" for k in env.WHEELS4 for x in ("state", "blend")] + [
                    "front_wheels_icing_grip_scale", "rear_wheels_icing_grip_scale", "is_any_wheel_on_ice_or_snow"]:
                ice[name] = max(ice.get(name, -1e9), _f(rt, name, float("nan")))
        if term or trunc:
            break
    print("steering:", "analog -127..127 (PPO)" if env.analog else env.STEER.tolist())
    print("route points:", len(env.route) - 1, "| route length: %.0f m" % env.lap_len,
          "| reference line:", "yes" if env.dense is not None else "not yet")
    print("local velocity m/s (expect z > 0, x ~ 0):", np.round(o[1:4] * 100, 2))
    print("world up in car frame (expect ~ 0, 1, 0):", np.round(o[7:10], 2))
    print("first lookahead (angle/pi, elevation):", np.round(o[env.LOOK_OFF:env.LOOK_OFF + 2], 2))
    if env.extra:
        print("missing telemetry fields (these always read their default):", missing_fields(env) or "none")
        print("extra observations:", {n: round(float(x), 3) for n, x in zip(EXTRA_NAMES, o[N_BASE0:env.N_BASE])})
    _vocab_load()
    for kind, c in seen.items():
        voc = _VOCAB["gameplay" if "gameplay" in kind else "material"]
        print(f"{kind} ids seen (id: count -> input slot):",
              ", ".join(f"{k}: {n} -> {voc.get(k, N_SURF) + 1 if voc.get(k, N_SURF) < N_SURF else 'other'}"
                        for k, n in sorted(c.items())) or "none")
    if ice:
        print("ice raw values (max over the test drive; on a summer map icing_state/blend and is_any_wheel_on_ice"
              "_or_snow must be 0, grip scales 1):", {k: round(v, 3) for k, v in ice.items()})
    print("lidar min/max m:", np.round((o[env.N_BASE:] * env.ray_range).min(), 1),
          np.round((o[env.N_BASE:] * env.ray_range).max(), 1), "| obs size:", o.shape[0])
    env.close()


# --- diagnose: short A/B PPO trials on the real game ------------------------------------------------------
DIAG_TRIALS = [("a", "21 levels, 3 modes", dict(analog_steer=False, gas_brake=False)),
               ("b", "analog, 3 modes", dict(analog_steer=True, gas_brake=False)),
               ("c", "21 levels, gas+brake", dict(analog_steer=False, gas_brake=True)),
               ("d", "analog, gas+brake", dict(analog_steer=True, gas_brake=True))]
DIAG_FAIL = ("stuck", "far off line", "off-zone")


def diag_metrics(run_dir, n_modes):
    """Summary of one trial from its events.jsonl."""
    laps, run = [], []
    try:
        with open(Path(run_dir) / "events.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("k") == "lap" and not e.get("pool"):
                    laps.append(e)
                elif e.get("k") == "rb":
                    run.append(sum(v for k, v in e["r"].items() if k not in ("finish", "end", "checkpoint")))
    except OSError:
        pass
    ast = [e["ast"] for e in laps if e.get("ast")]
    stuck = [e for e in laps if e.get("end") in DIAG_FAIL]
    spots = {}
    for e in stuck:
        if e.get("x") is not None:
            k = (int(e.get("cp", 0)), int(round(e["x"] / 30.0)) * 30, int(round(e["z"] / 30.0)) * 30)
            spots[k] = spots.get(k, 0) + 1
    top = max(spots.items(), key=lambda kv: kv[1]) if spots else None
    prog = [a["s"] for a in ast]
    late = prog[len(prog) // 2:] or prog
    fin = [e["ms"] for e in laps if e.get("ms") and not e.get("mid")]
    modes = np.mean([a["m"] + [0.0] * (n_modes - len(a["m"])) for a in ast], axis=0) if ast else np.zeros(n_modes)
    names = ["gas", "coast", "brake", "gas+brake"][:n_modes]
    return {
        "laps": len(laps), "driving_earns_per_step": round(float(np.mean(run)), 4) if run else None,
        "driving_earns_last": round(float(np.mean(run[-5:])), 4) if run else None,
        "stuck_share": round(len(stuck) / len(laps), 3) if laps else None,
        "stuck_spot": ({"cp": top[0][0], "x": top[0][1], "z": top[0][2], "share_of_stuck": round(top[1] / len(stuck), 3)}
                       if top else None),
        "mean_speed_ms": round(float(np.mean([a["v"] for a in ast])), 1) if ast else None,
        "furthest_m": round(float(max(prog)), 1) if prog else None,
        "progress_p90_late_m": round(float(np.percentile(late, 90)), 1) if late else None,
        "throttle_share": {n: round(float(x), 3) for n, x in zip(names, modes)},
        "steer_mean_abs": round(float(np.mean([a["sa"] for a in ast])) * 127, 1) if ast else None,
        "steer_full_lock_share": round(float(np.mean([a["fl"] for a in ast])), 3) if ast else None,
        "best_lap_s": round(min(fin) / 1000.0, 3) if fin else None,
        "finished_laps": len(fin)}


def _diag_score(m):
    return (1 if m.get("best_lap_s") else 0, -(m.get("best_lap_s") or 0.0), m.get("progress_p90_late_m") or 0.0)


def _diag_worse(w, wo):
    """True if the trial WITH a feature (w) learned clearly worse than the one without (wo)."""
    pw, po = w.get("progress_p90_late_m") or 0.0, wo.get("progress_p90_late_m") or 0.0
    if wo.get("best_lap_s") and not w.get("best_lap_s"):
        return True
    if w.get("best_lap_s") and wo.get("best_lap_s"):
        return w["best_lap_s"] > wo["best_lap_s"] + 0.5
    return pw < 0.75 * po or (w.get("stuck_share") or 0) - (wo.get("stuck_share") or 0) > 0.2


def diag_verdict(res):
    t, out = {k: v["metrics"] for k, v in res.items()}, []
    if all(k in t for k in "abcd"):
        for name, pairs in (("analog steering", (("b", "a"), ("d", "c"))), ("gas+brake", (("c", "a"), ("d", "b")))):
            bad = [_diag_worse(t[w], t[wo]) for w, wo in pairs]
            out.append(f"{name}: " + ("BREAKS learning here (worse in both comparisons)" if all(bad) else
                                      "probably hurts (worse in one comparison)" if any(bad) else "no harm seen"))
    if "e" in t and res["e"].get("from") in t:
        out.append("novelty rewards: " + ("hurt learning" if _diag_worse(t["e"], t[res["e"]["from"]]) else "no harm seen"))
    return out


def diagnose(cfg, map_path, pb="", minutes=4.0):
    """Short PPO trials from the same fresh start (runs diag_a .. diag_e), report -> runs/_diag/diagnose.json."""
    import subprocess
    root = cfgmod.ROOT
    ddir = root / "runs" / "_diag"
    ddir.mkdir(parents=True, exist_ok=True)
    base = dict(cfg)
    if map_path and map_path != cfg["map"]:
        base.update(map=map_path, pb_file=pb or "", pb_seconds=0.0)
    elif pb:
        base["pb_file"] = pb
    base.update(algo="ppo", run_type="map", time_limit_min=0, stop_at_pb=False, resume=True,
                w_novel=0.0, w_known=0.0, w_focus=0.0, reward_v=4)
    secs = float(os.environ.get("TMAI_DIAG_S") or minutes * 60.0)
    say(f"Diagnose: {len(DIAG_TRIALS) + 1} PPO trials of {secs / 60:.1f} min on {base['map']} "
        f"({base['envs']} cars, {base['ticks_per_step'] * 10} ms per step) ...")
    res = {}

    def trial(key, label, over):
        name = f"diag_{key}"
        rd = root / "runs" / name
        if rd.exists():
            import shutil
            shutil.rmtree(rd, ignore_errors=True)               # same fresh start for every trial
        c = cfgmod.coerce(dict(base, run=name, **over))
        cpath = ddir / f"config_{key}.json"
        cpath.write_text(json.dumps(c, indent=1), encoding="utf-8")
        say(f"Diagnose trial {key} ({label}) ...")
        env = dict(os.environ, TMAI_CONFIG=str(cpath))
        with open(ddir / f"{name}.log", "w", encoding="utf-8") as logf:
            p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "train", "--new", "--run", name],
                                 cwd=str(root), env=env, stdout=logf, stderr=subprocess.STDOUT)
            t0 = None
            while p.poll() is None:
                time.sleep(1.0)
                if t0 is None and (rd / "events.jsonl").exists():
                    t0 = time.time()                            # count from the training start, not the start-up
                if t0 is not None and time.time() - t0 > secs:
                    (rd / "STOP").write_text("diagnose")
                    try:
                        p.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break
        m = diag_metrics(rd, 4 if c["gas_brake"] else 3)
        res[key] = {"label": label, "settings": over, "metrics": m}
        say(f"  {key} {label}: earns/step {m['driving_earns_per_step']}, stuck {m['stuck_share']}, "
            f"speed {m['mean_speed_ms']} m/s, furthest {m['furthest_m']} m, best {m['best_lap_s']}")

    for key, label, over in DIAG_TRIALS:
        trial(key, label, over)
    best = max("abcd", key=lambda k: _diag_score(res[k]["metrics"]))
    nov = {k: cfgmod.WEIGHTS[k][0] for k in ("w_novel", "w_known", "w_focus")}
    trial("e", f"{res[best]['label']} + novelty defaults", dict(res[best]["settings"], **nov))
    res["e"]["from"] = best
    verdict = diag_verdict(res)
    report = {"t": time.time(), "map": base["map"], "pb_file": base.get("pb_file"), "minutes": round(secs / 60, 2),
              "ticks_per_step": base["ticks_per_step"], "cars": base["envs"], "trials": res, "best": best,
              "verdict": verdict}
    write_json(ddir / "diagnose.json", report)
    print("\n" + "=" * 100)
    hdr = f"{'trial':26s}{'earns/step':>11s}{'stuck':>7s}{'speed':>7s}{'furthest':>9s}{'|steer|':>8s}{'lock':>6s}  throttle share{'':22s}best"
    print(hdr)
    for k, r in res.items():
        m = r["metrics"]
        thr = " ".join(f"{n[:4]} {v:.0%}" for n, v in m["throttle_share"].items())
        print(f"{k + ' ' + r['label']:26.26s}{str(m['driving_earns_per_step']):>11s}{str(m['stuck_share']):>7s}"
              f"{str(m['mean_speed_ms']):>7s}{str(m['furthest_m']):>9s}{str(m['steer_mean_abs']):>8s}"
              f"{str(m['steer_full_lock_share']):>6s}  {thr:36s}{m['best_lap_s']}")
        if m.get("stuck_spot"):
            sp = m["stuck_spot"]
            print(f"{'':26s}most stuck laps: after CP {sp['cp']} near x {sp['x']}, z {sp['z']} "
                  f"({sp['share_of_stuck']:.0%} of the stuck laps)")
    print("Verdict: " + " | ".join(verdict) + f" | best of a-d: {best}")
    print(f"Report: {ddir / 'diagnose.json'}")
    say("Diagnose finished: " + " | ".join(verdict))


def pack_cli(argv):
    """publish / verify / benchmark / export / import / hub: brain packs and the public hub (hub.py, packs.py)."""
    import hub
    import packs
    cfg = cfgmod.load()
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"tm_ai.py {cmd}")
    if cmd == "hub":
        return hub.main(cfg, rest)
    ap.add_argument("target", help="run name, base:NAME, pack zip or pack id")
    ap.add_argument("--map", action="append", help="UID=PATH of a local map (verify / benchmark)")
    ap.add_argument("--wr", type=float, default=None, help="publish/export: world record time in seconds")
    ap.add_argument("--note", default="")
    ap.add_argument("--as", dest="name", default=None, help="import: run (or base) name")
    ap.add_argument("--base", action="store_true", help="import as a base model")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="verify: also run the benchmark maps")
    a = ap.parse_args(rest)
    if cmd == "publish":
        hub.cmd_publish(cfg, a.target, a.wr, a.note, open_browser=not a.no_browser)
    elif cmd == "export":
        path, man = packs.export_pack(cfg, a.target, cfgmod.ROOT / "runs" / "_hub" / "out",
                                      author=cfg["github_user"] or "anonymous", wr=a.wr, note=a.note)
        say(f"Exported {path}")
    elif cmd == "import":
        name = a.name or Path(a.target).name.split(".")[0]
        dest = cfgmod.base_dir(name) if a.base else cfgmod.ROOT / "runs" / name
        man = packs.import_pack(a.target, dest, as_base=a.base)
        say(f"Imported pack {man['pack_id']} into {dest}")
    elif cmd == "verify":
        hub.cmd_verify(cfg, a.target, hub.parse_overrides(a.map), open_browser=not a.no_browser, bench=a.benchmark)
    elif cmd == "benchmark":
        hub.cmd_benchmark(cfg, a.target, hub.parse_overrides(a.map))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("hub", "publish", "verify", "benchmark", "export", "import"):
        pack_cli(sys.argv[1:])
        raise SystemExit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "train", "drive", "bench", "transfer", "diagnose"])
    ap.add_argument("diag_map", nargs="?", default=None, help="diagnose: map path (default: the configured map)")
    ap.add_argument("--min", type=float, default=4.0, help="diagnose: minutes per trial")
    ap.add_argument("--pb", default="", help="transfer: optional PB file of that map (for a target line)")
    ap.add_argument("--map")
    ap.add_argument("--run")
    ap.add_argument("--envs", type=int)
    ap.add_argument("--model")
    ap.add_argument("--fresh", action="store_true", help="ignore latest.zip")
    ap.add_argument("--new", action="store_true", help="start this new run from scratch even if another run of the "
                                                         "same map has a trained model")
    a = ap.parse_args()
    cfg = cfgmod.load()
    for key in ("map", "run", "envs"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    cfg = cfgmod.coerce(cfg)
    if a.fresh:
        cfg["resume"] = False
    if a.cmd == "diagnose":
        diagnose(cfg, a.diag_map or a.map or cfg["map"], a.pb, a.min)
        raise SystemExit(0)
    if a.cmd == "transfer":
        if not a.map:
            raise SystemExit("transfer needs --map <map not in the pool>")
        transfer(cfg, a.map, a.pb)
        raise SystemExit(0)
    if not cfg["map"] and not (cfg["run_type"] == "base" and cfg["map_pool"]):
        raise SystemExit("No map set. Pass --map or set it in the dashboard (Settings).")
    if a.cmd == "train" and not a.new and a.run is None:
        others = cfgmod.start_conflict(cfg)
        if others:
            print(f"Run '{cfg['run']}' has no model yet, but this map already has trained runs:")
            for r in others:
                print(f"  {r['name']:30s} champion {r['champion_ms'] / 1000:.3f} s" if r["champion_ms"] else
                      f"  {r['name']:30s} no champion yet", f" {r['steps']:,} steps")
            print("Continue one with --run NAME (or pick it in the dashboard), or start fresh with --new.")
            raise SystemExit(2)
    if cfg["run_type"] == "base" and cfg["algo"] == "iqn":
        raise SystemExit("Base runs are PPO only: set the algorithm to PPO.")
    if cfg["algo"] == "iqn" and a.cmd in ("train", "drive"):
        if cfg.get("analog_steer"):
            say("Analog steering is PPO-only: IQN drives with steering levels (setting: Steering levels (IQN)).")
        import iqn
        {"train": iqn.train, "drive": iqn.drive}[a.cmd](cfg)
    else:
        {"selftest": selftest, "train": train, "drive": lambda c: drive(c, a.model), "bench": bench}[a.cmd](cfg)
