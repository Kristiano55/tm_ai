"""
Shared helpers + constants of tm_ai: telemetry groups, steering levels / action space, watchdog + freeze guard,
surface vocabulary, JSON / npy files, line geometry, PB trace, target line (build_guide), map waypoints.
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


LOW_MEM_MSG = ("Windows ran out of virtual memory (error 1455: the paging file is too small). Raise Windows virtual "
               "memory (System > About > Advanced system settings > Performance > Settings > Advanced > Virtual "
               "memory) or use fewer cars.")


class LowMemory(RuntimeError):
    """Out of (virtual) memory: the run stops cleanly with LOW_MEM_MSG instead of crashing."""


def is_low_memory(text):
    t = str(text).lower()
    return "1455" in t or "paging file" in t or "memoryerror" in t


def run_rewards(run_dir):
    """The run's frozen reward weights (rewards.json in its folder or a parent folder up to runs/: base maps, pool
    maps and transfer tests share their run's file), or None = the weights in Settings."""
    if not run_dir:
        return None
    d, top = Path(run_dir).resolve(), (cfgmod.ROOT / "runs").resolve()
    while True:
        f = d / "rewards.json"
        if f.exists():
            w = read_json(f).get("weights")
            if w:
                return w
        if d == top or d.parent == d or top not in d.parents:
            return None
        d = d.parent


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


START_WP_M = 30.0        # a waypoint group this close to the spawn is the start block, not a checkpoint


def _wp_kind(wp):
    for a in ("tag", "kind", "type", "waypoint_type", "role", "name"):
        v = getattr(wp, a, None)
        if v is not None:
            return str(v).lower()
    return ""


def map_targets(m, start):
    """Centres of the map's checkpoint groups (+ finish), WITHOUT the start block. A start block among the waypoints
    used to become "checkpoint 1" a few metres after the spawn: progress was capped there until it was "taken"
    (never), so driving on paid nothing, the stuck rule fired ~30 m out and the AI learned to stop pressing gas.
    Dropped: groups tagged start (not start/finish), and on one-lap maps any group within START_WP_M of the spawn."""
    groups, kinds = {}, {}
    for wp in m.waypoints:
        if wp.group_id is not None:
            groups.setdefault(wp.group_id, []).append(np.asarray(wp.vertices, dtype=np.float64).mean(axis=0))
            kinds.setdefault(wp.group_id, _wp_kind(wp))
    one_lap = int(getattr(m, "lap_count", 1) or 1) <= 1
    keep, dropped = [], []
    for g, c in groups.items():
        centre, k = np.mean(c, axis=0), kinds[g]
        tagged_start = "start" in k and "finish" not in k
        near = one_lap and float(np.linalg.norm((centre - start)[[0, 2]])) < START_WP_M
        (dropped if (tagged_start or near) else keep).append(centre)
    if not keep:                                    # never drop everything
        keep, dropped = [np.mean(c, axis=0) for c in groups.values()], []
    return keep, dropped


AI_SECTION_MAX_X = 2.0   # an AI section drive slower than 2x your PB's section time is no target (Shemesh: 84 s pace)


def section_route(run_dir, k, pb=None):
    """The AI's best real drive through section k as (duration ticks, label, rows) or None: sec_best_<k> (it must start
    with checkpoint count <= k and end on checkpoint k+1) or the best finished lap (best_trace, same checkpoint count as
    the PB) - the faster one, and only if <= AI_SECTION_MAX_X x the PB's own time for that section."""
    run_dir = Path(run_dir)
    try:
        pb = pb if pb is not None else np.load(run_dir / "pb_trace.npy")
    except (OSError, ValueError):
        return None
    pc = _cp_cuts(pb)
    if k + 1 >= len(pc):
        return None
    pb_dur = pb[pc[k + 1], 3] - (pb[pc[k], 3] if k else 0.0)
    out = []
    try:
        seg = np.load(run_dir / f"sec_best_{k}.npy")
        if (seg.ndim == 2 and seg.shape[1] >= 5 and len(seg) >= 3 and seg[0, 4] <= k and seg[-1, 4] >= k + 1
                and seg[-1, 3] > seg[0, 3]):
            out.append((seg[-1, 3] - seg[0, 3], "AI*", seg))
    except (OSError, ValueError):
        pass
    try:
        ai = np.load(run_dir / "best_trace.npy")
        ac = _cp_cuts(ai) if ai.ndim == 2 and ai.shape[1] >= 5 else []
        if len(ac) == len(pc) and ai[-1, 4] >= k + 1:
            out.append((ai[ac[k + 1], 3] - (ai[ac[k], 3] if k else 0.0), "AI", ai))
    except (OSError, ValueError):
        pass
    out = [c for c in out if 0 < c[0] <= AI_SECTION_MAX_X * max(pb_dur, 1.0)]
    return min(out, key=lambda c: c[0]) if out else None


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
            pb_dur = next(c[0] for c in cand if c[1] == "PB")
            ai_c = [c for c in cand if c[1] != "PB" and c[0] <= AI_SECTION_MAX_X * max(pb_dur, 1.0)]
            try:
                seg = np.load(run_dir / f"sec_best_{j}.npy")
                if (seg.ndim == 2 and seg.shape[1] >= 5 and len(seg) >= 3 and seg[0, 4] <= j and seg[-1, 4] >= j + 1
                        and 0 < seg[-1, 3] - seg[0, 3] <= AI_SECTION_MAX_X * max(pb_dur, 1.0)):   # a real route
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


def time_budget(cfg):
    """Seconds a lap may take in the worst case (long maps need longer timeouts)."""
    return float(cfg["max_lap_seconds"])
