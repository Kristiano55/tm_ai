"""
Trackmania 2020 RL agent on tickgym (PPO, runs on CPU).

    python tm_ai.py selftest --map "C:\\Maps\\X.Map.Gbx"
    python tm_ai.py train                (settings come from config.json / the dashboard)
    python tm_ai.py drive                (plays runs/<run>/champion.zip, writes play.txt for TICK)

Normally you start/stop all of this from the dashboard: python dashboard.py
"""
import argparse
import json
import os
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


def build_guide(run_dir):
    """Target line = per checkpoint section, the faster of your PB and the AI's best finished run.
    So the line starts as your PB and switches section by section wherever the AI finds something faster."""
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
    rows, t0, chosen = [], 0.0, []
    for j in range(n):
        dur = {k: srcs[k][cuts[k][j + 1], 3] - (srcs[k][cuts[k][j], 3] if j else 0.0) for k in use}
        k = min(use, key=lambda kk: dur[kk])
        tr, a, b = srcs[k], cuts[k][j], cuts[k][j + 1]
        part = tr[a:b + 1] if j == n - 1 else tr[a:b]
        base = tr[a, 3] if j else 0.0
        for r in part:
            rows.append([r[0], r[1], r[2], t0 + r[3] - base])
        t0 += dur[k]
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


class TMEnv(gym.Env):
    """One car on one map. Action = (steer level, throttle mode)."""

    LOOK = (10, 25, 50, 90, 140)      # route lookahead in metres
    N_BASE = N_BASE0 + len(EXTRA_NAMES)   # non-lidar features
    LOOK_OFF = 28                         # where the lookahead features start

    def __init__(self, cfg, run_dir=None, shorten=True, role="training"):
        self.STEER = STEER_LEVELS[cfg["steer_mode"]]
        self.W = {k: float(cfg[k]) for k in cfgmod.WEIGHTS}
        self.cut_behind = float(cfg["cut_behind_s"])
        self.run_dir, self.role, self.shorten = (Path(run_dir) if run_dir else None), role, shorten
        self.dense, self._line_m, self.prof, self._prof_m, self.champ_ms = None, None, None, None, None
        self.phi, self._prev_lv, self.since_hit, self.air = 0.0, None, 20, 0
        self.hist = []
        self.extra = bool(cfg["extra_obs"])
        self.groups = GROUPS_EXTRA if self.extra else GROUPS
        self.gspd = None
        self.mid_p = float(cfg["mid_start_p"]) if role == "training" else 0.0
        self.mid_every = max(20, int(cfg["mid_start_every_s"] * 100 / cfg["ticks_per_step"]))
        self.starts, self.fails, self._ca_m, self.start_i, self.acts = [], np.zeros(0), "unset", None, []
        self._starts_line = "unset"
        self.guide = bool(cfg.get("pb_file")) and self.run_dir is not None and (self.run_dir / "pb_trace.npy").exists()
        self.cprof, self._cprof_key = None, None

        self.session = Client().open_map(cfg["map"])
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
        self.action_space = spaces.MultiDiscrete([len(self.STEER), 3])    # 3 = gas / coast / brake
        self.observation_space = spaces.Box(-10.0, 10.0, (self.N_BASE + len(self.lidar.rays),), np.float32)
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
            limit //= (2 if self.guide else 1)
            for i, (steer, mode) in enumerate(acts[:-self.mid_every // 2]):
                t = self.g.step(self.ticks, steer=int(steer), accelerate=(mode == 0), brake=(mode == 2))
                pa = np.concatenate([[steer / 127.0], np.eye(3)[int(mode)], pa[:4]])
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
            if self.starts:
                say(f"[{self.role}] {len(self.starts)} mid-track start points built (champion lap + your PB)")
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
        self.g = self.session.connect(self.groups, self.lidar)

    def _read_weights(self):
        """Pick up weights saved from the dashboard (applies from the next lap on)."""
        d = read_json(cfgmod.CONFIG_PATH)
        for k in cfgmod.WEIGHTS:
            try:
                self.W[k] = float(d[k])
            except (KeyError, TypeError, ValueError):
                pass
        try:
            self.cut_behind = float(d["cut_behind_s"])
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

    def _phi(self, s, tick):
        """Seconds ahead of the ghost (negative = behind)."""
        if self.prof is None:
            return 0.0
        return float((np.interp(s, self.prof[0], self.prof[1]) - tick) / 100.0)

    def _finished(self, ms):
        """Called on every finished lap: keep the fastest run ever (best_run.txt) and update the ghost."""
        try:
            self._offer_best(ms)
            self._offer_profile(ms)
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
        if self.guide:
            r = build_guide(self.run_dir)
            if r:
                say(f"[{self.role}] New best run {ms / 1000:.3f} s -> target line sections: {' '.join(r[1])} "
                    f"(theoretical {r[0] / 1000:.3f} s)")

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

    def _limit_ticks(self):
        """Once a champion exists, cut hopeless laps at 1.6x its time: much faster training."""
        if self.shorten and self.champ_ms:
            return min(self.max_ticks, int(self.champ_ms / 10 * 1.6) + 300)
        return self.max_ticks

    def _progress(self, pos, taken):
        """Metres driven along the route (reference line if there is one)."""
        if self.dense is not None:
            s, self.di = line_progress(*self.dense, pos, self.di)
            return s
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

    def _behind_champ(self, s, tick, phi):
        if self.cprof is not None:
            return float((np.interp(s, self.cprof[0], self.cprof[1]) - tick) / 100.0)
        return phi if self.prof is not None else 0.0

    def _pointless_release(self, t, pos, s, speed):
        """0..1: how clearly releasing is pointless here. 1 = straight road ahead and nothing in front;
        0 in corners, near obstacles, in the air or at low speed, so the agent stays free to release there."""
        if speed < 15.0 or not np.any(np.asarray(t.wheels.contact, dtype=bool)):
            return 0.0
        q = t.vehicle.rotation
        ang = 0.0
        for d in (25.0, 50.0, max(60.0, speed * 1.5)):          # how much the route bends in the next ~1.5 s
            loc = to_car_frame(q, self._route_point(s + d) - pos)
            ang = max(ang, abs(float(np.arctan2(loc[0], loc[2]))))
        straight = float(np.clip(1.0 - ang / 0.12, 0.0, 1.0))   # fades out from ~0 to ~7 degrees of bend
        hit = np.asarray(t.lidar.hit, dtype=bool)
        dist = np.where(hit, np.asarray(t.lidar.distance, dtype=np.float32), self.ray_range)
        fwd = float(np.min(dist[13:19]))                         # centre of the 32-ray fan
        clear = float(np.clip((fwd - speed * 1.2) / (speed * 0.8 + 1.0), 0.0, 1.0))
        return straight * clear

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
        out += [0.0]          # icing: read 1.0 on a summer map in selftest -> disabled until the field is understood
        hit = _f(rt, "wall_hit_this_step")
        out += [_f(rt, "forward_motion_alignment_scale", 1.0), hit, _f(rt, "roof_contact_this_step"),
                _f(rt, "wetness"), _f(rt, "surface_grip_scale", 1.0), _f(rt, "soft_surface_wheel_count") / 4.0,
                0.0, _f(v, "in_water")]     # on_ice disabled (same reason as icing)
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
        cp, ahead = self._route_point(s), self._route_point(s + 5.0)
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
        for dist in self.LOOK:                                      # where the route goes next
            loc = to_car_frame(q, self._route_point(s + dist) - pos)
            n = float(np.linalg.norm(loc)) + 1e-6
            f += [float(np.arctan2(loc[0], loc[2]) / np.pi), float(loc[1] / n)]
        f += self._extras(t, q, pos, s)
        hit = np.asarray(t.lidar.hit, dtype=bool)
        dist = np.where(hit, np.asarray(t.lidar.distance, dtype=np.float32), self.ray_range)
        f += list(np.minimum(dist, self.ray_range) / self.ray_range)
        self._last_obs = np.clip(np.nan_to_num(np.asarray(f, dtype=np.float32)), -10, 10)
        return self._last_obs

    # --- gym API ---------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        try:
            t = self.g.reset(seed=0)            # physics is deterministic
        except TransportError:
            self._connect()
            t = self.g.reset(seed=0)
        self._read_weights()
        self._read_best()
        self._load_line()
        self._load_profile()
        self._load_champ()
        self._build_starts()
        self.taken, self.slow, self.prev_speed, self.di = 0, 0, 0.0, 0
        self._prev_lv, self.since_hit, self.air = None, 20, 0
        self.pa = np.zeros(8)
        self.limit = self._limit_ticks()
        self.start_i, self.acts = None, []
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
        pos = np.asarray(t.vehicle.position, dtype=np.float64)
        tick = int(t.race.race_tick) if self.start_i is not None else 0
        self.di = 0
        self._s = self._global_progress(pos, self.taken) if self.start_i is not None else self._progress(pos, 0)
        self.phi = self._phi(self._s, tick)
        self.hist = [[*pos, tick, self.taken]]
        return self._obs(t, self._s), {}

    def step(self, action):
        si, mode = int(action[0]), int(action[1])
        steer = int(self.STEER[si])
        self.acts.append((steer, mode))
        try:
            t = self.g.step(self.ticks, steer=steer, accelerate=(mode == 0), brake=(mode == 2))
        except TransportError:                  # game hiccup: reopen the gym, end this lap
            self._connect()
            info = {"finished": False, "race_ms": 0, "cp": self.taken, "tick": 0, "end": "gym lost"}
            return self._last_obs, 0.0, False, True, info

        W = self.W
        pos = np.asarray(t.vehicle.position, dtype=np.float64)
        taken, tick = int(t.race.checkpoints_taken), int(t.race.race_tick)
        s = self._progress(pos, taken)
        speed = float(t.vehicle.speed)
        reward = W["w_progress"] * float(np.clip(s - self._s, -10.0, 10.0)) - W["w_time"]
        reward -= W["w_steer_smooth"] * abs(steer / 127.0 - self.pa[0])
        if taken > self.taken:
            reward += W["w_checkpoint"]
            self.taken = taken
        phi = self._phi(s, tick)                                                 # seconds ahead of the ghost
        if self.prof is not None and W["ghost_weight"]:
            reward += float(np.clip(W["ghost_weight"] * (phi - self.phi), -5.0, 5.0))   # = time gained this step
        self.phi = phi
        if self.guide and W["w_guide"] and self.dense is not None:            # pulled toward the target line,
            off = pos - self._route_point(s)                                     # but only while slower than it
            d = float(np.hypot(off[0], off[2]))
            behind = float(np.clip(-phi / 0.3, 0.0, 1.0))
            reward -= W["w_guide"] * min(d / 8.0, 1.0) * behind
        reward -= W["w_crash"] * max(0.0, self.prev_speed - speed - 3.0)         # sudden speed loss = wall hit
        if mode == 2:
            reward -= W["w_brake"]
        elif mode == 1:
            reward -= W["w_coast"]
            if W["w_release"]:
                reward -= W["w_release"] * self._pointless_release(t, pos, s, speed)
        if W["w_slip"]:
            reward -= W["w_slip"] * float(np.mean(np.asarray(t.wheels.slipping, dtype=np.float32)))
        if W["w_air"] and not np.any(np.asarray(t.wheels.contact, dtype=bool)):
            reward -= W["w_air"]
        if W["w_speed"]:
            reward += W["w_speed"] * speed / 100.0
        if W["w_hit"] and self.extra:
            reward -= W["w_hit"] * _f(t.runtime, "wall_hit_this_step")
        if W["w_wall"]:                                                          # side rays of the fan
            hit = np.asarray(t.lidar.hit, dtype=bool)
            dist = np.where(hit, np.asarray(t.lidar.distance, dtype=np.float32), self.ray_range)
            near = float(np.min(np.concatenate([dist[:4], dist[28:32]])))
            reward -= W["w_wall"] * speed / 100.0 * max(0.0, 1.0 - near / 2.0)
        self.prev_speed, self._s = speed, s
        self.pa = np.concatenate([[steer / 127.0], np.eye(3)[mode], self.pa[:4]])
        self.hist.append([*pos, tick, taken])

        end, terminated, truncated = "", False, False
        if t.finished:
            terminated, end = True, "finished"
            ref = self.champ_ms if self.champ_ms else self.max_ticks * 10
            gain = (ref - int(t.race.race_time_ms)) / 1000.0                     # seconds faster than the champion
            reward += W["w_finish"] + float(np.clip(W["w_finish_time"] * gain, -20.0, 20.0))
            if self.start_i is None:            # mid-track laps are not full runs: not saved as best run / ghost
                self._finished(int(t.race.race_time_ms))
        elif t.race.entered_off_zone:
            terminated, end, reward = True, "off-zone", reward - W["w_offzone"]
        else:
            self.slow = self.slow + self.ticks if (speed < 1.5 and tick > 300) else 0
            if self.slow >= 200:                                                # stuck for 2 s
                terminated, end, reward = True, "stuck", reward - W["w_stuck"]
            elif (self.role == "training" and self.cut_behind and tick > 500
                  and self._behind_champ(s, tick, phi) < -self.cut_behind):      # tests always drive the full lap
                truncated, end = True, "behind ghost"                            # hopeless lap: save the time
            elif tick >= self.limit:
                truncated, end = True, "timeout"
        if terminated or truncated:
            self._blame(s, end not in ("finished",))
        info = {"finished": bool(t.finished), "race_ms": int(t.race.race_time_ms),
                "cp": taken, "tick": tick, "end": end, "mid": self.start_i is not None}
        return self._obs(t, s), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass


def _make_env(cfg, run_dir):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(TMEnv(cfg, run_dir))


def run_episode(env, model):
    obs, _ = env.reset()
    done, info = False, {}
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        done = term or trunc
    return info


# --- models: creation and warm-start migration between steering layouts ------------------------
def make_model(env):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, device="cpu", verbose=0,                 # MLP policy -> CPU is the right device
               n_steps=512, batch_size=512, n_epochs=10, gamma=0.995, gae_lambda=0.95,
               learning_rate=3e-4, ent_coef=0.005, target_kl=0.02,    # target_kl: no destructive updates
               policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))


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
        sd_new[key] = torch.tensor(np.concatenate([steer, o[n_old:]]), dtype=sd_new[key].dtype)
    new.policy.load_state_dict(sd_new)
    new.num_timesteps, new._n_updates = old.num_timesteps, old._n_updates


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
    if not same_levels or old_base != new_base:
        for name in ("latest", "champion"):
            p = run_dir / f"{name}.zip"
            if not p.exists():
                continue
            try:
                old, new = PPO.load(str(p), device="cpu"), make_model(env)
                transfer_policy(old, new, old_levels, new_levels, old_base, new_base)
                new.save(str(run_dir / name))
                say(f"Converted {name}.zip to the current steering/observation layout.")
            except Exception as e:     # never load weights under the wrong layout
                p.replace(run_dir / f"{name}.unconverted.zip")
                say(f"Could not convert {name}.zip ({str(e)[:80]}); moved aside, starting fresh.")
    write_json(lv_file, {"steer": new_levels.tolist(), "obs_base": new_base})


# --- champion tests, dashboard events, stop conditions ----------------------------------------
def _make_callback(cfg, run_dir):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import get_schedule_fn

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
            self.reason, self.last_v, self.test_env, self.tests_off = "step budget reached", 0.0, None, False
            self.best = read_json(best_file)
            self.ema_update(reset=True)
            log(k="start", cfg=cfg)
            self.status("running")
            say("Training started: %d cars, %s steering, CPU." % (cfg["envs"], cfg["steer_mode"]))

        def status(self, state, reason=""):
            write_json(status_file, {"state": state, "reason": reason, "pid": os.getpid(),
                                     "heartbeat": time.time(), "started": self.t0})

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
            ds = cfg["decay_steps"]
            if ds:                                         # exploit more as training goes on
                f = 0.5 ** (self.model.num_timesteps / ds)
                self.model.lr_schedule = get_schedule_fn(max(3e-4 * f, 3e-5))
                self.model.ent_coef = max(0.005 * f, 0.0005)
            nv = self.model.logger.name_to_value
            if "train/value_loss" in nv:
                steps = self.model.num_timesteps - self.steps0
                log(k="upd", u=int(self.model._n_updates), steps=int(self.model.num_timesteps),
                    v=self.last_v, vl=float(nv["train/value_loss"]),
                    ent=float(nv.get("train/entropy_loss", 0.0)), ev=float(nv.get("train/explained_variance", 0.0)),
                    speed=steps * cfg["ticks_per_step"] / 100.0 / max(time.time() - self.t0, 1e-6))
                self.save_all()

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done:
                    self.laps += 1
                    self.since += 1
                    fin = bool(info.get("finished"))
                    log(k="lap", ms=int(info["race_ms"]) if fin else None, cp=int(info.get("cp", 0)),
                        end=info.get("end", ""), mid=1 if info.get("mid") else 0)
            now = time.time()
            if now - self.last_chk > 2.0:
                self.last_chk = now
                if (run_dir / "STOP").exists():
                    self.reason = "stopped by user"
                    return False
                if cfg["time_limit_min"] and now - self.t0 > cfg["time_limit_min"] * 60:
                    self.reason = "time limit reached"
                    return False
                self.status("running")
            if self.since >= cfg["test_every_laps"] and not self.tests_off:
                self.champion_test()
                champ, pb = self.best.get("champion_ms"), cfg["pb_seconds"]
                if cfg["stop_at_pb"] and pb and champ and champ <= pb * 1000:
                    self.reason = "reached your PB"
                    return False
            return True

        def champion_test(self):
            self.since = 0
            try:
                if self.test_env is None:      # needs one extra gym slot besides the training cars
                    self.test_env = TMEnv(cfg, run_dir, shorten=False, role="champion test")
                cur = {k: v.detach().clone() for k, v in self.model.policy.state_dict().items()}
                if getattr(self, "ema", None) is not None:
                    self.model.policy.load_state_dict(self.ema)
                try:
                    info = run_episode(self.test_env, self.model)
                    fin = bool(info["finished"])
                    rec = self.test_env.g.export() if fin else None
                    if fin and (self.best.get("champion_ms") is None or int(rec.finish_ms) < self.best["champion_ms"]):
                        self.model.save(run_dir / "champion")        # champion = the smoothed weights
                finally:
                    self.model.policy.load_state_dict(cur)
            except Exception as e:             # e.g. no free gym slot: keep training, skip tests
                self.tests_off = True
                say(f"Champion tests disabled: {e}. Lower 'envs' by one to enable them.")
                return
            self.tests += 1
            ms = int(rec.finish_ms) if fin else None
            champ = self.best.get("champion_ms")
            promoted = rolled = False
            if fin:
                self.best.setdefault("first_test_ms", ms)
            if fin and (champ is None or ms < champ):
                promoted, self.bad = True, 0
                self.best["champion_ms"] = champ = ms
                save_npy(run_dir / "champ_trace.npy", np.asarray(self.test_env.hist, dtype=np.float64))

                InputText.write(str(run_dir / "champion.txt"), rec.actions)
            elif champ is not None and (not fin or ms > champ + cfg["rollback_margin_s"] * 1000):
                self.bad += 1
                if self.bad >= cfg["rollback_after"]:
                    try:
                        self.model.set_parameters(str(run_dir / "champion.zip"), exact_match=True, device="cpu")
                        self.ema_update(reset=True)
                        rolled = True
                    except Exception as e:
                        say(f"Rollback failed ({str(e)[:60]})")
                    self.bad = 0
            else:
                self.bad = 0
            ca = run_dir / "champion_actions.json"
            if fin and (promoted or not ca.exists()):          # route for the mid-track starts
                write_json(ca, {"ms": ms, "acts": self.test_env.acts})
            if fin and not (run_dir / "champ_trace.npy").exists():
                save_npy(run_dir / "champ_trace.npy", np.asarray(self.test_env.hist, dtype=np.float64))
            line_ms = read_json(line_meta).get("ms") or self.best.get("first_test_ms")   # old runs: line came from the first test
            refresh = cfg["line_refresh_s"] and promoted and line_ms and ms <= line_ms - cfg["line_refresh_s"] * 1000
            if self.test_env.guide:
                refresh = False                                  # guide mode: the line is PB/best sections
            if fin and (not line_file.exists() or refresh):     # reference line follows the champion
                self.rebuild_line(ms, "rebuilt from the new champion" if line_file.exists() else "created")
            write_json(best_file, self.best)
            log(k="test", lap_n=self.laps, ms=ms, champ_ms=champ, promoted=promoted, rolled_back=rolled)
            say(f"Champion test {self.tests}: " + (f"{ms / 1000:.3f} s" if fin else f"no finish ({info.get('end')})")
                + (" -> NEW CHAMPION" if promoted else "") + (" -> rolled back to champion" if rolled else ""))

        def rebuild_line(self, ms, what):
            """New reference line from the test lap just driven, and a fresh ghost on it (= that lap)."""
            env = self.test_env
            save_npy(line_file, make_line(np.asarray(env.hist)[:, :3]))
            env._load_line()
            for f in ("profile.json", "profile.npy"):
                (run_dir / f).unlink(missing_ok=True)
            env._offer_profile(ms)
            write_json(line_meta, {"ms": ms, "t": time.time()})
            say(f"Reference line + ghost {what} ({ms / 1000:.3f} s lap). Expect a short dip while cars adapt.")

        def save_all(self):
            self.model.save(run_dir / "latest")
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(str(run_dir / "vecnorm.pkl"))

        def _on_training_end(self):
            self.save_all()
            log(k="stop", reason=self.reason)
            self.status("stopped", self.reason)
            say("Stopped: " + self.reason)
            if self.test_env is not None:
                self.test_env.close()

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


def train(cfg):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    run_dir = cfgmod.run_dir(cfg)
    (run_dir / "STOP").unlink(missing_ok=True)
    torch.set_num_threads(2)                                   # tiny MLP: more threads only slow it down
    prepare_guide(cfg, run_dir)
    venv = SubprocVecEnv([partial(_make_env, cfg, str(run_dir)) for _ in range(cfg["envs"])])
    vn_file = run_dir / "vecnorm.pkl"
    env = None
    if cfg["resume"] and vn_file.exists():
        try:
            env = VecNormalize.load(str(vn_file), venv)
            env.training = True
            say("Loaded reward scaling (vecnorm.pkl)")
        except Exception as e:
            say(f"Could not load vecnorm.pkl ({str(e)[:60]}), relearning reward scale")
    if env is None:
        env = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)
    migrate(run_dir, cfg, env)

    model, latest = None, run_dir / "latest.zip"
    if cfg["resume"] and latest.exists():
        try:
            model = PPO.load(str(latest), env=env, device="cpu")
            model.target_kl = 0.02
            say("Resumed from latest.zip")
        except Exception as e:
            say(f"Cannot resume ({str(e)[:80]}), starting fresh")
    if model is None:
        model = make_model(env)
    cb = _make_callback(cfg, run_dir)
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


def drive(cfg, model_path=None):
    from stable_baselines3 import PPO

    run_dir = cfgmod.run_dir(cfg)
    model = PPO.load(model_path or str(run_dir / "champion.zip"), device="cpu")
    env = TMEnv(cfg, run_dir, shorten=False, role="play")
    say("Driving the champion (the lap is simulated in the background, the game window will not show it) ...")
    info = run_episode(env, model)
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
    env = TMEnv(cfg, cfgmod.run_dir(cfg), shorten=False)
    o, _ = env.reset()
    for _ in range(40):
        o, r, term, trunc, info = env.step(np.array([10, 0]))              # straight, full gas
    print("steering levels:", env.STEER.tolist())
    print("route points:", len(env.route) - 1, "| route length: %.0f m" % env.lap_len,
          "| reference line:", "yes" if env.dense is not None else "not yet")
    print("local velocity m/s (expect z > 0, x ~ 0):", np.round(o[1:4] * 100, 2))
    print("world up in car frame (expect ~ 0, 1, 0):", np.round(o[7:10], 2))
    print("first lookahead (angle/pi, elevation):", np.round(o[env.LOOK_OFF:env.LOOK_OFF + 2], 2))
    if env.extra:
        print("missing telemetry fields (these always read their default):", missing_fields(env) or "none")
        print("extra observations:", {n: round(float(x), 3) for n, x in zip(EXTRA_NAMES, o[N_BASE0:env.N_BASE])})
    print("lidar min/max m:", np.round((o[env.N_BASE:] * env.ray_range).min(), 1),
          np.round((o[env.N_BASE:] * env.ray_range).max(), 1), "| obs size:", o.shape[0])
    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "train", "drive"])
    ap.add_argument("--map")
    ap.add_argument("--run")
    ap.add_argument("--envs", type=int)
    ap.add_argument("--model")
    ap.add_argument("--fresh", action="store_true", help="ignore latest.zip")
    a = ap.parse_args()
    cfg = cfgmod.load()
    for key in ("map", "run", "envs"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    cfg = cfgmod.coerce(cfg)
    if a.fresh:
        cfg["resume"] = False
    if not cfg["map"]:
        raise SystemExit("No map set. Pass --map or set it in the dashboard (Settings).")
    {"selftest": selftest, "train": train, "drive": lambda c: drive(c, a.model)}[a.cmd](cfg)
