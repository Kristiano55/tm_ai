"""
TMEnv (gymnasium): one car on one map - observations, progress along the line, lap rules (stuck, off-zone,
far off line, behind ghost, timeout), mid-track starts, section failure tracking.
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
from common import *  # noqa: F401,F403
from common import _f, _cp_cuts, _wp_kind, _vocab_load, _VOCAB, _hang_file, _HANG, _P  # noqa: F401
from rewards import *  # noqa: F401,F403
import rewards as RW  # noqa: F401


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
        # every term the reward code knows; Settings only has the minimal set (+ ghost, wall hit), the legacy terms
        # are 0 unless the run has frozen weights (rewards.json: old runs, minimal-preset runs)
        self.W = {k: float(cfg.get(k, 0.0)) for k in cfgmod.ALL_TERMS}
        fw = run_rewards(run_dir)
        if fw:
            self.W.update({k: float(fw.get(k, 0.0)) for k in cfgmod.ALL_TERMS})
        lg = fw or {}                                 # settings only legacy terms used (VCPs, coverage stretches)
        self.iqn = cfg.get("algo") == "iqn"          # IQN charges failure itself: no give-up penalty here
        self.STEER = STEER_LEVELS[cfg["steer_mode"]]
        if self.iqn and int(cfg.get("_iqn_levels") or 21) != len(self.STEER):     # finer IQN steering (new runs)
            self.STEER = steer_levels(int(cfg["_iqn_levels"]), cfg["steer_mode"])
        self.slug, self.pool_car = cfg.get("_slug") or None, bool(cfg.get("_pool"))   # base / pool map tags
        self.pnorm, self.pscale = float(cfg.get("_pnorm") or 0.0), 1.0     # map-neutral progress (base runs)
        self.cut_behind = float(cfg["cut_behind_s"])
        self.ghost_window = float(cfg["ghost_window_s"])
        self.vcp_every = float(lg.get("vcp_every_s", 0.0))
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
        self.start_focus = float(cfg.get("start_focus", 0.5))
        self.tloss, self._seg_k, self._seg_phi = np.zeros(0), None, 0.0
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
        self.cov_len = max(5.0, float(lg.get("cov_len", 25)))
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
        start = np.asarray(self.g.map.start_position, dtype=np.float64)
        targets, self.dropped_wp = map_targets(self.g.map, start)
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
        """start_focus (0.5) of the picks go where laps fail (fails) or lose time to the ghost (tloss), half each
        when both are known; the rest uniform, so every start keeps some chance."""
        n, f = len(self.starts), self.start_focus
        if len(self.tloss) != n:
            self.tloss = np.zeros(n)
        parts = [w / w.sum() for w in (self.fails, self.tloss) if len(w) == n and w.sum() > 0]
        focus = np.mean(parts, axis=0) if parts else np.full(n, 1.0 / n)
        p = (1.0 - f) / n + f * focus
        i = int(self.np_random.choice(n, p=p / p.sum()))
        self.fails *= 0.995
        self.tloss *= 0.995
        return i

    def _track_loss(self, s, phi):
        """Time lost to the ghost between consecutive start points (per start point, smoothed): where to start more."""
        ss = np.array([st.get("s", np.nan) for st in self.starts], dtype=np.float64)
        if len(self.tloss) != len(ss):
            self.tloss = np.zeros(len(ss))
        ok = np.where(ss <= s, ss, -np.inf)
        k = int(np.argmax(ok)) if np.isfinite(ok).any() else -1
        if k == self._seg_k:
            return
        if self._seg_k is not None and self._seg_k >= 0 and k >= 0 and ss[k] > ss[self._seg_k]:
            lost = max(0.0, self._seg_phi - phi)                  # seconds lost to the ghost in that stretch
            self.tloss[self._seg_k] = 0.8 * self.tloss[self._seg_k] + 0.2 * min(lost, 5.0)
        self._seg_k, self._seg_phi = k, phi

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

    def _read_weights(self):
        """Pick up weights saved from the dashboard (applies from the next lap on); runs with frozen weights
        (rewards.json) keep theirs."""
        d = run_rewards(self.run_dir) or read_json(cfgmod.CONFIG_PATH)
        for k in cfgmod.ALL_TERMS:
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
        pb = None
        if self.run_dir is not None and (self.run_dir / "pb_trace.npy").exists():
            try:
                pb = np.load(self.run_dir / "pb_trace.npy")
            except (OSError, ValueError):
                pb = None
        if pb is not None and len(pb) > 2:         # exact: where your PB's checkpoint counter went up
            cut = [i for i in range(1, len(pb)) if pb[i, 4] > pb[i - 1, 4]]
            arcs = []
            for i in cut:
                d = np.linalg.norm(pts - pb[i, :3], axis=1)
                j = int(np.argmin(d))
                if d[j] > 60.0:
                    arcs = None
                    break
                arcs.append(float(cum[j]))
            if arcs:
                return np.sort(np.array(arcs))
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
        return RW.hit_penalty(self, w, tf, hit, speed_loss)

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
        return RW.pointless_release(self, t, pos, s, speed)

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
        self._seg_k, self._seg_phi = None, 0.0
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
        r, phi, kn, ck = RW.step_terms(self, r, t, pos, s, dp, speed, steer, mode, taken, tick)   # all per-step terms
        if self.starts and self.prof is not None and self.mid_p > 0:
            self._track_loss(s, phi)
        tf = self.tf
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
        give_up = RW.give_up_cost(self, W, length, s)
        if t.finished:
            terminated, end = True, "finished"
            r["finish"] = RW.finish_bonus(self, W, int(t.race.race_time_ms))
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
