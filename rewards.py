"""
Reward terms of TMEnv (per step, lap end) and the reward state kept by the main process / testers:
reward presets, champion ghost, wall-hit ramp, coverage (novelty -> speed focus).
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


def reward_notice(run_dir):
    """Log once at the start which reward weights this run uses: frozen ones (rewards.json) + any legacy terms on."""
    w = run_rewards(run_dir)
    if not w:
        say("Rewards: " + ", ".join(f"{k} {float(v):g}" for k, v in sorted(_settings_weights().items()) if float(v))
            + " (Settings).")
        return
    d = read_json(Path(run_dir) / "rewards.json")
    legacy = [f"{k} {float(w[k]):g}" for k in cfgmod.LEGACY_TERMS if float(w.get(k, 0) or 0)]
    say(f"Rewards: this run keeps its saved weights ({d.get('preset') or 'rewards.json'}; Settings changes don't "
        "apply to it). Legacy terms active: " + (", ".join(legacy) if legacy else "none") + ".")


def _settings_weights():
    c = read_json(cfgmod.CONFIG_PATH)
    return {k: c.get(k, d) for k, (d, _, _) in cfgmod.WEIGHTS.items()}


def uses_coverage(run_dir):
    """Only the legacy novelty / known / focus terms need the coverage counts."""
    w = run_rewards(run_dir) or {}
    return any(float(w.get(k, 0) or 0) for k in ("w_novel", "w_known", "w_focus"))


def champion_ghost(env, run_dir, ms, log=None):
    """Rebuild the ghost (reward target) from the champion lap: called only when the champion improves (or no ghost
    exists). PB-guide runs rebuild the guide (faster of PB vs AI best per section). Logs event k=target."""
    run_dir = Path(run_dir)
    try:
        if own_on(run_dir):                       # own line: line + ghost = the new champion's lap
            gms = build_own_line(run_dir)
            if gms is None:
                return False
            say(f"Champion {ms / 1000:.3f} s -> own line + ghost rebuilt from it")
        elif env.guide:
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


class Coverage:
    """How well the AI knows each cov_len-m stretch of the target line (main process; saved as coverage.json).
    passes = laps that drove through the whole stretch; known = 1 - exp(-passes / cov_passes).
    A section's stretches reset when its target route swaps (PB <-> AI, sections.json)."""

    def __init__(self, cfg, run_dir):
        self.cfg, self.dir = cfg, Path(run_dir)
        self.file = self.dir / "coverage.json"
        lg = run_rewards(run_dir) or {}
        self.L, self.tau = max(5.0, float(lg.get("cov_len", 25))), max(1.0, float(lg.get("cov_passes", 100)))
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
        W = run_rewards(self.dir) or read_json(cfgmod.CONFIG_PATH)
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


def hit_penalty(env, w, tf, hit, speed_loss):
    """Wall hit: costs in proportion to the speed it cost (a glancing touch little, >= 10 m/s lost the full
    amount), times the run's ramp-in factor (hit_ramp.json: lower while the run rarely finishes)."""
    if not hit:
        return 0.0
    return -w * tf * float(hit) * env.hit_factor * min(1.0, max(0.0, float(speed_loss)) / 10.0)


def pointless_release(env, t, pos, s, speed):
    """0..1: how clearly releasing is pointless here. 1 = straight road ahead and nothing in front;
    0 in corners, near obstacles, in the air or at low speed, so the agent stays free to release there."""
    if speed < 15.0 or not np.any(np.asarray(t.wheels.contact, dtype=bool)):
        return 0.0
    q = t.vehicle.rotation
    ang = 0.0
    ds = np.array([25.0, 50.0, max(60.0, speed * 1.5)])     # how much the route bends in the next ~1.5 s
    loc = to_car_frame_many(q, env._route_points(s + ds) - pos)
    ang = max(ang, float(np.max(np.abs(np.arctan2(loc[:, 0], loc[:, 2])))))
    straight = float(np.clip(1.0 - ang / 0.12, 0.0, 1.0))   # fades out from ~0 to ~7 degrees of bend
    hit, dist = env._lid
    fwd = float(np.min(dist[13:19]))                         # centre of the 32-ray fan
    clear = float(np.clip((fwd - speed * 1.2) / (speed * 0.8 + 1.0), 0.0, 1.0))
    return straight * clear


def give_up_cost(env, W, length, s):
    """Ending a lap early (stuck / off the track) costs what driving the rest of it would have paid,
    so giving up is never cheaper than pushing on through a bad section."""
    give_up = W["w_progress"] * max(0.0, length - s) * env.pscale
    if env.iqn:                            # IQN charges failure itself (time cost for the rest of the mini-race)
        give_up = 0.0
    return give_up


def finish_bonus(env, W, race_ms):
    ref = env.champ_ms if env.champ_ms else env.max_ticks * 10
    gain = (ref - race_ms) / 1000.0                                      # seconds faster than the champion
    return W["w_finish"] + float(np.clip(W["w_finish_time"] * gain, -20.0, 20.0))


def step_terms(env, r, t, pos, s, dp, speed, steer, mode, taken, tick):
    """All per-step reward terms of TMEnv.step, in r (dict, summed by the caller). Same order and state updates as
    before the split: env.taken (checkpoint), env.vcp_i (virtual checkpoints), env.phi (ghost gap).
    Returns (r, phi, known, stretch)."""
    W = env.W
    r["progress"] = W["w_progress"] * dp * env.pscale
    kn, ck = env._known(env._s)
    if kn is not None and dp > 0 and (W["w_novel"] or W["w_known"]):    # new stretch pays more, known less
        r["novel"] = r["progress"] * (W["w_novel"] * (1.0 - kn) - min(W["w_known"], 0.5) * env._wks(ck) * kn)
    tf = env.tf
    r["time"] = -W["w_time"] * tf
    r["steer"] = -W["w_steer_smooth"] * abs(steer / 127.0 - env.pa[0])
    if taken > env.taken:
        r["checkpoint"] = W["w_checkpoint"]
        env.taken = taken
    phi = env._phi(s, tick)                                                 # seconds ahead of the ghost
    while env.vcp_i < len(env.vcp_s) and s >= env.vcp_s[env.vcp_i]:   # virtual checkpoint passed
        split = (env.vcp_t[env.vcp_i] - tick) / 100.0                      # seconds ahead of the PB/ghost here
        r["vcp"] = r.get("vcp", 0.0) + W["w_vcp"] + float(np.clip(W["w_vsplit"] * split, -3.0, 3.0))
        env.vcp_i += 1
    if env.prof is not None and W["ghost_weight"]:
        # time gained on the ghost this step, but only inside a window behind it: once a lap is far
        # behind, losing more time costs nothing extra, so ending the lap early can never pay off
        lo = -max(env.ghost_window, 0.1)
        r["ghost"] = float(np.clip(W["ghost_weight"] * (max(phi, lo) - max(env.phi, lo)), -5.0, 5.0))
        if kn is not None and W["w_focus"]:                              # known stretch: time gained counts more
            r["focus"] = r["ghost"] * W["w_focus"] * kn
    env.phi = phi
    if env.guide and W["w_guide"] and env.dense is not None:            # pulled toward the target line,
        off = pos - env._route_point(s)                                     # but only while slower than it
        d = float(np.hypot(off[0], off[2]))
        behind = float(np.clip(-phi / 0.3, 0.0, 1.0))
        r["guide"] = -W["w_guide"] * tf * min(d / 8.0, 1.0) * behind
    r["crash"] = -W["w_crash"] * max(0.0, env.prev_speed - speed - 3.0)    # sudden speed loss = wall hit
    if mode in BRAKE:
        r["brake"] = -W["w_brake"] * tf
    elif mode == 1:
        r["coast"] = -W["w_coast"] * tf
        if W["w_release"]:
            r["release"] = -W["w_release"] * tf * pointless_release(env, t, pos, s, speed)
    if W["w_slip"]:
        r["slip"] = -W["w_slip"] * tf * float(np.mean(np.asarray(t.wheels.slipping, dtype=np.float32)))
    if W["w_air"] and not np.any(np.asarray(t.wheels.contact, dtype=bool)):
        r["air"] = -W["w_air"] * tf
    if W["w_speed"]:
        r["speed"] = W["w_speed"] * tf * speed / 100.0
    if W["w_hit"] and env.extra:
        r["hit"] = hit_penalty(env, W["w_hit"], tf, _f(t.runtime, "wall_hit_this_step"), env.prev_speed - speed)
    if W["w_wall"]:                                                          # side rays of the fan
        hit, dist = env._lid
        near = float(np.min(np.concatenate([dist[:4], dist[28:32]])))
        r["wall"] = -W["w_wall"] * tf * speed / 100.0 * max(0.0, 1.0 - near / 2.0)
    return r, phi, kn, ck
