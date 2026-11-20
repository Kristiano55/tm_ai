"""
Cars in their own processes (CarPool + car_worker), PPO's lock-step VecEnv on it, profiling, car benchmark.
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
from env import *  # noqa: F401,F403
from env import _make_env  # noqa: F401


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
        self.cov = (Coverage(cfg, run_dir) if coverage and cfg.get("run_type") != "base" and uses_coverage(run_dir)
                    else None)               # only runs that still use the legacy novelty terms; base runs: none
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
