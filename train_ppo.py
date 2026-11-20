"""
PPO training: model creation + checkpoint conversion (migrate), champion test worker, training callback,
PB guide preparation, base runs, train(), play champion (drive), transfer test.
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
from cars import *  # noqa: F401,F403


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


def break_even(cfg, run_dir):
    """Speed (m/s) from which progress pays more than the time cost: per step w_time x ticks/5 vs
    w_progress x v x ticks/100, so v = 20 w_time / w_progress - the same at 10, 30 or 50 ms."""
    w = run_rewards(run_dir) or cfg
    wp, wt = float(w.get("w_progress", 0.05)), float(w.get("w_time", 0.02))
    return 20.0 * wt / wp if wp > 0 else float("inf")


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
            if cfg.get("obs_norm", True) and hasattr(self.model.policy, "obs_norm"):
                # input statistics of this rollout; applied at the next rollout START, so the stats a rollout was
                # collected with are also the ones its PPO update uses (old log-probs stay valid)
                ob = self.model.rollout_buffer.observations
                self.norm_pending = np.asarray(ob, np.float32).reshape(-1, ob.shape[-1])

        def ema_update(self, reset=False):
            """Smoothed copy of the weights. Champion tests drive with it: steadier than the latest update."""
            sd = self.model.policy.state_dict()
            if reset or getattr(self, "ema", None) is None:
                self.ema = {k: v.detach().clone() for k, v in sd.items()}
                return
            for k, v in sd.items():
                if v.dtype.is_floating_point and not k.startswith("obs_norm."):   # input stats: the current ones
                    self.ema[k].mul_(0.7).add_(v.detach(), alpha=0.3)
                else:
                    self.ema[k] = v.detach().clone()

        def _on_rollout_start(self):
            if getattr(self, "norm_pending", None) is not None:
                self.model.policy.obs_norm.update(self.norm_pending)
                self.norm_pending = None
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
                            + f". Progress beats the time cost from {break_even(cfg, run_dir):.0f} m/s (any decision "
                            "interval). Stopping can look attractive; lower penalties or raise Progress.")
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
    reward_notice(run_dir)
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
