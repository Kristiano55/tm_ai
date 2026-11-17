"""
IQN learner for the Trackmania AI, modelled on Linesight (the TMNF AI that beat world records).

  * Implicit Quantile Network (distributional Q-learning), dueling head, 21 steering levels x gas/coast/brake.
  * Replay buffer: every experience is learned from many times (iqn_reuse), so rare great laps are not wasted.
  * n-step returns (3) and "mini-races": the AI maximises progress over the next `minirace_s` seconds.
    The time left in the mini-race is an input; each sample gets a random one, as in Linesight.
  * Exploration: epsilon-greedy + Boltzmann, decaying over training.
  * Champion tests run in their own process on a copy of the weights: training never pauses for them.

Started by tm_ai.py when the "algo" setting is "iqn".
"""
import json
import math
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfgmod
import tm_ai as T

N_STEP = 3
EMB = 64
N_Q, K_Q = 8, 32          # quantiles per sample for the loss / for choosing actions
KAPPA = 5e-3
BATCH = 512
WARMUP = int(os.environ.get("TMAI_IQN_WARMUP", 20000))   # experiences before learning starts


def sched(points, x):
    """Piecewise-linear schedule over env steps."""
    xs, ys = zip(*points)
    return float(np.interp(x, xs, ys))


LR = [(0, 1e-3), (3e6, 5e-5), (15e6, 1e-5)]
EPS = [(0, 1.0), (50e3, 1.0), (300e3, 0.1), (3e6, 0.03)]
BOLTZ = [(0, 0.15), (3e6, 0.03)]
GAMMA = [(0, 0.999), (2.5e6, 1.0)]
TAU_B = 0.01


class IQNNet(nn.Module):
    def __init__(self, n_in, n_act, hid=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hid), nn.LeakyReLU(), nn.Linear(hid, hid), nn.LeakyReLU())
        self.phi = nn.Linear(EMB, hid)
        self.adv = nn.Sequential(nn.Linear(hid, hid), nn.LeakyReLU(), nn.Linear(hid, n_act))
        self.val = nn.Sequential(nn.Linear(hid, hid), nn.LeakyReLU(), nn.Linear(hid, 1))
        self.register_buffer("ipi", torch.arange(1, EMB + 1, dtype=torch.float32) * math.pi)

    def forward(self, x, taus):                       # x [B, n_in], taus [B, N] -> quantiles [B, N, A]
        b, n = taus.shape
        h = self.body(x)
        e = F.leaky_relu(self.phi(torch.cos(taus.unsqueeze(-1) * self.ipi)))
        z = (h.unsqueeze(1) * e).reshape(b * n, -1)
        a, v = self.adv(z), self.val(z)
        return (v + a - a.mean(1, keepdim=True)).view(b, n, -1)

    def q(self, x, k=K_Q):
        return self(x, torch.rand(x.shape[0], k)).mean(1)


class Replay:
    def __init__(self, cap, dim):
        self.cap, self.i, self.n = cap, 0, 0
        self.o = np.zeros((cap, dim), np.float16)
        self.o2 = np.zeros((cap, dim), np.float16)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros((cap, N_STEP), np.float32)
        self.k = np.zeros(cap, np.int64)            # real steps in this sample (<= N_STEP)
        self.end = np.zeros(cap, np.int8)           # 0 = goes on, 1 = finished, 2 = failed (stuck/off-zone)
        self.c = np.zeros(cap, np.float32)          # time cost per step (charged for the rest of a failed race)

    def add(self, o, a, rs, k, o2, end, c):
        j = self.i
        self.o[j], self.a[j], self.o2[j], self.k[j], self.end[j], self.c[j] = o, a, o2, k, end, c
        self.r[j] = 0.0
        self.r[j, :k] = rs
        self.i, self.n = (j + 1) % self.cap, min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        return (self.o[idx].astype(np.float32), self.a[idx], self.r[idx], self.k[idx], self.o2[idx].astype(np.float32),
                self.end[idx], self.c[idx])


def act_index(k):
    return np.array([k // 3, k % 3])


def with_tau(obs, tau_frac):
    return np.concatenate([obs, np.full((len(obs), 1), tau_frac, np.float32)], axis=1).astype(np.float32)


def choose(net, obs, eps, beps, rng, seed_noise=None):
    """obs [B, dim] -> action indices; full mini-race ahead when acting."""
    with torch.no_grad():
        q = net.q(torch.from_numpy(with_tau(obs, 1.0))).numpy()
    out = q.argmax(1)
    for i in range(len(out)):
        u = rng.random()
        if u < eps:
            out[i] = rng.integers(q.shape[1])
        elif u < eps + beps:
            out[i] = int(np.argmax(q[i] + TAU_B * rng.gumbel(size=q.shape[1])))
    return out


# --- champion tests in their own process -----------------------------------------------------------
def test_worker(cfg, run_dir, n_in, n_act, inbox, busy):
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    best_file, events = run_dir / "best.json", run_dir / "events.jsonl"
    line_file, line_meta = run_dir / "centerline.npy", run_dir / "centerline.json"

    def log(**kw):
        kw["t"] = round(time.time(), 2)
        with open(events, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")

    env = None
    net = IQNNet(n_in, n_act)
    best, bad, tests = T.read_json(best_file), 0, 0
    while True:
        msg = inbox.get()
        if msg is None:
            break
        try:
            if env is None:
                env = T.TMEnv(cfg, run_dir, shorten=False, role="champion test")
            net.load_state_dict(msg["sd"])
            best_try = None
            for sd in [None] + [1000 + i for i in range(max(0, cfg["test_tries"] - 1))]:
                rng = np.random.default_rng(sd)
                obs, _ = env.reset()
                done, info = False, {}
                while not done:
                    a = choose(net, obs[None], 0.0 if sd is None else 0.02, 0.0, rng)[0]
                    obs, _, term, trunc, info = env.step(act_index(a))
                    done = term or trunc
                tr = {"info": info, "seed": sd, "hist": list(env.hist), "acts": list(env.acts), "rec": env.g.export()}
                key = (0, info["race_ms"]) if info["finished"] else (1, -info.get("cp", 0))
                if best_try is None or key < best_try[0]:
                    best_try = (key, tr)
            tr = best_try[1]
            info, fin, rec = tr["info"], bool(tr["info"]["finished"]), tr["rec"]
            tests += 1
            T.InputText.write(str(run_dir / "last_test.txt"), rec.actions)
            ms = int(rec.finish_ms) if fin else None
            best = T.read_json(best_file)
            champ = best.get("champion_ms")
            promoted = rolled = False
            if fin:
                best.setdefault("first_test_ms", ms)
            if fin and (champ is None or ms < champ):
                promoted, bad = True, 0
                best["champion_ms"], best["champion_seed"] = ms, tr["seed"]
                champ = ms
                torch.save(msg["sd"], run_dir / "champion.pt")
                T.InputText.write(str(run_dir / "champion.txt"), rec.actions)
                T.save_npy(run_dir / "champ_trace.npy", np.asarray(tr["hist"], dtype=np.float64))
            elif champ is not None and (not fin or ms > champ + cfg["rollback_margin_s"] * 1000):
                bad += 1
                if bad >= cfg["rollback_after"] and (run_dir / "champion.pt").exists():
                    (run_dir / "ROLLBACK").write_text("1")
                    rolled, bad = True, 0
            else:
                bad = 0
            ca = run_dir / "champion_actions.json"
            if fin and (promoted or not ca.exists()):
                T.write_json(ca, {"ms": ms, "acts": tr["acts"]})
            if fin and not (run_dir / "champ_trace.npy").exists():
                T.save_npy(run_dir / "champ_trace.npy", np.asarray(tr["hist"], dtype=np.float64))
            line_ms = T.read_json(line_meta).get("ms") or best.get("first_test_ms")
            refresh = (not env.guide and cfg["line_refresh_s"] and promoted and line_ms
                       and ms <= line_ms - cfg["line_refresh_s"] * 1000)
            if fin and (not line_file.exists() or refresh):
                T.save_npy(line_file, T.make_line(np.asarray(tr["hist"])[:, :3]))
                for f in ("profile.json", "profile.npy"):
                    (run_dir / f).unlink(missing_ok=True)
                env.hist = tr["hist"]
                env._load_line()
                env._offer_profile(ms)
                T.write_json(line_meta, {"ms": ms, "t": time.time()})
                T.say(f"Reference line + ghost from the {ms / 1000:.3f} s champion lap")
            T.write_json(best_file, best)
            log(k="test", lap_n=msg["lap_n"], ms=ms, champ_ms=champ, promoted=promoted, rolled_back=rolled)
            T.say(f"Champion test {tests}: " + (f"{ms / 1000:.3f} s" if fin else f"no finish ({info.get('end')})")
                  + (" -> NEW CHAMPION" if promoted else "") + (" -> rolling back to champion" if rolled else ""))
        except Exception as e:
            T.say(f"Champion test failed: {str(e)[:120]}")
            env = None
        finally:
            busy.value = 0
    if env is not None:
        env.close()


# --- training ---------------------------------------------------------------------------------------
def train(cfg):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    run_dir = cfgmod.run_dir(cfg)
    (run_dir / "STOP").unlink(missing_ok=True)
    (run_dir / "ROLLBACK").unlink(missing_ok=True)
    torch.set_num_threads(4)
    T.prepare_guide(cfg, run_dir)
    events, status_file = run_dir / "events.jsonl", run_dir / "status.json"

    def log(**kw):
        kw["t"] = round(time.time(), 2)
        with open(events, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")

    t0 = time.time()

    def status(state, reason=""):
        T.write_json(status_file, {"state": state, "reason": reason, "pid": os.getpid(), "heartbeat": time.time(),
                                   "started": t0})

    n_env = cfg["envs"]
    envs = SubprocVecEnv([partial(T._make_env, cfg, str(run_dir)) for _ in range(n_env)])
    dim = envs.observation_space.shape[0]
    n_act = int(envs.action_space.nvec[0] * envs.action_space.nvec[1])
    n_in = dim + 1
    H = max(N_STEP + 1, int(round(cfg["minirace_s"] * 100 / cfg["ticks_per_step"])))      # mini-race, in steps
    time_cost = cfg["w_time"] * cfg["ticks_per_step"] / 5.0

    net, tgt = IQNNet(n_in, n_act), IQNNet(n_in, n_act)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4)
    steps, trained, upd = 0, 0, 0
    ckpt = run_dir / "latest.pt"
    meta = T.read_json(run_dir / "iqn.json")
    if cfg["resume"] and ckpt.exists() and meta.get("n_in") == n_in and meta.get("n_act") == n_act:
        try:
            d = torch.load(ckpt, map_location="cpu")
            net.load_state_dict(d["net"])
            tgt.load_state_dict(d["tgt"])
            opt.load_state_dict(d["opt"])
            steps = int(d.get("steps", 0))
            trained = max(0, (steps - WARMUP)) * cfg["iqn_reuse"]
            T.say(f"Resumed IQN from latest.pt ({steps:,} steps)")
        except Exception as e:
            T.say(f"Cannot resume IQN ({str(e)[:80]}), starting fresh")
    else:
        tgt.load_state_dict(net.state_dict())
    T.write_json(run_dir / "iqn.json", {"n_in": n_in, "n_act": n_act})
    buf = Replay(cfg["iqn_buffer"], dim)

    ctx = mp.get_context("spawn")
    inbox, busy = ctx.Queue(), ctx.Value("i", 0)
    tester = ctx.Process(target=test_worker, args=(cfg, str(run_dir), n_in, n_act, inbox, busy), daemon=True)
    tester.start()

    def save():
        torch.save({"net": net.state_dict(), "tgt": tgt.state_dict(), "opt": opt.state_dict(), "steps": steps}, ckpt)

    rng = np.random.default_rng()
    obs = envs.reset()
    pend = [[] for _ in range(n_env)]          # n-step windows: (obs, action, reward)
    laps = since = 0
    rb, rb_steps, last_log, last_chk, losses, qs = {}, 0, time.time(), 0.0, [], []
    reason = "stopped"
    log(k="start", cfg=cfg)
    status("running")
    T.say(f"Training started: IQN, {n_env} cars, mini-race {cfg['minirace_s']:.0f} s, reuse x{cfg['iqn_reuse']}, CPU.")
    try:
        while True:
            T.watchdog()
            eps, beps = sched(EPS, steps), sched(BOLTZ, steps)
            a = choose(net, obs, eps, beps, rng)
            nobs, rew, dones, infos = envs.step(np.stack([act_index(x) for x in a]))
            steps += n_env
            for i in range(n_env):
                pend[i].append((obs[i], int(a[i]), float(rew[i])))
                if dones[i]:
                    info = infos[i]
                    last = info.get("terminal_observation", nobs[i])
                    trunc = bool(info.get("TimeLimit.truncated", False))
                    end = 0 if trunc else (1 if info.get("finished") else 2)
                    for j in range(len(pend[i])):                     # flush every window to the episode end
                        w = pend[i][j:j + N_STEP]
                        ends_here = j + N_STEP >= len(pend[i])
                        o2 = last if ends_here else pend[i][j + N_STEP][0]
                        buf.add(w[0][0], w[0][1], [x[2] for x in w], len(w), o2, end if ends_here else 0, time_cost)
                    pend[i] = []
                    laps += 1
                    since += 1
                    fin = bool(info.get("finished"))
                    log(k="lap", ms=int(info["race_ms"]) if fin else None, cp=int(info.get("cp", 0)),
                        end=info.get("end", ""), mid=1 if info.get("mid") else 0, x=info.get("x"), z=info.get("z"))
                    for kk, vv in (info.get("rb") or {}).items():
                        rb[kk] = rb.get(kk, 0.0) + vv
                    rb_steps += int(info.get("rsteps", 0))
                elif len(pend[i]) > N_STEP:
                    w = pend[i][:N_STEP]
                    buf.add(w[0][0], w[0][1], [x[2] for x in w], N_STEP, pend[i][N_STEP][0], 0, time_cost)
                    pend[i].pop(0)
            obs = nobs

            # --- learning: keep (experiences learned from) ~= reuse x (experiences collected) ---
            gamma = sched(GAMMA, steps)
            for g in opt.param_groups:
                g["lr"] = sched(LR, steps)
            while buf.n >= WARMUP and trained < (steps - WARMUP) * cfg["iqn_reuse"]:
                o, ac, r, k, o2, end, c = buf.sample(BATCH)
                tau = np.random.randint(1, H + 1, BATCH)                  # time left in this sample's mini-race
                m = np.minimum(k, tau)
                disc = gamma ** np.arange(N_STEP)
                mask = np.arange(N_STEP)[None, :] < m[:, None]
                R = (r * disc * mask).sum(1)
                # a failed race keeps paying the time cost for the rest of its mini-race (giving up never pays)
                R -= np.where(end == 2, c * np.maximum(tau - m, 0), 0.0)
                boot = (end == 0) & (tau > k)
                x = torch.from_numpy(np.concatenate([o, (tau / H)[:, None]], 1).astype(np.float32))
                x2 = torch.from_numpy(np.concatenate([o2, (np.maximum(tau - k, 0) / H)[:, None]], 1).astype(np.float32))
                with torch.no_grad():
                    a2 = tgt.q(x2, N_Q).argmax(1)
                    z2 = tgt(x2, torch.rand(BATCH, N_Q)).gather(2, a2[:, None, None].expand(-1, N_Q, 1)).squeeze(2)
                    tgt_z = (torch.from_numpy(R.astype(np.float32))[:, None]
                             + torch.from_numpy((gamma ** k * boot).astype(np.float32))[:, None] * z2)
                taus = torch.rand(BATCH, N_Q)
                z = net(x, taus).gather(2, torch.from_numpy(ac)[:, None, None].expand(-1, N_Q, 1)).squeeze(2)
                td = tgt_z[:, None, :] - z[:, :, None]
                hub = torch.where(td.abs() <= KAPPA, 0.5 * td ** 2 / KAPPA, td.abs() - 0.5 * KAPPA)
                loss = ((taus[:, :, None] - (td.detach() < 0).float()).abs() * hub).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 30.0)
                opt.step()
                trained += BATCH
                upd += 1
                losses.append(loss.item())
                qs.append(float(z.mean()))
                if upd % 4 == 0:                                          # soft target update
                    with torch.no_grad():
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(0.98).add_(p, alpha=0.02)

            # --- tests, rollback, logging, stop ---
            if since >= cfg["test_every_laps"] and not busy.value:
                busy.value = 1
                since = 0
                inbox.put({"sd": {k: v.clone() for k, v in net.state_dict().items()}, "lap_n": laps})
            if (run_dir / "ROLLBACK").exists():
                try:
                    sd = torch.load(run_dir / "champion.pt", map_location="cpu")
                    net.load_state_dict(sd)
                    tgt.load_state_dict(sd)
                except Exception as e:
                    T.say(f"Rollback failed ({str(e)[:60]})")
                (run_dir / "ROLLBACK").unlink(missing_ok=True)
            now = time.time()
            if now - last_log > 20 and losses:
                log(k="upd", u=upd, steps=steps, v=float(np.mean(qs)), vl=float(np.mean(losses)), ent=eps, ev=0.0,
                    speed=steps * cfg["ticks_per_step"] / 100.0 / max(now - t0, 1e-6))
                if rb_steps:
                    log(k="rb", r={kk: round(vv / rb_steps, 5) for kk, vv in rb.items()})
                rb, rb_steps, losses, qs, last_log = {}, 0, [], [], now
                save()
            if now - last_chk > 2.0:
                last_chk = now
                if (run_dir / "STOP").exists():
                    reason = "stopped by user"
                    break
                if cfg["time_limit_min"] and now - t0 > cfg["time_limit_min"] * 60:
                    reason = "time limit reached"
                    break
                champ, pb = T.read_json(run_dir / "best.json").get("champion_ms"), cfg["pb_seconds"]
                if cfg["stop_at_pb"] and pb and champ and champ <= pb * 1000:
                    reason = "reached your PB"
                    break
                status("running")
    except KeyboardInterrupt:
        reason = "stopped by user"
    except Exception as e:
        reason = f"crashed: {str(e)[:120]}"
        raise
    finally:
        save()
        log(k="stop", reason=reason)
        status("stopped", reason)
        T.say("Stopped: " + reason)
        try:
            inbox.put(None)
            tester.join(timeout=20)
        except Exception:
            pass
        envs.close()


def drive(cfg):
    """Play champion for IQN: the same best-of-N lap the champion test kept (same weights + same seed)."""
    run_dir = cfgmod.run_dir(cfg)
    meta = T.read_json(run_dir / "iqn.json")
    net = IQNNet(meta["n_in"], meta["n_act"])
    net.load_state_dict(torch.load(run_dir / "champion.pt", map_location="cpu"))
    env = T.TMEnv(cfg, run_dir, shorten=False, role="play")
    sd = T.read_json(run_dir / "best.json").get("champion_seed")
    rng = np.random.default_rng(sd)
    T.say("Driving the champion (simulated in the background; load play.txt in TICK to watch) ...")
    obs, _ = env.reset()
    done, info = False, {}
    while not done:
        a = choose(net, obs[None], 0.0 if sd is None else 0.02, 0.0, rng)[0]
        obs, _, term, trunc, info = env.step(act_index(a))
        done = term or trunc
    res = {"t": time.time(), "finished": bool(info["finished"]), "end": info["end"], "ms": None, "file": None}
    if info["finished"]:
        rec = env.g.export()
        T.InputText.write(str(run_dir / "play.txt"), rec.actions)
        res.update(ms=int(rec.finish_ms), file=str(run_dir / "play.txt"))
        T.say(f"Champion play finished in {rec.finish_ms / 1000:.3f} s -> {run_dir / 'play.txt'}")
    else:
        T.say(f"Champion play: no finish ({info['end']})")
    T.write_json(run_dir / "play_result.json", res)
    env.close()
