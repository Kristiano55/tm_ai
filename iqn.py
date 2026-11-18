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
        return self(x, torch.rand(x.shape[0], k, device=x.device)).mean(1)


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
def test_worker(cfg, run_dir, n_in, n_act, inbox, busy, tstat=None):
    torch.set_num_threads(1)
    T.raise_priority()
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
        t_test = time.time()
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
                if cfg["iqn_rollback"] and bad >= cfg["rollback_after"] and (run_dir / "champion.pt").exists():
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
            if tstat is not None:
                tstat[0] += 1
                tstat[1] += time.time() - t_test
    if env is not None:
        env.close()


# --- training ---------------------------------------------------------------------------------------
def eps_points(cfg):
    """Random-action schedule from the settings: 1.0 for the first steps, then mid, then end."""
    f, m, e = cfg["iqn_eps_full_steps"], max(cfg["iqn_eps_mid_steps"], cfg["iqn_eps_full_steps"] + 1), cfg["iqn_eps_end_steps"]
    return [(0, 1.0), (f, 1.0), (m, cfg["iqn_eps_mid"]), (max(e, m + 1), cfg["iqn_eps_end"])]


PUBLISH_EVERY = 20        # learner updates between weight copies to the driving network (Linesight-style)
PROF_EVERY = float(os.environ.get("TMAI_PROF_S", 300))     # profiling summary every 5 min


def pick_device(cfg):
    want = cfg["iqn_device"]
    if want == "cpu":
        return "cpu"
    ok = torch.cuda.is_available()
    if want == "cuda" and not ok:
        T.say("IQN: 'cuda' was chosen but PyTorch sees no GPU (CPU-only PyTorch?); learning on the CPU")
    return "cuda" if ok else "cpu"


# --- replay memory in shared memory (the cars' process writes, the learner process samples) -------------
def _attach(name):
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, track=False)            # Python 3.13+
    except TypeError:
        shm = shared_memory.SharedMemory(name=name)
        if os.name != "nt":                      # POSIX: don't let this process's exit delete the block
            try:
                from multiprocessing import resource_tracker
                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        return shm


class SharedReplay(Replay):
    def __init__(self, cap, dim, names=None):
        from multiprocessing import shared_memory
        fields = [("o", (cap, dim), np.float16), ("o2", (cap, dim), np.float16), ("a", (cap,), np.int64),
                  ("r", (cap, N_STEP), np.float32), ("k", (cap,), np.int64), ("end", (cap,), np.int8),
                  ("c", (cap,), np.float32), ("ctr", (2,), np.int64)]
        self.cap, self.shms, self.names, self.owner = cap, [], {}, names is None
        for f, shape, dt in fields:
            size = max(1, int(np.prod(shape)) * np.dtype(dt).itemsize)
            shm = shared_memory.SharedMemory(create=True, size=size) if self.owner else _attach(names[f])
            self.shms.append(shm)
            self.names[f] = shm.name
            setattr(self, f, np.ndarray(shape, dt, buffer=shm.buf))
        if self.owner:
            self.ctr[:] = 0

    i = property(lambda self: int(self.ctr[0]), lambda self, v: self.ctr.__setitem__(0, v))
    n = property(lambda self: int(self.ctr[1]), lambda self, v: self.ctr.__setitem__(1, v))

    def close(self):
        for f in ("o", "o2", "a", "r", "k", "end", "c", "ctr"):
            setattr(self, f, None)
        for shm in self.shms:
            try:
                shm.close()
                if self.owner:
                    shm.unlink()
            except Exception:
                pass


def flat_size(net):
    return sum(v.numel() for v in net.state_dict().values())


def sd_to_flat(sd, out):
    np.copyto(out, torch.cat([v.detach().reshape(-1).float().cpu() for v in sd.values()]).numpy())


def flat_to_sd(net, flat):
    sd, i = {}, 0
    for k, v in net.state_dict().items():
        sd[k] = torch.from_numpy(np.array(flat[i:i + v.numel()], np.float32)).view_as(v).to(v.dtype)
        i += v.numel()
    return sd


# --- learner process ----------------------------------------------------------------------------------
# ctr (int64): 0 steps driven (main writes), 1 samples learned, 2 updates, 3 weights version
# stats (double): 0 loss sum, 1 q sum, 2 count, 3 busy seconds
def learner_proc(cfg, run_dir, dim, n_act, H, time_cost, rnames, rlock, ctr, stats, wbuf, wlock, stop_evt, ready_evt,
                 errq):
    import traceback
    T.raise_priority()
    run_dir = Path(run_dir)
    try:
        dev = pick_device(cfg)
        torch.set_num_threads(2)
        n_in = dim + 1
        net, tgt = IQNNet(n_in, n_act).to(dev), IQNNet(n_in, n_act).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4)
        ckpt = run_dir / "latest.pt"
        meta = T.read_json(run_dir / "iqn.json")
        steps0 = 0
        if cfg["resume"] and ckpt.exists() and meta and (meta.get("n_in"), meta.get("n_act")) != (n_in, n_act):
            ckpt.replace(run_dir / "latest.old_layout.pt")      # never load under a wrong layout, never lose it
            T.say("IQN: the input layout changed; old latest.pt moved to latest.old_layout.pt, starting fresh")
        if cfg["resume"] and ckpt.exists() and meta.get("n_in") == n_in and meta.get("n_act") == n_act:
            try:
                d = torch.load(ckpt, map_location=dev)
                net.load_state_dict(d["net"])
                tgt.load_state_dict(d["tgt"])
                opt.load_state_dict(d["opt"])
                steps0 = int(d.get("steps", 0))
                T.say(f"Resumed IQN from latest.pt ({steps0:,} steps)")
            except Exception as e:
                T.say(f"Cannot resume IQN ({str(e)[:80]}), starting fresh")
                net, tgt = IQNNet(n_in, n_act).to(dev), IQNNet(n_in, n_act).to(dev)
                opt = torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4)
                tgt.load_state_dict(net.state_dict())
        elif (cfg["resume"] and (run_dir / "champion.pt").exists() and meta.get("n_in") == n_in
              and meta.get("n_act") == n_act):
            try:                                            # latest.pt missing: continue from the champion
                sd = torch.load(run_dir / "champion.pt", map_location=dev)
                net.load_state_dict(sd)
                tgt.load_state_dict(sd)
                T.say("!" * 12 + " Resumed IQN from champion.pt because latest.pt is missing " + "!" * 12)
            except Exception as e:
                T.say("!" * 12 + f" STARTING FROM SCRATCH because champion.pt could not be loaded ({str(e)[:80]}) "
                      + "!" * 12)
                tgt.load_state_dict(net.state_dict())
        else:
            if not cfg["resume"] and (ckpt.exists() or (run_dir / "champion.pt").exists()):
                T.say("!" * 12 + " STARTING FROM SCRATCH because 'Resume from latest checkpoint' is off " + "!" * 12)
            tgt.load_state_dict(net.state_dict())
        T.write_json(run_dir / "iqn.json", {"n_in": n_in, "n_act": n_act})
        ctr[0], ctr[1], ctr[2] = steps0, max(0, steps0 - WARMUP) * cfg["iqn_reuse"], 0
        buf = SharedReplay(cfg["iqn_buffer"], dim, names=rnames)
        wnp = np.frombuffer(wbuf, dtype=np.float32)

        def publish():
            with wlock:
                sd_to_flat(net.state_dict(), wnp)
                ctr[3] += 1

        def save():
            torch.save({"net": net.state_dict(), "tgt": tgt.state_dict(), "opt": opt.state_dict(),
                        "steps": int(ctr[0])}, ckpt)

        publish()
        ready_evt.set()
        last_save = last_chk = time.time()
        while not stop_evt.is_set():
            now = time.time()
            if now - last_chk > 2.0:
                last_chk = now
                if (run_dir / "ROLLBACK").exists():
                    try:
                        sd = torch.load(run_dir / "champion.pt", map_location=dev)
                        net.load_state_dict(sd)
                        tgt.load_state_dict(sd)
                        publish()
                    except Exception as e:
                        T.say(f"Rollback failed ({str(e)[:60]})")
                    (run_dir / "ROLLBACK").unlink(missing_ok=True)
            if now - last_save > 20:
                last_save = now
                save()
            steps = int(ctr[0])
            if buf.n < WARMUP or ctr[1] >= (steps - WARMUP) * cfg["iqn_reuse"]:
                time.sleep(0.002)                         # ahead of the target: wait for more driving
                continue
            b0 = time.perf_counter()
            gamma = sched(GAMMA, steps)
            with rlock:
                o, ac, r, k, o2, end, c = buf.sample(BATCH)
            tau = np.random.randint(1, H + 1, BATCH)                  # time left in this sample's mini-race
            m = np.minimum(k, tau)
            disc = gamma ** np.arange(N_STEP)
            mask = np.arange(N_STEP)[None, :] < m[:, None]
            R = (r * disc * mask).sum(1)
            # a failed race keeps paying the time cost for the rest of its mini-race (giving up never pays)
            R -= np.where(end == 2, c * np.maximum(tau - m, 0), 0.0)
            boot = (end == 0) & (tau > k)
            x = torch.from_numpy(np.concatenate([o, (tau / H)[:, None]], 1).astype(np.float32)).to(dev)
            x2 = torch.from_numpy(np.concatenate([o2, (np.maximum(tau - k, 0) / H)[:, None]], 1).astype(np.float32)).to(dev)
            for g in opt.param_groups:
                g["lr"] = sched(LR, steps)
            with torch.no_grad():
                zt = tgt(x2, torch.rand(BATCH, N_Q, device=dev))            # one target pass: pick + value
                a2 = zt.mean(1).argmax(1)
                z2 = zt.gather(2, a2[:, None, None].expand(-1, N_Q, 1)).squeeze(2)
                tgt_z = (torch.from_numpy(R.astype(np.float32)).to(dev)[:, None]
                         + torch.from_numpy((gamma ** k * boot).astype(np.float32)).to(dev)[:, None] * z2)
            taus = torch.rand(BATCH, N_Q, device=dev)
            z = net(x, taus).gather(2, torch.from_numpy(ac).to(dev)[:, None, None].expand(-1, N_Q, 1)).squeeze(2)
            td = tgt_z[:, None, :] - z[:, :, None]
            hub = torch.where(td.abs() <= KAPPA, 0.5 * td ** 2 / KAPPA, td.abs() - 0.5 * KAPPA)
            loss = ((taus[:, :, None] - (td.detach() < 0).float()).abs() * hub).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 30.0)
            opt.step()
            ctr[1] += BATCH
            ctr[2] += 1
            if ctr[2] % 4 == 0:                                        # soft target update
                with torch.no_grad():
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(0.98).add_(p, alpha=0.02)
            if ctr[2] % PUBLISH_EVERY == 0:                            # fresh weights for the cars
                publish()
            lv, qv = loss.item(), float(z.detach().mean())
            with wlock:
                stats[0] += lv
                stats[1] += qv
                stats[2] += 1
                stats[3] += time.perf_counter() - b0
        save()
        buf.close()
    except Exception:
        errq.put(traceback.format_exc()[-1500:])
        ready_evt.set()


# --- training (driving loop) ----------------------------------------------------------------------------
def train(cfg):
    run_dir = cfgmod.run_dir(cfg)
    (run_dir / "STOP").unlink(missing_ok=True)
    (run_dir / "ROLLBACK").unlink(missing_ok=True)
    torch.set_num_threads(1)                                  # driving: tiny batches, one thread is fastest
    T.raise_priority(tick_too=True)
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
    if n_env + 3 > (os.cpu_count() or 1):
        T.say(f"Warning: {n_env} cars + learner + tester need ~{n_env + 3} CPU threads, this PC has "
              f"{os.cpu_count()}: fewer cars may run faster (try 'python tm_ai.py bench').")
    pool = T.CarPool(cfg, run_dir, n_env)                     # cars in their own processes, not in lock-step
    dim, nvec = pool.dim, pool.nvec
    n_act = int(nvec[0] * nvec[1])
    n_in = dim + 1
    H = max(N_STEP + 1, int(round(cfg["minirace_s"] * 100 / cfg["ticks_per_step"])))      # mini-race, in steps
    time_cost = cfg["w_time"] * cfg["ticks_per_step"] / 5.0
    EPS_PTS = eps_points(cfg)

    ctx = mp.get_context("spawn")
    buf = SharedReplay(cfg["iqn_buffer"], dim)
    act = IQNNet(n_in, n_act)
    rlock, wlock = ctx.Lock(), ctx.Lock()
    ctr, stats = ctx.RawArray("q", 4), ctx.RawArray("d", 4)
    wbuf = ctx.RawArray("f", flat_size(act))
    wnp = np.frombuffer(wbuf, dtype=np.float32)
    stop_evt, ready_evt, errq = ctx.Event(), ctx.Event(), ctx.Queue()
    device = pick_device(cfg)
    learner = ctx.Process(target=learner_proc, daemon=True,
                          args=(cfg, str(run_dir), dim, n_act, H, time_cost, buf.names, rlock, ctr, stats, wbuf, wlock,
                                stop_evt, ready_evt, errq))
    learner.start()
    while not ready_evt.wait(1.0):
        if not learner.is_alive():
            break
    if not errq.empty() or not learner.is_alive():
        msg = errq.get() if not errq.empty() else "learner process ended"
        pool.close()
        buf.close()
        raise RuntimeError("IQN learner could not start:\n" + msg)
    wver = 0

    def refresh_act():
        nonlocal wver
        if ctr[3] != wver:
            with wlock:
                wver = int(ctr[3])
                flat = wnp.copy()
            act.load_state_dict(flat_to_sd(act, flat))

    refresh_act()
    steps0 = int(ctr[0])                    # dashboard speed: steps driven in this session only

    tester = inbox = busy = None
    tstat = ctx.RawArray("d", 2)            # tests done, seconds spent

    def start_tester():
        nonlocal tester, inbox, busy
        inbox, busy = ctx.Queue(), ctx.Value("i", 0)
        tester = ctx.Process(target=test_worker, args=(cfg, str(run_dir), n_in, n_act, inbox, busy, tstat),
                             daemon=True)
        tester.start()

    start_tester()

    rng = np.random.default_rng()
    pend = [[] for _ in range(n_env)]          # n-step windows per car: (obs, action, reward)
    C = {"laps": 0, "since": 0}
    rb, rb_steps = {}, 0
    stuckw = T.StuckWatch()
    tracker = T.SectionTracker(run_dir, cfg, T.n_sections(run_dir, pool.n_seg))
    ramp = T.HitRamp(run_dir, cfg)

    def collect(i, o, a, nobs, rew, done, info, rows):
        """One step of car i -> n-step samples (rows to add); log a finished lap."""
        nonlocal rb, rb_steps
        if info.get("end") == "car restarted":              # hung car replaced: drop its unfinished window
            pend[i] = []
            return
        pend[i].append((o, int(a), float(rew)))
        if done:
            last = info.get("terminal_observation", nobs)
            trunc = bool(info.get("TimeLimit.truncated", False))
            end = 0 if trunc else (1 if info.get("finished") else 2)
            for j in range(len(pend[i])):                     # flush every window to the episode end
                w = pend[i][j:j + N_STEP]
                ends_here = j + N_STEP >= len(pend[i])
                o2 = last if ends_here else pend[i][j + N_STEP][0]
                rows.append((w[0][0], w[0][1], [x[2] for x in w], len(w), o2, end if ends_here else 0, time_cost))
            pend[i] = []
            C["laps"] += 1
            C["since"] += 1
            fin = bool(info.get("finished"))
            log(k="lap", ms=int(info["race_ms"]) if fin else None, cp=int(info.get("cp", 0)),
                end=info.get("end", ""), mid=1 if info.get("mid") else 0, x=info.get("x"), z=info.get("z"))
            for kk, vv in (info.get("rb") or {}).items():
                rb[kk] = rb.get(kk, 0.0) + vv
            rb_steps += int(info.get("rsteps", 0))
            stuckw.add(info)
            tracker.add(info)
            ramp.add(info)
        elif len(pend[i]) > N_STEP:
            w = pend[i][:N_STEP]
            rows.append((w[0][0], w[0][1], [x[2] for x in w], N_STEP, pend[i][N_STEP][0], 0, time_cost))
            pend[i].pop(0)

    # optional warm start: the first ~50k steps are driven by a PPO run's champion
    ppo, warm = None, cfg["iqn_warm_run"]
    if warm and int(ctr[0]) == 0:
        src = cfgmod.ROOT / "runs" / warm / "champion.zip"
        try:
            from stable_baselines3 import PPO
            if not src.exists():
                raise FileNotFoundError(f"no champion.zip in run '{warm}'")
            ppo = PPO.load(str(src), device="cpu")
            if ppo.observation_space.shape != (dim,):
                raise ValueError("that run uses a different input layout")
            T.say(f"Warm start: driving ~50k steps with the champion of PPO run '{warm}' into the replay memory ...")
        except Exception as e:
            ppo = None
            T.say(f"Warm start skipped ({str(e)[:100]})")

    def policy(obs_b, steps):
        nonlocal ppo
        if ppo is not None:
            if steps >= 50000:
                ppo = None
                T.say(f"Warm start done: {buf.n:,} experiences from the PPO champion.")
            else:
                pa, _ = ppo.predict(obs_b, deterministic=True)
                rnd = rng.random(len(pa)) < 0.1
                pa[rnd] = np.stack([rng.integers(nvec[0], size=rnd.sum()), rng.integers(nvec[1], size=rnd.sum())], 1)
                return pa[:, 0] * nvec[1] + pa[:, 1]
        return choose(act, obs_b, sched(EPS_PTS, steps), sched(BOLTZ, steps), rng)

    cur_obs, cur_act = list(pool.first_obs), [0] * n_env
    reason = "stopped"
    log(k="start", cfg=cfg)
    status("running")
    T.say(f"Training started: IQN, {n_env} cars (each drives on its own, no waiting), mini-race "
          f"{cfg['minirace_s']:.0f} s, reuse x{cfg['iqn_reuse']}, learner on {device.upper()} in its own process.")
    a0 = policy(np.stack(cur_obs), int(ctr[0]))
    for i in range(n_env):
        cur_act[i] = int(a0[i])
        pool.send(i, act_index(cur_act[i]))
    last_log = last_chk = last_prof = time.time()
    T.freeze_guard(run_dir)
    mt = {"act": 0.0, "collect": 0.0, "wait": 0.0}
    ctr_p, st_p, ts_p = np.array(ctr[:]), np.array(stats[:]), np.array(tstat[:])
    try:
        while True:
            T.watchdog()
            w0 = time.perf_counter()
            items = pool.poll(0.5) + pool.hung()
            w1 = time.perf_counter()
            rows = []
            for i, nobs, r, done, info in items:
                collect(i, cur_obs[i], cur_act[i], nobs, r, done, info, rows)
                cur_obs[i] = nobs
            if rows:
                with rlock:
                    for row in rows:
                        buf.add(*row)
            ctr[0] += len(items)
            w2 = time.perf_counter()
            if items:
                refresh_act()
                ids = [it[0] for it in items]
                a = policy(np.stack([cur_obs[i] for i in ids]), int(ctr[0]))
                for i, ai in zip(ids, a):
                    cur_act[i] = int(ai)
                    pool.send(i, act_index(cur_act[i]))
            w3 = time.perf_counter()
            mt["wait"] += w1 - w0
            mt["collect"] += w2 - w1
            mt["act"] += w3 - w2

            # --- tests, logging, profiling, stop ---
            if C["since"] >= cfg["test_every_laps"] and not busy.value and (run_dir / "best_run.json").exists():
                busy.value = 1
                C["since"] = 0
                inbox.put({"sd": {k: v.clone() for k, v in act.state_dict().items()}, "lap_n": C["laps"]})
            now = time.time()
            if now - last_log > 20 and stats[2] > 0:
                with wlock:
                    lm, qm = stats[0] / stats[2], stats[1] / stats[2]
                    stats[0] = stats[1] = stats[2] = 0.0
                log(k="upd", u=int(ctr[2]), steps=int(ctr[0]), v=qm, vl=lm, ent=sched(EPS_PTS, int(ctr[0])), ev=0.0,
                    speed=(int(ctr[0]) - steps0) * cfg["ticks_per_step"] / 100.0 / max(now - t0, 1e-6))
                if rb_steps:
                    log(k="rb", r={kk: round(vv / rb_steps, 5) for kk, vv in rb.items()})
                rb, rb_steps, last_log = {}, 0, now
            if now - last_prof > PROF_EVERY:
                wall = now - last_prof
                c, s_, ts = np.array(ctr[:]), np.array(stats[:]), np.array(tstat[:])
                dsteps = max(1, c[0] - ctr_p[0])
                lrn = {"ups": round((c[2] - ctr_p[2]) / wall, 1), "busy": round(100 * (s_[3] - st_p[3]) / wall),
                       "reuse": round((c[1] - ctr_p[1]) / dsteps, 1), "device": device}
                tst = {"n": int(ts[0] - ts_p[0]), "avg_s": round((ts[1] - ts_p[1]) / max(1, ts[0] - ts_p[0]), 1)}
                car = pool.take_prof()
                rec = T.prof_summary(car, wall, main={k: v for k, v in mt.items()}, learner=lrn, tester=tst,
                                     ticks=cfg["ticks_per_step"])
                rec["cars"] = n_env
                log(**rec)
                T.say(T.prof_line(rec))
                mt = {k: 0.0 for k in mt}
                ctr_p, st_p, ts_p, last_prof = c, s_, ts, now
            if now - last_chk > 2.0:
                last_chk = now
                ramp.update(log)
                try:
                    tracker.update(log)
                except Exception as e:
                    T.say(f"Section tracking error: {str(e)[:100]}")
                if not learner.is_alive():
                    raise RuntimeError("IQN learner stopped: " + (errq.get() if not errq.empty() else "unknown"))
                if not tester.is_alive():                             # tester died: start a new one
                    T.say("Champion tester stopped unexpectedly, restarting it")
                    start_tester()
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
        stop_evt.set()                                    # learner saves latest.pt and exits
        learner.join(timeout=60)
        if learner.is_alive():
            learner.terminate()
        log(k="stop", reason=reason)
        status("stopped", reason)
        T.say("Stopped: " + reason)
        try:
            inbox.put(None)
            tester.join(timeout=20)
        except Exception:
            pass
        pool.close()
        buf.close()


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
