"""Shared settings. The dashboard edits config.json, tm_ai.py reads it."""
import json
import os as _os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(_os.environ["TMAI_CONFIG"]) if _os.environ.get("TMAI_CONFIG") else ROOT / "config.json"   # diagnose trials

DEFAULTS = {
    "map": "",                 # full path to the .Map.Gbx
    "run": "calm",             # run name -> runs/<run>/
    "envs": max(1, min(8, (_os.cpu_count() or 12) - 4)),   # parallel cars: 8 on 12 threads (+ learner, tester)
    "ticks_per_step": 5,       # 5 ticks = 50 ms per decision
    "max_lap_seconds": 90,     # hard time limit per training lap
    "pb_seconds": 0.0,         # your personal best (0 = unset)
    "stop_at_pb": False,       # stop training once the champion beats the PB
    "time_limit_min": 0,       # stop after N minutes (0 = until stopped)
    "discord_webhook": "",     # Discord channel webhook URL: training updates on your phone (empty = off)
    "discord_every_min": 15,   # minutes between Discord summaries while training (new champions post at once)
    "test_every_laps": 50,     # champion test after this many training laps
    "rollback_after": 3,       # roll back to the champion after N bad tests in a row
    "rollback_margin_s": 1.0,  # a test counts as bad if slower than champion by this much
    "resume": True,            # continue from runs/<run>/latest.zip if it fits
    "extra_obs": True,         # extra observations: effects/runtime telemetry, ghost speed, line offset ...
    "steer_mode": "progressive",   # "progressive" (fine near centre, coarse at the edges) or "uniform" (10% steps)
    "cut_behind_s": 5.0,
    "ghost_window_s": 2.0,
    "algo": "ppo",             # "ppo" or "iqn" (Linesight-style: replay buffer, distributional Q, mini-races)
    "minirace_s": 7.0,         # IQN: the AI maximises progress over the next N seconds (Linesight: 7)
    "iqn_reuse": 16,           # IQN: how often each experience is learned from on average
    "iqn_buffer": 300000,      # IQN: replay memory size (experiences)
    "iqn_eps_full_steps": 10000,    # IQN: fully random driving for the first N steps
    "iqn_eps_mid": 0.1,             # IQN: random-action share reached at iqn_eps_mid_steps
    "iqn_eps_mid_steps": 100000,
    "iqn_eps_end": 0.03,            # IQN: final random-action share, reached at iqn_eps_end_steps
    "iqn_eps_end_steps": 1000000,
    "iqn_mid_start_p": 0.2,         # IQN: share of laps that start along your PB/WR (after the first finish)
    "iqn_rollback": False,          # IQN: roll back to the champion after bad tests (PPO always can)
    "iqn_device": "auto",           # IQN learner: "auto" (GPU if PyTorch sees one), "cpu" or "cuda"; driving stays on CPU
    "lidar_every": 1,               # 1 = lidar scan every step; 2 = every other step (reuses the last scan)
    "iqn_warm_run": "",
    "sec_fallback": True,           # with a PB/WR target line: sections where most laps fail use the AI's own route             # IQN: fill the replay memory with laps of this PPO run's champion first (empty = off)
    "test_tries_close": 8,     # ... and this many once the champion is within test_close_s of your PB
    "test_close_s": 0.5,
    "test_tries": 3,           # champion test = best of N laps (most-likely actions + seeded ones)
    "horizon_s": 20.0,         # how far ahead (seconds) the AI weighs rewards     # ghost reward only counts within this many seconds behind the ghost (live)       # end a lap that falls this far behind the ghost (0 = off)
    "decay_steps": 4000000,    # learning rate + exploration halve every N env steps (0 = off)
    "line_refresh_s": 0.2,
    "pb_file": "",             # your PB as a TICK input text file: traced once into a target line + timing (empty = off)
    "mid_start_p": 0.25,       # share of training laps that start partway along the champion's lap (0 = off)
    "mid_start_every_s": 2.0,  # spacing of those start points along the champion lap, seconds     # rebuild the reference line + ghost when the champion beats it by this much (0 = never)
    "run_type": "map",         # "map" = one map (run folder runs/<run>), "base" = shared driver over map_pool
    "base_name": "base",       # base runs live in runs/_base/<base_name>/ (one sub-folder per pool map)
    "map_pool": [],            # [{"map": path, "pb_file": path or "", "pb_seconds": 0}, ...]
    "pool_rotate_laps": 20,    # base: a car moves to another pool map every N laps (if more maps than cars)
    "progress_norm_m": 1000,   # base (and runs started from a base): progress pays as if every lap were this long
    "base_from": "",           # map run: start from this base model (name in runs/_base); empty = off
    "pool_cars": 1,            # map run with a map pool: cars that keep driving pool maps (no forgetting); 0 = off
    "hub_index_url": "https://kristiano55.github.io/tm_ai-brains/index.json",   # public brain hub index
    "hub_benchmark_url": "",   # empty = benchmark.json next to the index
    "hub_repo": "Kristiano55/tm_ai-brains",
    "github_user": "",         # your GitHub name (pack author / verifier)
    "github_token": "",        # optional, for direct release uploads; empty = the browser does it (nothing stored)
    "maps_folder": "",         # where your .Map.Gbx files are (verify / benchmark find maps there by file name)
    "iqn_gas_brake": False,    # IQN: 4th throttle mode gas + brake (new IQN runs only: IQN checkpoints don't convert)
    "iqn_steer_levels": 21,    # IQN: 21 / 31 / 41 steering levels (new IQN runs only)
    "own_line_s": 2.0,         # with a PB file: once the champion is within N s of the PB, follow its own line (0 = never)
    "obs_norm": True,          # PPO + IQN: inputs normalised by running mean/std (clipped +-5), saved with the run
    "iqn_per": True,           # IQN: prioritized replay (alpha 0.6, beta 0.4 -> 1); new IQN runs only
    "start_focus": 0.5,        # mid-track starts: share picked by where laps fail / lose time (rest uniform)
    "analog_steer": True,      # PPO: steering is one analog value -127..127 instead of 21 levels (IQN keeps levels)
    "gas_brake": True,         # PPO: 4th throttle mode = gas + brake held together (IQN keeps 3 modes)
    "sil_weight": 0.1,         # PPO: self-imitation from the AI's own top-20 laps (0 = off)
    "aux_weight": 0.05,        # PPO: auxiliary heads (wall hit soon, time to next CP, wheels grounded), training only
    "w_hit_ramp": True,        # wall-hit penalty ramps in with the finish rate (25 % below 30 % finished, 100 % from 60 %)
    "reward_v": 5,             # reward layout version (old configs are upgraded once)
}


# Reward weights. ALL_TERMS = every term the reward code still knows (old runs may use any of them, frozen in their
# rewards.json); WEIGHTS = the ones in Settings (key -> (default, label, hint)), read live by the cars.
# Defaults = the minimal set: progress, time, checkpoint, finish, give-up cost (stuck / off-zone + the remaining
# metres); ghost and wall hit are the only optional extras (off by default).
_TERMS = {
    "w_progress": "Progress|reward per metre driven along the route",
    "w_time": "Time penalty|cost per 50 ms step; this is what makes going fast pay",
    "w_checkpoint": "Checkpoint bonus|flat reward per checkpoint",
    "w_finish": "Finish bonus|flat reward for finishing",
    "w_stuck": "Stuck penalty|when the car stands still for 2 s (plus the progress it gave up)",
    "w_offzone": "Off-zone penalty|when the car leaves the playable area (plus the progress it gave up)",
    "ghost_weight": "Ghost|optional: reward per second gained on the champion's ghost (try 10; 0 = off)",
    "w_hit": "Wall-hit penalty|optional: per wall hit, scaled by the speed it cost (try 0.3; 0 = off)",
}
WEIGHTS = {k: (d, *_TERMS[k].split("|")) for k, d in (
    ("w_progress", 0.05), ("w_time", 0.02), ("w_checkpoint", 3.0), ("w_finish", 50.0), ("w_stuck", 20.0),
    ("w_offzone", 20.0), ("ghost_weight", 0.0), ("w_hit", 0.0))}
DEFAULTS.update({k: v[0] for k, v in WEIGHTS.items()})

# Every term with its default BEFORE reward_v 5 (the old "Lean" defaults): what an old config meant for a key it
# didn't have, frozen into existing runs' rewards.json by the v5 migration. Legacy-only terms have no Settings row.
OLD_DEFAULTS = dict(w_progress=0.05, w_time=0.02, w_guide=0.05, ghost_weight=10.0, w_vcp=0.0, w_vsplit=0.0,
                    w_checkpoint=3.0, w_finish=50.0, w_finish_time=10.0, w_steer_smooth=0.0, w_crash=0.05,
                    w_offzone=20.0, w_stuck=20.0, w_brake=0.0, w_coast=0.0, w_release=0.0, w_slip=0.0, w_hit=0.3,
                    w_wall=0.0, w_air=0.02, w_speed=0.03, w_novel=0.5, w_known=0.4, w_focus=1.5)
ALL_TERMS = tuple(OLD_DEFAULTS)
LEGACY_TERMS = tuple(k for k in ALL_TERMS if k not in WEIGHTS)
LEGACY_SETTINGS = {"vcp_every_s": 4.0, "cov_len": 25.0, "cov_passes": 100.0}   # only the legacy terms used these
NEW_IN_V4 = {"w_novel": 0.0, "w_known": 0.0, "w_focus": 0.0, "w_air": 0.0, "w_speed": 0.0, "w_hit": 0.0,
             "ghost_weight": 10.0}     # values for configs from before reward_v 4 that don't have the key

# "Linesight" preset: exactly two terms. Linesight's 0.01/m vs 0.06 per 50 ms breaks even at 120 m/s (TMNF);
# for TM2020 progress is 0.03/m, so driving pays from ~40 m/s (0.06 / 0.05 s / 0.03 per m).
LINESIGHT = {k: 0.0 for k in WEIGHTS}
LINESIGHT.update(w_progress=0.03, w_time=0.06)
MODEL_FILES = ("latest.zip", "champion.zip", "latest.pt", "latest.prev.pt", "champion.pt")


def freeze_legacy(weights):
    """reward_v 5 migration: every existing run with a model (runs/<run>, runs/_base/<base>) that has no rewards.json
    gets one with the weights it trained with (all terms + the legacy settings), so it resumes identically. The cars
    find it in their run folder or a parent folder (base maps, pool maps, transfer tests)."""
    roots = [d for d in (ROOT / "runs").glob("*") if d.is_dir() and not d.name.startswith("_")]
    roots += [d for d in (ROOT / "runs" / "_base").glob("*") if d.is_dir()]
    n = 0
    for d in roots:
        f = d / "rewards.json"
        if f.exists() or not any((d / m).exists() for m in MODEL_FILES):
            continue
        try:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps({"preset": "legacy", "weights": weights, "t": time.time()}), encoding="utf-8")
            tmp.replace(f)
            n += 1
        except OSError:
            pass
    return n


def load():
    cfg, on_disk = dict(DEFAULTS), None
    if CONFIG_PATH.exists():
        try:
            on_disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # no reward_v = a hand-written config (the dashboard always saves it): treat it as current, migrate nothing
    v = int(on_disk.get("reward_v", 5) or 5) if on_disk is not None else 5
    if on_disk is not None and v < 5:
        cfg.update(OLD_DEFAULTS)                # what a config from before v5 meant for keys it doesn't have
        cfg.update(LEGACY_SETTINGS)
    if on_disk is not None:
        cfg.update(on_disk)
    if v < 3:            # v2: finish bonus vs champion; v3: progress always beats the time penalty, giving up costs
        for k in (("w_finish_time",) if v < 2 else ()) + ("w_progress", "w_stuck", "w_offzone"):
            cfg[k] = OLD_DEFAULTS[k]
        cfg["reward_v"] = 3
        try:
            save(cfg)
        except OSError:
            pass
    if on_disk is not None and v < 4:     # v4: novelty/speed-focus weights; old configs keep their behaviour (0)
        for k, d in NEW_IN_V4.items():
            if k not in on_disk:
                cfg[k] = d
        cfg["reward_v"] = 4
        try:
            save(cfg)
        except OSError:
            pass
    if on_disk is not None and v < 5:     # v5: minimal rewards for new runs; existing runs keep theirs (rewards.json)
        legacy = {}
        for k in list(ALL_TERMS) + list(LEGACY_SETTINGS):
            try:
                legacy[k] = float(cfg.get(k, OLD_DEFAULTS.get(k, LEGACY_SETTINGS.get(k, 0.0))))
            except (TypeError, ValueError):
                legacy[k] = float(OLD_DEFAULTS.get(k, LEGACY_SETTINGS.get(k, 0.0)))
        freeze_legacy(legacy)
        for k in WEIGHTS:                     # new runs: the minimal defaults
            cfg[k] = WEIGHTS[k][0]
        cfg["reward_v"] = 5
        try:
            save(cfg)
        except OSError:
            pass
    return coerce(cfg)


def coerce(cfg):
    out = {}
    for k, d in DEFAULTS.items():
        v = cfg.get(k, d)
        try:
            if isinstance(d, bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(d, int):
                v = int(float(v))
            elif isinstance(d, float):
                v = float(v)
            elif isinstance(d, list):
                v = json.loads(v) if isinstance(v, str) else list(v)
            else:
                v = str(v).strip().strip('"')
        except (TypeError, ValueError):
            v = d
        out[k] = v
    out["envs"] = max(1, out["envs"])
    out["iqn_steer_levels"] = min((21, 31, 41), key=lambda n: abs(n - out["iqn_steer_levels"]))
    out["own_line_s"] = max(0.0, out["own_line_s"])
    out["minirace_s"] = min(max(out["minirace_s"], 1.0), 15.0)
    out["start_focus"] = min(max(out["start_focus"], 0.0), 1.0)
    pool = []
    for e in out["map_pool"] if isinstance(out["map_pool"], list) else []:
        if isinstance(e, str):
            e = {"map": e}
        if isinstance(e, dict) and str(e.get("map", "")).strip():
            try:
                pbs = float(e.get("pb_seconds") or 0)
            except (TypeError, ValueError):
                pbs = 0.0
            pool.append({"map": str(e["map"]).strip().strip('"'), "pb_file": str(e.get("pb_file") or "").strip().strip('"'),
                         "pb_seconds": pbs})
    out["map_pool"] = pool
    if out["run_type"] not in ("map", "base"):
        out["run_type"] = "map"
    for k in ("base_name", "base_from"):
        out[k] = "".join(c for c in out[k] if c.isalnum() or c in "-_")
    out["base_name"] = out["base_name"] or "base"
    out["pool_rotate_laps"] = max(1, out["pool_rotate_laps"])
    out["pool_cars"] = max(0, out["pool_cars"])
    out["lidar_every"] = min(max(1, out["lidar_every"]), 5)
    if out["iqn_device"] not in ("auto", "cpu", "cuda"):
        out["iqn_device"] = "auto"
    if out["algo"] not in ("ppo", "iqn"):
        out["algo"] = "ppo"
    if out["steer_mode"] not in ("progressive", "uniform"):
        out["steer_mode"] = DEFAULTS["steer_mode"]
    out["run"] = "".join(c for c in out["run"] if c.isalnum() or c in "-_") or "calm"
    out["iqn_warm_run"] = "".join(c for c in out["iqn_warm_run"] if c.isalnum() or c in "-_")
    return out


def save(cfg):
    text = json.dumps(coerce(cfg), indent=2)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        tmp.replace(CONFIG_PATH)             # atomic: running cars never read a half-written file
    except OSError:
        CONFIG_PATH.write_text(text, encoding="utf-8")


def run_key(name):
    """Run names that differ only in spaces, '-', '_' or case mean the same run: "Summer 2020 - 02",
    "Summer2020-02" and "summer2020_02" are all "summer202002"."""
    return "".join(c for c in str(name).lower() if c.isalnum())


RUN_WARNINGS = []


def find_run(name):
    """The existing runs/<folder> for this name (by normalised key), or None. If two existing folders share the
    key, the exact name wins, else the most recently trained one; both are kept and a warning is recorded."""
    root = ROOT / "runs"
    key = run_key(name)
    if not key or not root.exists():
        return None
    hits = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_") and run_key(d.name) == key]
    if not hits:
        return None
    if len(hits) > 1:
        msg = (f"Run folders {', '.join(sorted(h.name for h in hits))} all mean the same run name; using "
               + (name if any(h.name == name for h in hits) else "the most recently trained one")
               + ". Rename or remove the others to avoid confusion.")
        if msg not in RUN_WARNINGS:
            RUN_WARNINGS.append(msg)
    exact = [h for h in hits if h.name == name]
    return exact[0] if exact else max(hits, key=_last_trained)


def _last_trained(d):
    ts = [0.0]
    for f in ("status.json", "latest.zip", "latest.pt", "events.jsonl"):
        try:
            ts.append((d / f).stat().st_mtime)
        except OSError:
            pass
    return max(ts)


def run_dir(cfg):
    """runs/<run>/ for a map run (an existing folder whose name differs only in spaces / - / _ / case is the same
    run), runs/_base/<base_name>/ for a base run."""
    if cfg.get("run_type") == "base":
        d = base_dir(cfg["base_name"])
    else:
        d = find_run(cfg["run"]) or ROOT / "runs" / cfg["run"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def has_model(d):
    return any((Path(d) / f).exists() for f in ("latest.zip", "champion.zip", "latest.pt", "champion.pt"))


def run_summary(d):
    """Name, map, champion time, steps and last training time of a run folder (for the run picker)."""
    d = Path(d)
    mi, best = _read(d / "map_info.json"), _read(d / "best.json")
    steps = 0
    try:                                                   # the last "upd" event has the step count
        with open(d / "events.jsonl", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            for line in reversed(f.read().splitlines()):
                if b'"upd"' in line:
                    steps = int(json.loads(line).get("steps", 0))
                    break
    except (OSError, ValueError):
        pass
    return {"name": d.name, "map": mi.get("path"), "map_name": mi.get("name"), "uid": mi.get("uid"),
            "champion_ms": best.get("champion_ms"), "steps": steps, "last": round(_last_trained(d)),
            "has_model": has_model(d)}


def _read(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_runs():
    root = ROOT / "runs"
    if not root.exists():
        return []
    out = [run_summary(d) for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
           and any((d / f).exists() for f in ("map_info.json", "events.jsonl", "latest.zip", "latest.pt"))]
    return sorted(out, key=lambda r: -r["last"])


def map_uid_for(path):
    """The map UID of a map file, if any run has driven it (runs/_maps.json, written by the cars)."""
    for uid, rec in _read(ROOT / "runs" / "_maps.json").items():
        if str(rec.get("path", "")).lower() == str(path).lower():
            return uid
    return None


def same_map_runs(cfg):
    """Other runs with a model on the same map as cfg["map"] (by UID, or by path before the UID is known)."""
    uid, mp = map_uid_for(cfg["map"]), str(cfg["map"]).lower()
    mine = find_run(cfg["run"])
    out = []
    for r in list_runs():
        if mine is not None and r["name"] == mine.name:
            continue
        if r["has_model"] and ((uid and r["uid"] == uid) or str(r["map"] or "").lower() == mp):
            out.append(r)
    return out


def start_conflict(cfg):
    """A new run (no model yet) on a map that already has a trained run: the runs to offer instead, else []."""
    if cfg.get("run_type") == "base":
        return []
    mine = find_run(cfg["run"])
    if mine is not None and has_model(mine):
        return []
    return same_map_runs(cfg)


def base_dir(name):
    return ROOT / "runs" / "_base" / name


def map_slug(path):
    """Folder name for a map: its file name + a short hash of the full path (two maps never share a folder)."""
    import hashlib
    stem = Path(str(path)).name
    for ext in (".Map.Gbx", ".map.gbx", ".Gbx", ".gbx"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    clean = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:40].strip("_") or "map"
    return f"{clean}_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:6]}"


def map_cfg(cfg, entry, pnorm=0, pool=False):
    """Settings of one car/tester on one pool map: the map, its PB, the progress normalisation."""
    return dict(cfg, map=entry["map"], pb_file=entry.get("pb_file", ""), pb_seconds=float(entry.get("pb_seconds") or 0),
                _slug=map_slug(entry["map"]), _pnorm=pnorm, _pool=pool)
