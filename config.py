"""Shared settings. The dashboard edits config.json, tm_ai.py reads it."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS = {
    "map": "",                 # full path to the .Map.Gbx
    "run": "calm",             # run name -> runs/<run>/
    "envs": 8,                 # parallel cars (gyms); one more is used for champion tests
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
    "iqn_warm_run": "",             # IQN: fill the replay memory with laps of this PPO run's champion first (empty = off)
    "vcp_every_s": 4.0,        # virtual checkpoints every N seconds of the PB/ghost pace (0 = off)
    "test_tries": 3,           # champion test = best of N laps (most-likely actions + seeded ones)
    "horizon_s": 20.0,         # how far ahead (seconds) the AI weighs rewards     # ghost reward only counts within this many seconds behind the ghost (live)       # end a lap that falls this far behind the ghost (0 = off)
    "decay_steps": 4000000,    # learning rate + exploration halve every N env steps (0 = off)
    "line_refresh_s": 0.2,
    "pb_file": "",             # your PB as a TICK input text file: traced once into a target line + timing (empty = off)
    "mid_start_p": 0.6,        # share of training laps that start partway along the champion's lap (0 = off)
    "mid_start_every_s": 2.0,  # spacing of those start points along the champion lap, seconds     # rebuild the reference line + ghost when the champion beats it by this much (0 = never)
    "reward_v": 3,             # reward layout version (old configs are upgraded once)
}


# Reward weights: key -> (default, label, hint). They are read live: running cars pick up a saved
# change at the start of their next lap (no restart needed).
WEIGHTS = {
    "w_progress": (0.05, "Progress", "reward per metre driven along the route (keep small: it pays the same for slow and fast laps)"),
    "w_time": (0.02, "Time penalty", "cost per 50 ms step; this is what makes going fast pay"),
    "w_guide": (0.05, "Pull toward the target line", "per step, by distance from the PB/best line; only while behind its pace"),
    "ghost_weight": (10.0, "Ghost", "reward per second gained on the fastest run so far (0 = off)"),
    "w_vcp": (1.0, "Virtual checkpoint bonus", "flat reward per virtual checkpoint (every few s of the PB pace)"),
    "w_vsplit": (2.0, "Virtual checkpoint split", "per second ahead (+) or behind (-) the PB/ghost at each virtual checkpoint"),
    "w_checkpoint": (3.0, "Checkpoint bonus", "flat reward per checkpoint"),
    "w_finish": (50.0, "Finish bonus", "flat reward for finishing"),
    "w_finish_time": (10.0, "Finish speed bonus", "reward per second faster than the champion at the finish (negative if slower)"),
    "w_steer_smooth": (0.002, "Smooth steering", "penalty per unit of steering change between steps"),
    "w_crash": (0.05, "Crash penalty", "per m/s of sudden speed loss (wall hits)"),
    "w_offzone": (20.0, "Off-zone penalty", "when the car leaves the playable area (plus the progress it gave up)"),
    "w_stuck": (20.0, "Stuck penalty", "when the car stands still for 2 s (plus the progress it gave up)"),
    "w_brake": (0.0, "Brake penalty", "per step while braking"),
    "w_coast": (0.0, "Coast penalty", "per step with neither gas nor brake, everywhere"),
    "w_release": (0.03, "Pointless release penalty", "per step released on a straight with nothing ahead; corners stay free"),
    "w_slip": (0.0, "Slip penalty", "per step, scaled by the share of slipping wheels"),
    "w_hit": (0.0, "Wall-hit penalty", "per step with a wall hit (exact flag from the game; try 0.3)"),
    "w_wall": (0.0, "Wall-scrape penalty", "near a wall at speed; can discourage tight racing lines"),
    "w_air": (0.0, "Airtime penalty", "per step with no wheel on the ground"),
    "w_speed": (0.0, "Speed reward", "per step, proportional to speed (direct speed incentive)"),
}
DEFAULTS.update({k: v[0] for k, v in WEIGHTS.items()})

# "Lean" preset: only the terms that define the task; everything else off.
LEAN = {k: 0.0 for k in WEIGHTS}
LEAN.update(w_progress=0.05, w_time=0.02, ghost_weight=10.0, w_checkpoint=3.0, w_finish=50.0, w_finish_time=10.0,
            w_crash=0.05, w_offzone=20.0, w_stuck=20.0, w_guide=0.05)

# "Linesight" preset: exactly two terms. Linesight's 0.01/m vs 0.06 per 50 ms breaks even at 120 m/s (TMNF);
# for TM2020 progress is 0.03/m, so driving pays from ~40 m/s (0.06 / 0.05 s / 0.03 per m).
LINESIGHT = {k: 0.0 for k in WEIGHTS}
LINESIGHT.update(w_progress=0.03, w_time=0.06)


def load():
    cfg, on_disk = dict(DEFAULTS), None
    if CONFIG_PATH.exists():
        try:
            on_disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update(on_disk)
        except Exception:
            pass
    v = int(on_disk.get("reward_v", 1) or 1) if on_disk is not None else 3
    if v < 3:            # v2: finish bonus vs champion; v3: progress always beats the time penalty, giving up costs
        for k in (("w_finish_time",) if v < 2 else ()) + ("w_progress", "w_stuck", "w_offzone"):
            cfg[k] = WEIGHTS[k][0]
        cfg["reward_v"] = 3
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
            else:
                v = str(v).strip().strip('"')
        except (TypeError, ValueError):
            v = d
        out[k] = v
    out["envs"] = max(1, out["envs"])
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


def run_dir(cfg):
    d = ROOT / "runs" / cfg["run"]
    d.mkdir(parents=True, exist_ok=True)
    return d
