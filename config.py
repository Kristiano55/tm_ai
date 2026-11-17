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
    "test_every_laps": 50,     # champion test after this many training laps
    "rollback_after": 3,       # roll back to the champion after N bad tests in a row
    "rollback_margin_s": 1.0,  # a test counts as bad if slower than champion by this much
    "resume": True,            # continue from runs/<run>/latest.zip if it fits
    "extra_obs": True,         # extra observations: effects/runtime telemetry, ghost speed, line offset ...
    "steer_mode": "progressive",   # "progressive" (fine near centre, coarse at the edges) or "uniform" (10% steps)
    "cut_behind_s": 5.0,       # end a lap that falls this far behind the ghost (0 = off)
    "decay_steps": 4000000,    # learning rate + exploration halve every N env steps (0 = off)
    "line_refresh_s": 0.2,
    "mid_start_p": 0.6,        # share of training laps that start partway along the champion's lap (0 = off)
    "mid_start_every_s": 2.0,  # spacing of those start points along the champion lap, seconds     # rebuild the reference line + ghost when the champion beats it by this much (0 = never)
    "reward_v": 2,             # reward layout version (old configs are upgraded once)
}


# Reward weights: key -> (default, label, hint). They are read live: running cars pick up a saved
# change at the start of their next lap (no restart needed).
WEIGHTS = {
    "w_progress": (0.02, "Progress", "reward per metre driven along the route (keep small: it pays the same for slow and fast laps)"),
    "w_time": (0.02, "Time penalty", "cost per 50 ms step; this is what makes going fast pay"),
    "ghost_weight": (10.0, "Ghost", "reward per second gained on the fastest run so far (0 = off)"),
    "w_checkpoint": (3.0, "Checkpoint bonus", "flat reward per checkpoint"),
    "w_finish": (50.0, "Finish bonus", "flat reward for finishing"),
    "w_finish_time": (10.0, "Finish speed bonus", "reward per second faster than the champion at the finish (negative if slower)"),
    "w_steer_smooth": (0.002, "Smooth steering", "penalty per unit of steering change between steps"),
    "w_crash": (0.05, "Crash penalty", "per m/s of sudden speed loss (wall hits)"),
    "w_offzone": (10.0, "Off-zone penalty", "when the car leaves the playable area"),
    "w_stuck": (5.0, "Stuck penalty", "when the car stands still for 2 s"),
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


def load():
    cfg, on_disk = dict(DEFAULTS), None
    if CONFIG_PATH.exists():
        try:
            on_disk = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update(on_disk)
        except Exception:
            pass
    if on_disk is not None and int(on_disk.get("reward_v", 1) or 1) < 2:       # v2: progress no longer dominates, finish bonus vs champion
        for k in ("w_progress", "w_finish_time"):
            cfg[k] = WEIGHTS[k][0]
        cfg["reward_v"] = 2
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
    if out["steer_mode"] not in ("progressive", "uniform"):
        out["steer_mode"] = DEFAULTS["steer_mode"]
    out["run"] = "".join(c for c in out["run"] if c.isalnum() or c in "-_") or "calm"
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
