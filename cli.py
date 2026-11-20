"""
Command line: selftest, diagnose, packs/hub commands, main() (tm_ai.py calls it).
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
from train_ppo import *  # noqa: F401,F403
from train_ppo import _make_callback  # noqa: F401


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
    from collections import Counter
    env = TMEnv(cfg, cfgmod.run_dir(cfg), shorten=False)
    o, _ = env.reset()
    seen = {"wheel material": Counter(), "wheel gameplay": Counter(), "ray material": Counter()}
    ice = {}
    for _ in range(40):
        o, r, term, trunc, info = env.step(np.array([0.0, 0]) if env.analog else np.array([10, 0]))   # straight, full gas
        t = env._last_t
        for m in getattr(t.wheels, "material_id", []):
            seen["wheel material"][int(m)] += 1
        hit = env._lid[0]
        for i in env.SURF_RAYS:
            if i < len(hit) and hit[i]:
                seen["ray material"][int(env._lid_mat[i])] += 1
        rt = getattr(t, "runtime", None)
        if rt is not None:
            for k in env.WHEELS4:
                seen["wheel gameplay"][int(_f(rt, f"{k}_surface_gameplay_id"))] += 1
            for name in [f"{k}_icing_{x}" for k in env.WHEELS4 for x in ("state", "blend")] + [
                    "front_wheels_icing_grip_scale", "rear_wheels_icing_grip_scale", "is_any_wheel_on_ice_or_snow"]:
                ice[name] = max(ice.get(name, -1e9), _f(rt, name, float("nan")))
        if term or trunc:
            break
    print("steering:", "analog -127..127 (PPO)" if env.analog else env.STEER.tolist())
    print("waypoint groups:", [(wp.group_id, _wp_kind(wp) or "?") for wp in env.g.map.waypoints],
          "| dropped as start block:", [np.round(c, 1).tolist() for c in env.dropped_wp] or "none")
    print("route points:", len(env.route) - 1, "| route length: %.0f m" % env.lap_len,
          "| reference line:", "yes" if env.dense is not None else "not yet")
    print("local velocity m/s (expect z > 0, x ~ 0):", np.round(o[1:4] * 100, 2))
    print("world up in car frame (expect ~ 0, 1, 0):", np.round(o[7:10], 2))
    print("first lookahead (angle/pi, elevation):", np.round(o[env.LOOK_OFF:env.LOOK_OFF + 2], 2))
    if env.extra:
        print("missing telemetry fields (these always read their default):", missing_fields(env) or "none")
        print("extra observations:", {n: round(float(x), 3) for n, x in zip(EXTRA_NAMES, o[N_BASE0:env.N_BASE])})
    _vocab_load()
    for kind, c in seen.items():
        voc = _VOCAB["gameplay" if "gameplay" in kind else "material"]
        print(f"{kind} ids seen (id: count -> input slot):",
              ", ".join(f"{k}: {n} -> {voc.get(k, N_SURF) + 1 if voc.get(k, N_SURF) < N_SURF else 'other'}"
                        for k, n in sorted(c.items())) or "none")
    if ice:
        print("ice raw values (max over the test drive; on a summer map icing_state/blend and is_any_wheel_on_ice"
              "_or_snow must be 0, grip scales 1):", {k: round(v, 3) for k, v in ice.items()})
    lid = o[env.N_BASE:env.N_BASE + len(env.ray_range)] * env.ray_range   # lidar only (PPO appends N_PRIV after)
    print("lidar min/max m:", np.round(lid.min(), 1), np.round(lid.max(), 1), "| obs size:", o.shape[0])
    env.close()


# --- diagnose: short A/B PPO trials on the real game ------------------------------------------------------
DIAG_TRIALS = [("a", "21 levels, 3 modes", dict(analog_steer=False, gas_brake=False)),
               ("b", "analog, 3 modes", dict(analog_steer=True, gas_brake=False)),
               ("c", "21 levels, gas+brake", dict(analog_steer=False, gas_brake=True)),
               ("d", "analog, gas+brake", dict(analog_steer=True, gas_brake=True))]
DIAG_FAIL = ("stuck", "far off line", "off-zone")


def diag_metrics(run_dir, n_modes):
    """Summary of one trial from its events.jsonl."""
    laps, run = [], []
    try:
        with open(Path(run_dir) / "events.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("k") == "lap" and not e.get("pool"):
                    laps.append(e)
                elif e.get("k") == "rb":
                    run.append(sum(v for k, v in e["r"].items() if k not in ("finish", "end", "checkpoint")))
    except OSError:
        pass
    ast = [e["ast"] for e in laps if e.get("ast")]
    stuck = [e for e in laps if e.get("end") in DIAG_FAIL]
    spots = {}
    for e in stuck:
        if e.get("x") is not None:
            k = (int(e.get("cp", 0)), int(round(e["x"] / 30.0)) * 30, int(round(e["z"] / 30.0)) * 30)
            spots[k] = spots.get(k, 0) + 1
    top = max(spots.items(), key=lambda kv: kv[1]) if spots else None
    prog = [a["s"] for a in ast]
    late = prog[len(prog) // 2:] or prog
    fin = [e["ms"] for e in laps if e.get("ms") and not e.get("mid")]
    modes = np.mean([a["m"] + [0.0] * (n_modes - len(a["m"])) for a in ast], axis=0) if ast else np.zeros(n_modes)
    names = ["gas", "coast", "brake", "gas+brake"][:n_modes]
    return {
        "laps": len(laps), "driving_earns_per_step": round(float(np.mean(run)), 4) if run else None,
        "driving_earns_last": round(float(np.mean(run[-5:])), 4) if run else None,
        "stuck_share": round(len(stuck) / len(laps), 3) if laps else None,
        "stuck_spot": ({"cp": top[0][0], "x": top[0][1], "z": top[0][2], "share_of_stuck": round(top[1] / len(stuck), 3)}
                       if top else None),
        "mean_speed_ms": round(float(np.mean([a["v"] for a in ast])), 1) if ast else None,
        "furthest_m": round(float(max(prog)), 1) if prog else None,
        "progress_p90_late_m": round(float(np.percentile(late, 90)), 1) if late else None,
        "throttle_share": {n: round(float(x), 3) for n, x in zip(names, modes)},
        "steer_mean_abs": round(float(np.mean([a["sa"] for a in ast])) * 127, 1) if ast else None,
        "steer_full_lock_share": round(float(np.mean([a["fl"] for a in ast])), 3) if ast else None,
        "best_lap_s": round(min(fin) / 1000.0, 3) if fin else None,
        "finished_laps": len(fin)}


def _diag_score(m):
    return (1 if m.get("best_lap_s") else 0, -(m.get("best_lap_s") or 0.0), m.get("progress_p90_late_m") or 0.0)


def _diag_worse(w, wo):
    """True if the trial WITH a feature (w) learned clearly worse than the one without (wo)."""
    pw, po = w.get("progress_p90_late_m") or 0.0, wo.get("progress_p90_late_m") or 0.0
    if wo.get("best_lap_s") and not w.get("best_lap_s"):
        return True
    if w.get("best_lap_s") and wo.get("best_lap_s"):
        return w["best_lap_s"] > wo["best_lap_s"] + 0.5
    return pw < 0.75 * po or (w.get("stuck_share") or 0) - (wo.get("stuck_share") or 0) > 0.2


def diag_verdict(res):
    t, out = {k: v["metrics"] for k, v in res.items()}, []
    if all(k in t for k in "abcd"):
        for name, pairs in (("analog steering", (("b", "a"), ("d", "c"))), ("gas+brake", (("c", "a"), ("d", "b")))):
            bad = [_diag_worse(t[w], t[wo]) for w, wo in pairs]
            out.append(f"{name}: " + ("BREAKS learning here (worse in both comparisons)" if all(bad) else
                                      "probably hurts (worse in one comparison)" if any(bad) else "no harm seen"))
    if "e" in t and res["e"].get("from") in t:
        out.append("novelty rewards: " + ("hurt learning" if _diag_worse(t["e"], t[res["e"]["from"]]) else "no harm seen"))
    return out


def diagnose(cfg, map_path, pb="", minutes=4.0):
    """Short PPO trials from the same fresh start (runs diag_a .. diag_e), report -> runs/_diag/diagnose.json."""
    import subprocess
    root = cfgmod.ROOT
    ddir = root / "runs" / "_diag"
    ddir.mkdir(parents=True, exist_ok=True)
    base = dict(cfg)
    if map_path and map_path != cfg["map"]:
        base.update(map=map_path, pb_file=pb or "", pb_seconds=0.0)
    elif pb:
        base["pb_file"] = pb
    base.update(algo="ppo", run_type="map", time_limit_min=0, stop_at_pb=False, resume=True,
                w_novel=0.0, w_known=0.0, w_focus=0.0, reward_v=4)
    secs = float(os.environ.get("TMAI_DIAG_S") or minutes * 60.0)
    say(f"Diagnose: {len(DIAG_TRIALS) + 1} PPO trials of {secs / 60:.1f} min on {base['map']} "
        f"({base['envs']} cars, {base['ticks_per_step'] * 10} ms per step) ...")
    res = {}

    def trial(key, label, over):
        name = f"diag_{key}"
        rd = root / "runs" / name
        if rd.exists():
            import shutil
            shutil.rmtree(rd, ignore_errors=True)               # same fresh start for every trial
        c = cfgmod.coerce(dict(base, run=name, **over))
        cpath = ddir / f"config_{key}.json"
        cpath.write_text(json.dumps(c, indent=1), encoding="utf-8")
        say(f"Diagnose trial {key} ({label}) ...")
        env = dict(os.environ, TMAI_CONFIG=str(cpath))
        with open(ddir / f"{name}.log", "w", encoding="utf-8") as logf:
            entry = str(Path(__file__).resolve().with_name("tm_ai.py"))
            p = subprocess.Popen([sys.executable, "-u", entry, "train", "--new", "--run", name],
                                 cwd=str(root), env=env, stdout=logf, stderr=subprocess.STDOUT)
            t0 = None
            while p.poll() is None:
                time.sleep(1.0)
                if t0 is None and (rd / "events.jsonl").exists():
                    t0 = time.time()                            # count from the training start, not the start-up
                if t0 is not None and time.time() - t0 > secs:
                    (rd / "STOP").write_text("diagnose")
                    try:
                        p.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    break
        m = diag_metrics(rd, 4 if c["gas_brake"] else 3)
        res[key] = {"label": label, "settings": over, "metrics": m}
        say(f"  {key} {label}: earns/step {m['driving_earns_per_step']}, stuck {m['stuck_share']}, "
            f"speed {m['mean_speed_ms']} m/s, furthest {m['furthest_m']} m, best {m['best_lap_s']}")

    for key, label, over in DIAG_TRIALS:
        trial(key, label, over)
    best = max("abcd", key=lambda k: _diag_score(res[k]["metrics"]))
    nov = {k: cfgmod.WEIGHTS[k][0] for k in ("w_novel", "w_known", "w_focus")}
    trial("e", f"{res[best]['label']} + novelty defaults", dict(res[best]["settings"], **nov))
    res["e"]["from"] = best
    verdict = diag_verdict(res)
    report = {"t": time.time(), "map": base["map"], "pb_file": base.get("pb_file"), "minutes": round(secs / 60, 2),
              "ticks_per_step": base["ticks_per_step"], "cars": base["envs"], "trials": res, "best": best,
              "verdict": verdict}
    write_json(ddir / "diagnose.json", report)
    print("\n" + "=" * 100)
    hdr = f"{'trial':26s}{'earns/step':>11s}{'stuck':>7s}{'speed':>7s}{'furthest':>9s}{'|steer|':>8s}{'lock':>6s}  throttle share{'':22s}best"
    print(hdr)
    for k, r in res.items():
        m = r["metrics"]
        thr = " ".join(f"{n[:4]} {v:.0%}" for n, v in m["throttle_share"].items())
        print(f"{k + ' ' + r['label']:26.26s}{str(m['driving_earns_per_step']):>11s}{str(m['stuck_share']):>7s}"
              f"{str(m['mean_speed_ms']):>7s}{str(m['furthest_m']):>9s}{str(m['steer_mean_abs']):>8s}"
              f"{str(m['steer_full_lock_share']):>6s}  {thr:36s}{m['best_lap_s']}")
        if m.get("stuck_spot"):
            sp = m["stuck_spot"]
            print(f"{'':26s}most stuck laps: after CP {sp['cp']} near x {sp['x']}, z {sp['z']} "
                  f"({sp['share_of_stuck']:.0%} of the stuck laps)")
    print("Verdict: " + " | ".join(verdict) + f" | best of a-d: {best}")
    print(f"Report: {ddir / 'diagnose.json'}")
    say("Diagnose finished: " + " | ".join(verdict))


def pack_cli(argv):
    """publish / verify / benchmark / export / import / hub: brain packs and the public hub (hub.py, packs.py)."""
    import hub
    import packs
    cfg = cfgmod.load()
    cmd, rest = argv[0], argv[1:]
    ap = argparse.ArgumentParser(prog=f"tm_ai.py {cmd}")
    if cmd == "hub":
        return hub.main(cfg, rest)
    ap.add_argument("target", help="run name, base:NAME, pack zip or pack id")
    ap.add_argument("--map", action="append", help="UID=PATH of a local map (verify / benchmark)")
    ap.add_argument("--wr", type=float, default=None, help="publish/export: world record time in seconds")
    ap.add_argument("--note", default="")
    ap.add_argument("--as", dest="name", default=None, help="import: run (or base) name")
    ap.add_argument("--base", action="store_true", help="import as a base model")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="verify: also run the benchmark maps")
    a = ap.parse_args(rest)
    if cmd == "publish":
        hub.cmd_publish(cfg, a.target, a.wr, a.note, open_browser=not a.no_browser)
    elif cmd == "export":
        path, man = packs.export_pack(cfg, a.target, cfgmod.ROOT / "runs" / "_hub" / "out",
                                      author=cfg["github_user"] or "anonymous", wr=a.wr, note=a.note)
        say(f"Exported {path}")
    elif cmd == "import":
        name = a.name or Path(a.target).name.split(".")[0]
        dest = cfgmod.base_dir(name) if a.base else cfgmod.ROOT / "runs" / name
        man = packs.import_pack(a.target, dest, as_base=a.base)
        say(f"Imported pack {man['pack_id']} into {dest}")
    elif cmd == "verify":
        hub.cmd_verify(cfg, a.target, hub.parse_overrides(a.map), open_browser=not a.no_browser, bench=a.benchmark)
    elif cmd == "benchmark":
        hub.cmd_benchmark(cfg, a.target, hub.parse_overrides(a.map))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("hub", "publish", "verify", "benchmark", "export", "import"):
        pack_cli(sys.argv[1:])
        raise SystemExit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "train", "drive", "bench", "transfer", "diagnose"])
    ap.add_argument("diag_map", nargs="?", default=None, help="diagnose: map path (default: the configured map)")
    ap.add_argument("--min", type=float, default=4.0, help="diagnose: minutes per trial")
    ap.add_argument("--pb", default="", help="transfer: optional PB file of that map (for a target line)")
    ap.add_argument("--map")
    ap.add_argument("--run")
    ap.add_argument("--envs", type=int)
    ap.add_argument("--model")
    ap.add_argument("--fresh", action="store_true", help="ignore latest.zip")
    ap.add_argument("--new", action="store_true", help="start this new run from scratch even if another run of the "
                                                         "same map has a trained model")
    a = ap.parse_args()
    cfg = cfgmod.load()
    for key in ("map", "run", "envs"):
        if getattr(a, key) is not None:
            cfg[key] = getattr(a, key)
    cfg = cfgmod.coerce(cfg)
    if a.fresh:
        cfg["resume"] = False
    if a.cmd == "diagnose":
        diagnose(cfg, a.diag_map or a.map or cfg["map"], a.pb, a.min)
        raise SystemExit(0)
    if a.cmd == "transfer":
        if not a.map:
            raise SystemExit("transfer needs --map <map not in the pool>")
        transfer(cfg, a.map, a.pb)
        raise SystemExit(0)
    if not cfg["map"] and not (cfg["run_type"] == "base" and cfg["map_pool"]):
        raise SystemExit("No map set. Pass --map or set it in the dashboard (Settings).")
    if a.cmd == "train" and not a.new and a.run is None:
        others = cfgmod.start_conflict(cfg)
        if others:
            print(f"Run '{cfg['run']}' has no model yet, but this map already has trained runs:")
            for r in others:
                print(f"  {r['name']:30s} champion {r['champion_ms'] / 1000:.3f} s" if r["champion_ms"] else
                      f"  {r['name']:30s} no champion yet", f" {r['steps']:,} steps")
            print("Continue one with --run NAME (or pick it in the dashboard), or start fresh with --new.")
            raise SystemExit(2)
    if cfg["run_type"] == "base" and cfg["algo"] == "iqn":
        raise SystemExit("Base runs are PPO only: set the algorithm to PPO.")
    if cfg["algo"] == "iqn" and a.cmd in ("train", "drive"):
        if cfg.get("analog_steer"):
            say("Analog steering is PPO-only: IQN drives with steering levels (setting: Steering levels (IQN)).")
        import iqn
        {"train": iqn.train, "drive": iqn.drive}[a.cmd](cfg)
    else:
        {"selftest": selftest, "train": train, "drive": lambda c: drive(c, a.model), "bench": bench}[a.cmd](cfg)
