"""Fresh runs must get going at the start: on a fake map whose start block is listed as a waypoint group (as on
real maps), a fresh PPO run and a fresh IQN run must hold gas > 60 % of the steps in their first laps and get past
the start (not every lap "stuck"). Usage: python tests/test_start.py   (about 2.5 minutes)"""
import collections, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(algo):
    d = Path(tempfile.mkdtemp(prefix="tmai_start_"))
    for f in REPO.glob("*.py"):
        shutil.copy(f, d)
    (d / "config.json").write_text(json.dumps({"map": "x", "run": "t", "envs": 2, "test_every_laps": 3,
                                               "time_limit_min": 1, "reward_v": 4, "algo": algo}))
    env = dict(os.environ, PYTHONPATH=str(REPO / "tests"), TICKGYM_FAKE_STARTWP="Start", TMAI_IQN_WARMUP="3000")
    subprocess.run([sys.executable, "-u", "tm_ai.py", "train"], cwd=d, env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=300)
    ev = [json.loads(l) for l in open(d / "runs" / "t" / "events.jsonl")]
    laps = [e for e in ev if e["k"] == "lap" and e.get("ast")]
    first = laps[:30]
    gas = sum(e["ast"]["m"][0] + (e["ast"]["m"][3] if len(e["ast"]["m"]) > 3 else 0) for e in first) / len(first)
    ends = collections.Counter(e["end"] for e in laps)
    far = max(e["ast"]["s"] for e in laps)
    shutil.rmtree(d, ignore_errors=True)
    ok = gas > 0.6 and ends.get("stuck", 0) < len(laps) and far > 100
    print(f"{'PASS' if ok else 'FAIL'} {algo}: gas {gas:.0%} of the steps in the first {len(first)} laps, "
          f"furthest {far:.0f} m, ends {dict(ends)}")
    return ok


if __name__ == "__main__":
    res = [run("ppo"), run("iqn")]
    print("ALL PASSED" if all(res) else "FAILED")
    sys.exit(0 if all(res) else 1)
