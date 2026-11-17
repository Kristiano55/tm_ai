# tm_ai — Trackmania 2020 RL agent on tickgym

Owner: Kris (Kristiano55). Windows PC, Ryzen 5 5600, Python 3.12, TICK 3.0 at `%LOCALAPPDATA%\TICK`.
Install folder on his PC: `C:\Users\Kris\Documents\tm-ai`. Maps: `C:\Users\Kris\Documents\Trackmania\Maps\...`.
He is a TM2020 TAS / SpeedTAS player, not an ML engineer: explain changes in Trackmania terms, keep replies short.

## What it is
An AI that learns to drive Trackmania 2020 maps through **tickgym** (https://tm.achepta.com/tickgym/), a Python
library that runs the game's own deterministic physics via TICK. The same seed + inputs always give the same race.
Gyms can `snapshot()` / `restore()`, cast lidar rays, read telemetry, and `export()` a run as TICK input text.
Nothing is shown in the game window: laps are simulated; results are saved as TICK input files (`*.txt`).

## Files
- `tm_ai.py` — environment (`TMEnv`, gymnasium), observations, rewards, reference line / ghost / PB guide,
  mid-track starts, `CarPool` (asynchronous cars: each car process steps as soon as its own action arrives, pipes +
  `connection.wait`), `bench`, profiling (`prof_summary`), `raise_priority` (psutil, Windows only), PPO training (SB3) with **champion tests in a separate spawn process** (`ppo_test_worker`, fed
  EMA weights by the callback; rollback via `ROLLBACK` file), `selftest`, `drive`. CLI:
  `python tm_ai.py selftest|train|drive|bench` (settings from `config.json`). Dispatches to `iqn.py` when `algo == "iqn"`.
- `iqn.py` — Linesight-style learner: IQN (dueling, 64-dim cos embedding, N=8/K=32, kappa 5e-3), replay buffer
  (float16), 3-step returns, 7 s **mini-races** (time-left input appended, random per sample, gamma 0.999 -> 1),
  epsilon + Boltzmann exploration (eps points from `iqn_eps_*` settings), Linesight LR schedule, soft target updates,
  **learner in its own process** (`learner_proc`; 2 torch threads, CUDA if `iqn_device` allows; owns net/tgt/opt,
  saves `latest.pt`, handles `ROLLBACK`), **replay in shared memory** (`SharedReplay`, written by the driving loop
  under a lock), weights published to the driving copy every 20 updates via a shared float array; cars driven
  asynchronously through `T.CarPool`; `iqn_reuse` is a target, cars never wait; **champion tests in a
  separate spawn process** (sends `ROLLBACK` file to the learner), optional **warm start** (`iqn_warm_run`: ~50k
  steps of a PPO run's `champion.zip`, eps 0.1, into the replay memory; fresh runs only).
  Saves `latest.pt`, `champion.pt`, `iqn.json`.
- `config.py` — `DEFAULTS`, `WEIGHTS` (reward weights: key -> (default, label, hint)), presets `LEAN`, `LINESIGHT`,
  `load()/coerce()/save()` (atomic), `reward_v` migration of old configs, `run_dir()`.
- `dashboard.py` + `dashboard.html` — stdlib HTTP server on 127.0.0.1:8765; starts/stops `tm_ai.py` as a subprocess,
  reads `runs/<run>/events.jsonl` incrementally; vanilla JS + canvas charts (no libraries). Has a frozen-training
  guard (restarts training if heartbeat is stale 150 s).
- `setup.bat` (pip install, CPU torch), `start_dashboard.bat`, `requirements.txt`.
- `tests/tickgym.py` — **fake tickgym** for testing without Windows/TICK/the game (see Testing).

## Run folder `runs/<run>/`
`events.jsonl` (k = start | lap | test | upd | rb | prof | stop; prof = 5-min "where the time goes"), `status.json` (heartbeat), `train.log`, `best.json`
(champion_ms, first_test_ms, champion_seed), `champion.zip|.pt`, `champion.txt` (TICK input), `best_run.txt/.json`
(fastest finished full lap of any kind), `last_test.txt`, `play.txt`, `centerline.npy/.json` (reference line),
`profile.npy/.json` (ghost: arc length -> race tick), `champ_trace.npy`, `champion_actions.json` (for mid-starts),
`pb_trace.npy/.json` (traced PB), `best_trace.npy`, `latest.zip|.pt`, `vecnorm.pkl` (PPO reward scaling),
`bench.json` (car-count benchmark), `levels.json` (steering levels + obs layout, for checkpoint conversion), `tester_base.zip` (PPO network layout
  for the test process, rewritten each start), `STOP` / `ROLLBACK` flag files.

## Key design decisions (and why)
- **Actions:** 21 progressive steering levels (fine near centre) x {gas, coast, brake}; `ticks_per_step` 5 (50 ms)
  default, 3 (30 ms) possible. Per-step costs scale with `tf = ticks/5` so they mean the same per second.
- **Observations** (~290 + 41 lidar): car-frame velocity/rotation/up, wheels, steer/gear, last actions, route
  lookahead, 41 extras (effects/runtime/ghost/line offset), track preview (10 pts), short history (5 steps),
  40-point 3D line preview every 10 m (Linesight), 3 older actions, lidar (32 fan + 4 down + 5 up).
  All route points of a step are interpolated in one batch (`_route_points`, `to_car_frame_many`): ~5x faster obs.
  **New features must be appended at the end of the non-lidar block** so `migrate()` / `widen_columns()` can
  convert old PPO checkpoints (new input weights start at 0 = same behaviour). IQN checkpoints don't convert.
- **`icing` and `on_ice` inputs are forced to 0**: on a summer map selftest read 1.0 for both (unknown semantics).
- **Rewards** are a dict per step (`r[...]`), summed; per-term averages are logged (`rb`) for the dashboard's
  reward breakdown. Lessons learned the hard way:
  - Giving up must never pay: stuck/off-zone cost `w_stuck/w_offzone + w_progress * remaining metres` (PPO); with
    IQN the remaining-metres part is skipped and IQN charges the time cost for the rest of the mini-race instead.
  - LINESIGHT preset is progress 0.03/m, time 0.06 per 50 ms (driving pays from ~40 m/s; Linesight's TMNF ratio
    0.01/m broke even at 120 m/s, so IQN on TM2020 got stuck at ~17 m/s).
  - Ghost reward only within `ghost_window_s` behind the ghost, otherwise slow laps learned to quit.
  - **Progress off the target line** (the WR line can take a cut the AI doesn't): the matched arc length is clamped
    between the last taken and next checkpoint (`cp_arc`, waypoints mapped onto the line); > `NEAR_LINE` 15 m off,
    the line is re-acquired anywhere in that section. Progress already paid is `_s`: a skipped stretch is paid out at
    <= 10 m per step when the car rejoins further on, a backward jump costs once (-10 m max).
  - **Stuck rule:** "<2 m along the line in 2 s" only if the car was within 15 m of the line the whole 2 s; otherwise
    stuck = < 3 m actually driven in 2 s (sum of movements). Speed < 1.5 m/s for 2 s as before. "far off line" ends a
    lap after 8 s continuously > 40 m off. `StuckWatch` logs a hint when >80% of the last 200 laps end stuck at one spot.
  - Laps are cut only against the AI's **own champion** pace (`champ_trace`), never before a champion exists.
  - A dense positive "while driving" reward is required for PPO; dashboard warns when it goes negative.
- **PB guide:** `pb_file` (TICK input text of Kris's PB) is replayed once to trace *where/when* it drove; it becomes
  the target line + ghost; per checkpoint section the faster of PB vs AI best run is used (`build_guide`). The AI
  never drives the PB inputs (Kris was explicit about this).
- **Virtual checkpoints** every `vcp_every_s` of ghost pace: flat bonus + split vs ghost.
- **Mid-track starts** (snapshots along champion lap / PB) only after the first finished lap; IQN uses
  `iqn_mid_start_p` (0.2) and, with a `pb_file`, starts only along the PB/WR states (to learn the WR route's cut); built BEFORE
  `g.reset()` in `reset()` (an earlier bug made laps start at the finish).
- **Champion tests:** best of `test_tries` (argmax + seeded sampling, reproducible); PPO uses EMA weights and
  `target_kl=0.02`; rollback after `rollback_after` bad tests (IQN only if `iqn_rollback`, default off). Both algos test in a spawn process (busy flag,
  restarted if it dies); the tester writes `ROLLBACK`, the learner loads the champion and deletes the flag.
- **Kris rejected:** input bruteforce / TAS polishing ("pure AI learning is best"); cross-map base model (parked
  for later); random switching of steering layouts mid-training.

## Speed
- Profiling every 5 min (`TMAI_PROF_S`): per car step TICK / inputs / rewards / waiting-for-action ms, learner
  updates/s, busy %, achieved reuse, tester seconds per test; dashboard card "Where the time goes".
- All telemetry groups requested are used by inputs (SHAPE is not requested). `lidar_every` > 1 opens the gym
  without lidar and calls `g.raycast()` every N steps (an extra round trip, so it only pays if the step's own
  scan is expensive: measure with bench, which compares both).
- PPO keeps SB3's lock-step SubprocVecEnv (its rollout needs it); it has priority + profiling only.

## Testing (no game needed)
```
mkdir /tmp/t && cp tm_ai.py iqn.py config.py /tmp/t && cd /tmp/t
echo '{"map":"x","run":"t","envs":2,"test_every_laps":3,"time_limit_min":1}' > config.json
PYTHONPATH=<repo>/tests python3 -u tm_ai.py train          # PPO; add "algo":"iqn" for IQN
TMAI_IQN_WARMUP=3000 ...                                   # shorter IQN warm-up for tests
```
The fake car drives straight along z for 1000 m (2 checkpoints); finishing ~13.3 s at 30 ms is optimal.
`TICKGYM_FAKE_MS=5` makes each fake step take ~5 ms (random, like a busy TICK) for speed tests;
`TMAI_BENCH_S` / `TMAI_BENCH_CARS` shorten the benchmark. `TICKGYM_FAKE_DETOUR=1` adds a 600 m side loop at z=200 (put a straight `centerline.npy` in the run to test a line
that cuts the AI's route). Always also:
`python -m py_compile *.py` and extract the `<script>` from dashboard.html and `node --check` it.
Test checkpoint conversion by training with the previous git version first, then resuming with the new one.

## Conventions
- Never break running users: config changes go through `coerce`/migration; resume must keep working or move old
  files aside (never load weights under a wrong layout).
- File writes that other processes read are atomic (`write_json`, `save_npy` -> tmp + `os.replace`).
- Errors from TICK (`TransportError`) reconnect and end the lap; file trouble must never stop training.
- Dashboard: no external JS libs, dark theme, labels in plain language with defaults shown.
- Commit messages end with the Co-Authored-By line; the GitHub repo is public (Kristiano55/tm_ai).

## Open ideas / backlog
- Wheel contact material input (check tickgym telemetry fields via selftest).
- Learning-rate warm restart when layout/timing changes (PPO).
- Cross-map base driver with map pool (parked by Kris).
- Run comparison view in the dashboard.
- Fix the ice inputs properly from raw selftest values.
