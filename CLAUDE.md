# tm_ai — Trackmania 2020 RL agent on tickgym

Owner: Kris (Kristiano55). Windows PC, Ryzen 5 5600, Python 3.12, TICK 3.0 at `%LOCALAPPDATA%\TICK`.
Install folder on his PC: `C:\Users\Kris\Documents\tm-ai`. Maps: `C:\Users\Kris\Documents\Trackmania\Maps\...`.
He is a TM2020 TAS / SpeedTAS player, not an ML engineer: explain changes in Trackmania terms, keep replies short.

## Standing rules for every task
- Existing runs must convert and resume; new inputs/actions start neutral (same first actions after conversion).
- PPO is primary; IQN must keep running (skip unsupported features there with a clear message).
- Pure AI learning: never imitate PB/WR/pack inputs, never bruteforce.
- Test only what the change touches on the fake tickgym, plus one short PPO resume smoke test; no screenshots
  unless the UI changed; read only the code you need.
- Finish with: update CLAUDE.md if behaviour changed, commit, push to main.
- Report in at most 10 lines: what changed, settings added (name + default), tests run, anything not done.

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
  `python tm_ai.py selftest|train|drive|bench|transfer` (settings from `config.json`); packs/hub:
  `tm_ai.py export|import|publish|verify|benchmark <target>` and `tm_ai.py hub list|search|import|refresh|verify-pending`. Dispatches to `iqn.py` when `algo == "iqn"`.
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
- `ppo_ext.py` — `PrivPolicy` (asymmetric critic: last `N_PRIV` inputs masked to 0 for the actor; `aux_head`),
  `TMPPO` (train() = normal PPO update, then 4 extra steps of self-imitation + aux losses), `SILBuffer`, `aux_labels`.
  Its own module so saved models reference `ppo_ext.*` (not `__main__`).
- `packcheck.py` — pack safety checks (stdlib + numpy; byte-identical copy in `hub_template/tools/`).
- `packs.py` — brain packs: `export_pack` (run or `base:NAME` -> data-only zip), `import_pack` (safe: never unpickles
  or runs pack content; rebuilds a local SB3 model, remaps surface one-hots to the local vocabulary), `CODE_VERSION`,
  `input_fingerprint()`.
- `hub.py` — hub client: cached index, list/search/import by id, publish, verify, benchmark, verify-pending.
- `hub_template/` — the hub repo (to be pushed as Kristiano55/tm_ai-brains): schema, Actions, tools, docs.
- `config.py` — `DEFAULTS`, `WEIGHTS` (reward weights: key -> (default, label, hint)), presets `LEAN`, `LINESIGHT`,
  `load()/coerce()/save()` (atomic), `reward_v` migration of old configs, `run_dir()`.
- `dashboard.py` + `dashboard.html` — stdlib HTTP server on 127.0.0.1:8765; starts/stops `tm_ai.py` as a subprocess,
  reads `runs/<run>/events.jsonl` incrementally; vanilla JS + canvas charts (no libraries). Has a frozen-training
  guard (restarts training if heartbeat is stale 150 s).
- `setup.bat` (pip install, CPU torch), `start_dashboard.bat`, `requirements.txt`.
- `tests/tickgym.py` — **fake tickgym** for testing without Windows/TICK/the game (see Testing).

## Base driver (PPO only) `runs/_base/<base_name>/`
- `run_type` "base" + `map_pool` ([{map, pb_file, pb_seconds}], dashboard: one line per map "path | PB file | PB s").
  Base folder: latest.zip, vecnorm.pkl (one shared reward scaling), levels.json, tester_base.zip, events.jsonl (laps +
  tests tagged `map`), status.json, `base.json` (`base_id` uuid, maps). Per map `maps/<slug>/` (`map_slug` = file name +
  path hash): everything of a normal run folder (line, ghost, PB trace, mid-starts, best.json + champion.zip/.txt,
  sections, best runs). `setup_cars`: car i -> map i (`CarPool` specs); with more maps than cars each car rotates
  itself (`car_worker` rot: + `envs` maps every `pool_rotate_laps` laps). Tester: one map per test, round-robin, no
  rollback; trackers + stuck hints per map. Play champion = first pool map's champion.
- Map-neutral progress: `_pnorm` = `progress_norm_m` (1000) in base runs and runs started from a base:
  `pscale = clip(1000 / line length, 0.2, 5)` on progress + give-up cost. Finish/checkpoint bonuses unchanged. Inputs are
  map-neutral (all car-frame / relative / per-lap fractions / shared surface vocabulary; no absolute positions or race time).
- New run from base: `base_from` in a map run without latest.zip -> copies latest.zip + vecnorm.pkl + levels.json
  (`migrate()` converts an older layout), writes `runs/<run>/base.json` {base_name, base_id, base_steps}.
- No forgetting: `pool_cars` (default 1) cars of a map run drive the pool maps (`runs/<run>/pool/<slug>/`, line/ghost
  seeded from the base's map folder); their laps carry `pool: 1` and are ignored by stats, trackers, dashboard.
- Transfer score: `python tm_ai.py transfer --map <path> [--pb file]` (dashboard button): base latest.zip, best of 3, no
  learning, folder `transfer/<slug>/`; result in `transfers.json` (time or end reason, checkpoint, x/z, share of lap).

## Brain packs + hub (data only, nothing from a pack is ever executed)
- Pack zip `<pack_id>.tmbrain.zip`: manifest.json (pack_format 1, pack_id `<author>-<8 hex>`, kind map|base, code
  version, input fingerprint, base_id, lineage, maps by UID with claimed champion/best times + PB/WR, sha256 of every
  file), model.json + model.npz (policy weights, plain float arrays), vecnorm.json, materials.json, per map
  `maps/<uid>/` champion.txt + best.txt (TICK input files of the claims), centerline/profile .npy/.json, sections.json.
- packcheck rules: only .json .txt .npz .npy .md; no absolute paths / `..` / symlinks / encryption; pickle, exe, ELF,
  script and zip magic rejected by content; npy/npz loaded with allow_pickle=False, no object dtype; size, member
  count and compression-ratio limits; every file listed in the manifest with matching sha256.
- Map identity: `session.info.map.map_uid`; TMEnv writes `map_info.json` per run and `runs/_maps.json` (uid -> path),
  used by verify/benchmark (also `--map UID=PATH`, or the `maps_folder` setting by file name, confirmed by UID).
- verify: replay each claimed TICK input file (`trace_record`, exact ms must match), then the pack's driver best-of-3
  with seeds None/1000/1001 on the pack's line/ghost; verdict verified only if all replays match and it finishes.
  Output `runs/_hub/verifications/<id>.json` with `digest` (sha256 of canonical JSON) + `signed_off`; opens GitHub's
  pre-filled new-file page `verifications/<id>/<user>.json`. `--benchmark` adds the benchmark score (in the digest).
- benchmark: hub benchmark.json maps (uid, name, par_ms); per map 100 x par/time (max 150), no finish 50 x share of
  lap, missing map 0; score = average over all maps.
- publish: export + sha256/size + `packs/<id>.json` entry (status unverified) -> opens the release page of the user's
  fork (tag `pack-<id>`, attach the zip) and the pre-filled new-file page (GitHub makes the PR). With `github_token`
  the release asset is uploaded via the API. The dashboard masks the token ("********" keeps the saved one).
- Hub repo: one PR = one file (`packs/<id>.json` or `verifications/<id>/<user>.json`); `validate.yml` runs
  `tools/validate_pr.py` from main (pull_request_target; PR files read as data only). `build.yml` runs
  `tools/build_index.py` -> index.json (committed) + Pages leaderboard. Status: verified/failed only from a
  verifiers.txt user's file with matching sha256 + digest; quarantined by an open `report` issue "Report: <id>" newer
  than the last verification; votes = 👍 on a `vote` issue "Vote: <id>". compat.json lists accepted fingerprints:
  ADD the new fingerprint there whenever the input layout changes (and bump CODE_VERSION).
- Dashboard Hub tab: cached index (`runs/_hub/index.json`, Refresh), filters map / verified / same lineage / min
  benchmark, sort verified / vs WR / benchmark / votes, Import (runs `tm_ai.py hub import`), Report (pre-filled
  issue), "Verify pending packs" for verifiers. Hub jobs log to `runs/_hub/hub.log`.
- Pack combine (merging packs) is NOT implemented here (the brain-pack prompt was never pushed to this repo).

## Runs never reset by accident
- Run names: `config.run_key()` ignores spaces, '-', '_' and case; `run_dir()` -> `find_run()` maps the name to an
  existing folder with the same key (exact name wins, else the most recently trained; colliding folders are kept and
  `RUN_WARNINGS` is logged + shown on the dashboard).
- `start_conflict(cfg)`: the chosen run has no model but another run on the same map (UID via `runs/_maps.json`, or
  path) has one -> CLI `train` refuses (exit 2, lists the runs) unless `--new` or `--run NAME`; dashboard /api/start
  answers 409 + choice -> modal "Continue <run>" / "Start a new run from scratch" (`--new`). The freeze-guard restart
  passes `--new` (same run). Settings: run name = combo box (datalist of `list_runs()`: map, champion, steps, last
  trained); changing the map pre-selects its newest run.
- Resume: latest.zip, else champion.zip (missing or unloadable latest.zip) with a loud notice; Resume off / layout
  conversion failed / unloadable model -> "STARTING FROM SCRATCH because ..." in the log, in the start event
  (`notice`) and as a dashboard banner. IQN: champion.pt when latest.pt is missing.

## Run folder `runs/<run>/`
`events.jsonl` (k = start | lap | test | upd | rb | prof | stop; prof = 5-min "where the time goes"), `status.json` (heartbeat), `train.log`, `best.json`
(champion_ms, first_test_ms, champion_seed), `champion.zip|.pt`, `champion.txt` (TICK input), `best_run.txt/.json`
(fastest finished full lap of any kind), `last_test.txt`, `play.txt`, `centerline.npy/.json` (reference line),
`profile.npy/.json` (ghost: arc length -> race tick), `sections.json` (per-section fail outcomes, rate, target mode),
`sec_best_<k>.npy` (fastest drive through section k, any lap), `hit_ramp.json` (wall-hit ramp factor), `hang.txt` (stack dumps of hangs), `champ_trace.npy`, `champion_actions.json` (for mid-starts),
`pb_trace.npy/.json` (traced PB), `best_trace.npy`, `latest.zip|.pt`, `vecnorm.pkl` (PPO reward scaling),
`bench.json` (car-count benchmark), `levels.json` (steering levels + obs layout, for checkpoint conversion), `tester_base.zip` (PPO network layout
  for the test process, rewritten each start), `STOP` / `ROLLBACK` flag files.

## Key design decisions (and why)
- **Actions:** 21 progressive steering levels (fine near centre) x {gas, coast, brake, gas+brake} (`gas_brake`,
  PPO only; IQN keeps 3). `GAS`/`BRAKE` mode sets + `MODE_VEC` (last-action inputs: gas+brake = [1,0,1], layout
  unchanged) everywhere (steps, mid-start replays). Conversion: new logit = brake logit - 3 (same argmax); `ticks_per_step` 5 (50 ms)
  default, 3 (30 ms) possible. Per-step costs scale with `tf = ticks/5` so they mean the same per second.
- **Observations** (564 non-lidar + 41 lidar): car-frame velocity/rotation/up, wheels, steer/gear, last actions, route
  lookahead, 41 extras (effects/runtime/ghost/line offset), track preview (10 pts), short history (5 steps),
  40-point 3D line preview every 10 m (Linesight), 3 older actions, lidar (32 fan + 4 down + 5 up).
  All route points of a step are interpolated in one batch (`_route_points`, `to_car_frame_many`): ~5x faster obs.
  **New features must be appended at the end of the non-lidar block** so `migrate()` / `widen_columns()` can
  convert old PPO checkpoints (new input weights start at 0 = same behaviour). IQN checkpoints don't convert.
- **Privileged critic inputs** (PPO, `N_PRIV` = 6, AFTER the lidar): ghost time left, PB time left, m to next CP, m to
  finish, ghost gap, section failure rate. Actor sees 0 (`MaskedMlpExtractor`), so first actions stay identical.
- **Self-imitation** (`sil_weight` 0.1): top-20 finished full laps per map (obs f16, action, RAW reward) in
  `sil.npz` (run folder, never in packs). Returns use the current VecNormalize scale + per-step clip like PPO;
  loss -logp x max(R - V, 0) + 0.005 x adv^2. Being active on ~all samples early is normal (laps beat V).
- **Aux heads** (`aux_weight` 0.05, training only, on the actor latent): wall hit within 0.5 s, wheels grounded
  within 0.3 s (BCE), seconds to next CP /10 (MSE). Labels from `info["aux"]` per step (callback -> `aux_rows`).
  upd events carry `aux`, `sil`, `sil_act`, `sil_n`; shown under the value-loss card.
- `levels.json` also stores `modes`, `obs_dim`, `policy` ("priv1"); any difference -> `migrate()` converts.
- **Old `icing` / `on_ice` extras slots stay 0** (trained checkpoints have untrained weights there). Real ice inputs
  are in the surface block: per wheel `{fl,fr,rl,rr}_icing_state` + `_icing_blend`, front/rear icing grip scale - 1,
  `is_any_wheel_on_ice_or_snow`. selftest prints the raw values (summer map: 0, grip scales 1).
- **Surface block** (`_surface`, after the difficulty input; `N_SURF_IN` = 317): one-hot per wheel of
  `wheels.material_id` and of RUNTIME `{w}_surface_gameplay_id` (4x17 each), lidar `material_id` of the 4 down
  rays + 6 centre fan rays (`SURF_RAYS`, 10x17, zero when no hit), ice (11). Gameplay + ice need extra_obs.
- **Surface vocabulary** `runs/_materials.json` (shared by ALL runs, later base model / packs): separate
  "material" and "gameplay" maps id -> slot in order of first appearance, 16 slots + 16 = other; new ids are added
  under a lock file (`_materials.lock`) and logged. Never reorder or edit it.
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
  - **Stuck rule:** "<2 m along the line in 2 s" only if the car was within 15 m of the line the whole 2 s, and then
    only after 4 more seconds (`STALL_TICKS`) if the car still moves (line over a gap/wall it can't take); otherwise
    stuck = < 3 m actually driven in 2 s (sum of movements). Speed < 1.5 m/s for 2 s as before. "far off line" ends a
    lap after 8 s continuously > 40 m off. `StuckWatch` logs a hint when >80% of the last 200 laps end stuck at one spot.
  - Laps are cut only against the AI's **own champion** pace (`champ_trace`), never before a champion exists.
  - A dense positive "while driving" reward is required for PPO; dashboard warns when it goes negative.
- **Section fallback** (`SectionTracker`, PPO + IQN, only with a `pb_file`): per checkpoint section, outcome of
  every lap that reaches it (fail = stuck / off-zone / far off line / behind ghost / timeout), window 200. >70%
  failing (>=50 laps) and an AI route exists (`sec_best_<k>` or best lap) -> mode "ai": `build_guide` uses the AI's
  fastest route there (line + ghost + VCPs). Back to "pb" when 3 laps drive the whole section within 15 m of the PB
  route (`secs_wr` in lap info), or a periodic "try" (after 10 min, doubling to 1 h; 50 laps, <=50% fail = keep).
  Every switch is logged (event k=section). `sec_fallback` setting turns it off.
- **Difficulty input** (`DIFF_COL`, right before the surface block): failure rate of the current section. PPO weights each
  sample's entropy bonus by `1 + 3 * difficulty` (`focus_entropy`, wraps `policy.evaluate_actions`): more
  exploration only where laps keep failing. Old PPO checkpoints are widened by `migrate()`.
- **PB guide:** `pb_file` (TICK input text of Kris's PB) is replayed once to trace *where/when* it drove; it becomes
  the target line + ghost; per checkpoint section the faster of PB vs AI best run is used (`build_guide`). The AI
  never drives the PB inputs (Kris was explicit about this).
- **Virtual checkpoints** every `vcp_every_s` of ghost pace: flat bonus + split vs ghost.
- **Mid-track starts** (snapshots along champion lap / PB) only after the first finished lap; IQN uses
  `iqn_mid_start_p` (0.2) and, with a `pb_file`, starts only along the PB/WR states (to learn the WR route's cut); built BEFORE
  `g.reset()` in `reset()` (an earlier bug made laps start at the finish).
- **Champion tests only after the first finished full lap** of the run (`tests_ready()`: best_run.json / champion;
  base: per map, the tester skips pool maps never finished; IQN too). Dashboard: "Champion tests start after ...".
- **Wall-hit penalty** (`_hit_penalty`): w_hit x min(1, speed lost / 10 m/s) x ramp factor. `HitRamp` (setting
  `w_hit_ramp`, default on): finish rate of the last 100 full laps < 30 % -> 25 %, linear to 100 % at 60 %; once 60 %
  is reached (>= 20 laps) it stays 100 % (`reached`). Per run folder (per map in base runs); event k=hit_ramp on
  changes; shown in the reward breakdown card.
- **Champion tests:** best of `test_tries` (argmax + seeded sampling, reproducible), `test_tries_close` (8) once the
  champion is within `test_close_s` (0.5 s) of the PB (`pb_seconds`, else the traced PB file's time; `test_tries()`);
  the test event logs tries + winning seed, Play champion replays `champion_seed`. Ghost: fastest finished lap (tests
  included); after a line rebuild `_ghost_from_best()` keeps the best saved run if it beats the champion lap; PPO uses EMA weights and
  `target_kl=0.02`; rollback after `rollback_after` bad tests (IQN only if `iqn_rollback`, default off). Both algos test in a spawn process (busy flag,
  restarted if it dies); the tester writes `ROLLBACK`, the learner loads the champion and deletes the flag.
- **PPO is the main algorithm** (Kris, after IQN stalled on Training - 22). IQN code + settings stay; the dashboard
  shows IQN settings only when IQN is chosen. Defaults = recommended PPO setup: Lean weights, 8 cars, 50 ms,
  mid-starts 0.25 (after the first finish), best-of-3 tests.
- **Kris rejected:** input bruteforce / TAS polishing ("pure AI learning is best");  random switching of steering layouts mid-training.

## Speed
- Profiling every 5 min (`TMAI_PROF_S`): per car step TICK / inputs / rewards / waiting-for-action ms, learner
  updates/s, busy %, achieved reuse, tester seconds per test; dashboard card "Where the time goes".
- All telemetry groups requested are used by inputs (SHAPE is not requested). `lidar_every` > 1 opens the gym
  without lidar and calls `g.raycast()` every N steps (an extra round trip, so it only pays if the step's own
  scan is expensive: measure with bench, which compares both).
- PPO uses `make_pool_vecenv` (lock-step VecEnv on `CarPool`, PPO's rollouts need lock-step) instead of SubprocVecEnv.

## Freezes
- tickgym calls time out (gym 20 s, HTTP 60 s); timeouts end the lap like a `TransportError`.
- `CarPool`: a car not answering within max(120 s, 3 x max_lap_seconds) (`TMAI_CAR_DEADLINE`) is killed and replaced;
  its lap ends "car restarted". Car start-up waits feed the watchdog.
- PPO: freeze guard max(180 s, 2 x max_lap_seconds + 60); a champion test running longer than
  max tries x (1.3 x max_lap_seconds + 20) + 180 s restarts the tester.
- `watchdog()` re-arms faulthandler (stacks -> `hang.txt`, kept open). `freeze_guard()` thread in the trainer: 180 s
  without `watchdog()` -> dump all threads, status.json "froze", kill children, exit 3. The dashboard restarts runs
  marked "froze" or with a stale heartbeat (pid from status.json), also runs it didn't start; max 5 per hour.

## Testing (no game needed)
```
mkdir /tmp/t && cp tm_ai.py iqn.py config.py /tmp/t && cd /tmp/t
echo '{"map":"x","run":"t","envs":2,"test_every_laps":3,"time_limit_min":1}' > config.json
PYTHONPATH=<repo>/tests python3 -u tm_ai.py train          # PPO; add "algo":"iqn" for IQN
TMAI_IQN_WARMUP=3000 ...                                   # shorter IQN warm-up for tests
```
The fake car drives straight along z for 1000 m (2 checkpoints); finishing ~13.3 s at 30 ms is optimal.
Fake map names `fake:<length>:<surface>` (asphalt 80 m/s / dirt 50 / grass 45, own material ids) for base tests.
`TICKGYM_FAKE_MS=5` makes each fake step take ~5 ms (random, like a busy TICK) for speed tests;
`TMAI_BENCH_S` / `TMAI_BENCH_CARS` shorten the benchmark. `TICKGYM_FAKE_DETOUR=1` adds a 620 m side loop at z=200 (just doable inside the 8 s far-off limit at full gas);
`TICKGYM_FAKE_DETOUR=<file>` only while that file exists. Section fallback test: trace `pb.txt` ("0 accel 1") with
`prepare_guide` WITHOUT the detour (straight "WR" line), then train with the detour, then delete the file.
`TICKGYM_FAKE_CORNER=1`: z 400..460 caps speed at 35 m/s unless gas AND brake are held (60). The fake brake now
slows the car when held without gas (random IQN driving rarely finishes in 1 min there; with the old fake it does).
`TICKGYM_FAKE_HANG=<file>`: the next car to step after the file appears hangs forever (car replacement test).
Fake surfaces: material 16 (asphalt), dirt (6) for z 600..800, gameplay id 3 for z 300..320, ice 0; the down rays
hit and see the material 20 m ahead. `pb_seconds` 13.8 triggers best-of-8 (fake champion 13.75 s).
`time_limit_min` is an int setting (0.7 -> 0 = no limit). Always also:
`python -m py_compile *.py` and extract the `<script>` from dashboard.html and `node --check` it.
Test checkpoint conversion by training with the previous git version first, then resuming with the new one.
Packs + hub: `python tests/test_hub.py` (temp copy; safety checks, PR validator, index builder, file:// import,
verify incl. a faked time, benchmark). The fake game records real inputs (`export()` / `InputText`) and has map UIDs.

## Conventions
- Never break running users: config changes go through `coerce`/migration; resume must keep working or move old
  files aside (never load weights under a wrong layout).
- File writes that other processes read are atomic (`write_json`, `save_npy` -> tmp + `os.replace`).
- Errors from TICK (`TransportError`) reconnect and end the lap; file trouble must never stop training.
- Dashboard: no external JS libs, dark theme, labels in plain language with defaults shown.
- Settings page is driven by ONE table in dashboard.html: `F` (key -> [short label, help, type, {min,max,step,opts,
  more,live,wide,group,iqn}]) and `SECTIONS` (order of sections + their keys). A new config key needs an entry in
  both (weights: type 'w', help comes from config.WEIGHTS). Help = "?" tooltip (with default + live/at start);
  `more` = under "More"; `live` = picked up by running cars (weights, ghost window, cut-behind). Search, changed-
  from-default dots, Reset section, validation, the sticky Save/Discard bar and the leave-with-unsaved banner all
  read that table. /api/state sends `defaults`. Dashboard cards collapse (localStorage `tmai.closed`).
  Check: every DEFAULTS key once in F and in one section; Save round trip (see the test notes in git history).
- Commit messages end with the Co-Authored-By line; the GitHub repo is public (Kristiano55/tm_ai).

## Open ideas / backlog
- Wheel contact material input (check tickgym telemetry fields via selftest).
- Learning-rate warm restart when layout/timing changes (PPO).
- Base: lap-time chart mixes maps in base runs (per-map table is the real view).
- Run comparison view in the dashboard.
- Check selftest's raw ice values on a real ice/snow map.
