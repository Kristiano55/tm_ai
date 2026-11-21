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
- Feature freeze: no new features (hub, packs, base, dashboard, new inputs, new rewards, new algorithms) until one
  real map beats its PB. Only fixes, simplification and tuning.

## What it is
An AI that learns to drive Trackmania 2020 maps through **tickgym** (https://tm.achepta.com/tickgym/), a Python
library that runs the game's own deterministic physics via TICK. The same seed + inputs always give the same race.
Gyms can `snapshot()` / `restore()`, cast lidar rays, read telemetry, and `export()` a run as TICK input text.
Nothing is shown in the game window: laps are simulated; results are saved as TICK input files (`*.txt`).

## Files
- `tm_ai.py` — thin entry point: `python tm_ai.py selftest|train|drive|bench|transfer|diagnose` (settings from
  `config.json`), packs/hub `tm_ai.py export|import|publish|verify|benchmark <target>` and `tm_ai.py hub
  list|search|import|refresh|verify-pending`. Re-exports every module below, so `import tm_ai as T` (iqn, packs, hub,
  tests) and the dashboard's commands work unchanged. Dispatches to `iqn.py` when `algo == "iqn"`.
  Import order (no cycles): common <- rewards <- env <- cars <- train_ppo <- cli <- tm_ai (each `from x import *`).
- `common.py` — shared helpers + constants: telemetry groups, EXTRA_NAMES, steering levels / `action_space` /
  `steer_value`, GAS/BRAKE/MODE_VEC, watchdog + `freeze_guard`, surface vocabulary, `say` / JSON / npy / `atomic_save`,
  line geometry (`make_line`, `line_progress`, `trace_to_profile`), `trace_record`, `map_targets`, `build_guide`,
  FAIL_ENDS, `raise_priority`, `time_budget`.
- `rewards.py` — every reward term: `step_terms` (all per-step terms of TMEnv.step, same order + state updates),
  `give_up_cost`, `finish_bonus`, `hit_penalty`, `pointless_release`; reward state: `freeze_preset`, `champion_ghost`,
  `HitRamp`, `Coverage`.
- `env.py` — `TMEnv` (observations, progress, lap rules: stuck / off-zone / far off line / behind ghost / timeout,
  mid-track starts), `SectionTracker`, `n_sections`, `StuckWatch`.
- `cars.py` — `car_worker`, `CarPool` (asynchronous cars: each car process steps as soon as its own action arrives,
  pipes + `connection.wait`), `make_pool_vecenv`, profiling (`prof_summary`, `prof_line`), `bench`.
- `train_ppo.py` — PPO (SB3): `make_model`, `migrate`/`transfer_policy`, **champion tests in a separate spawn process**
  (`ppo_test_worker`, fed EMA weights by the callback; rollback via `ROLLBACK` file), `_make_callback`,
  `prepare_guide`, base runs, `train`, `run_episode`, `replay_champion`, `drive`, `transfer`.
- `cli.py` — `selftest`, `check`, `diagnose` (+ shared `run_trial`), `pack_cli`, `main()`.
- `boot.py` — light entry points of the spawned car / IQN learner processes (report import errors such as 1455).
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
- **Analog steering** (`analog_steer`, default true, PPO only; IQN prints "Analog steering is PPO-only" and keeps
  levels): action = Box [steer -1..1, mode]; `AnalogPolicy` (ppo_ext) + `HybridDist`: Categorical mode x tanh-Gaussian
  steer (log_prob with tanh correction, |a| clamped 0.999 = still 127). Steer mean u = atanh(sum softmax(level logits /
  0.01) x level/127) + `steer_res` (init 0), so action_net keeps the 21+modes layout and old runs convert by copy
  (fake run: identical deterministic lap). Learnable `log_std` from 6/127, capped by the `decay_steps` halving (min
  10 %); upd events log `sstd` (dashboard). Sent/exported steer = round(a x 127) (`steer_value`). levels.json
  `policy` "analog1" (off -> "priv1", converts back). SIL stores float actions (`analog` flag in sil.npz, old
  level indices converted on load). `run_episode` drives the way the model was trained; VecNormalize.load gets the
  current action space; IQN warm start maps an analog champion to the nearest level; packs store `analog` +
  `steer_levels` in model.json.
- **Neutral fresh start** (`make_model(fresh=True)`, new PPO runs only): throttle logits biased `MODE_BIAS`
  (gas 64 %, coast 23 %, brake 9 %, gas+brake 3 %; before: uniform = brake held half the time -> crawl -> "stuck"),
  analog steer level logits exactly 0 (mean = straight) at `steer_t` 1.0 (before: near-random logits / 0.01 = jumps
  between random levels, often full lock). Conversions from levels use `steer_hard` (value = the old argmax level
  exactly, soft-expectation gradient at T 0.1): first actions identical even where old logits are nearly tied.
  Analog runs converted earlier keep soft T 0.01 (`steer_t`/`steer_hard` are policy kwargs; packs store them).
- **Tick-rate change:** levels.json stores `ticks` (else the last start event's cfg); a different ticks_per_step on
  resume resets VecNormalize's return stats (logged). The negative-driving warning names the 3 biggest costs.
- **Lap action stats** (`info["ast"]`, lap events `ast`): share per throttle mode, mean |steer|/127, full-lock share,
  mean speed, furthest progress m, and how the lap ended: `why` (stuck rule: "slow" < 1.5 m/s for 2 s / "no move"
  < 3 m in 2 s / "stall" on the line without progress / "far"), `v_end`, `hits` (wall hits this lap), `off` (m off the
  line). `diag_metrics` (check / diagnose) summarises them for the stuck laps (`stuck_rules`, `stuck_v_end`,
  `stuck_with_wall_hit`).
- **`python tm_ai.py check [map] [--pb file]`** (~3 min, real game): PASS / FAIL per line -> `runs/_check/check.json`
  (exit 1 if anything failed): car accelerates, forward speed, up vector, lidar hits, route/checkpoints (+ dropped start
  block), wheel surface ids (+ INFO: missing telemetry fields), determinism (a scripted drive twice = same finish /
  tick / position), Play champion = champion time +-0.01 s (`replay_champion` of this map's run with a recorded
  champion; SKIP if none), then 120 s fresh PPO (`run_trial` -> runs/check_fresh): laps past checkpoint 1, driving
  reward > 0 (+ break-even speed). `TMAI_CHECK_S` = training seconds (tests: 20).
- **`python tm_ai.py diagnose [map] [--pb file] [--min 4]`** (dashboard "Diagnose PPO"): fresh PPO trials
  `runs/diag_a..d` (deleted before each): a levels+3 modes, b analog+3, c levels+gas+brake, d analog+gas+brake. Each trial: own config file via `TMAI_CONFIG` (config.py
  CONFIG_PATH override, inherited by cars), subprocess `train --new --run diag_x`, STOP after the trial time (counted
  from the first event). Report table + verdict (feature "breaks" = worse in both A/B pairs) -> `runs/_diag/diagnose.json`,
  trial logs `runs/_diag/diag_x.log`. `TMAI_DIAG_S` = trial seconds (tests).
- **IQN layout:** `iqn_gas_brake` (false) 4th mode, `iqn_steer_levels` 21/31/41 (`steer_levels(n)`, same curve; 21 =
  the old set). New IQN runs only: `iqn.layout_cfg` keeps an existing run's layout from iqn.json (`levels`, `modes`;
  older files = 21 x 3), passes `_iqn_levels`/`_iqn_modes` to the cars; `iqn.NM` = modes for `act_index`.
- **Actions (analog_steer off / IQN):** 21 progressive steering levels (fine near centre) x {gas, coast, brake, gas+brake} (`gas_brake`,
  PPO only; IQN keeps 3). `GAS`/`BRAKE` mode sets + `MODE_VEC` (last-action inputs: gas+brake = [1,0,1], layout
  unchanged) everywhere (steps, mid-start replays). Conversion: new logit = brake logit - 3 (same argmax); `ticks_per_step` 5 (50 ms)
  default, 3 (30 ms) possible. Per-step costs scale with `tf = ticks/5` so they mean the same per second.
- **Observations** (564 non-lidar + 41 lidar): car-frame velocity/rotation/up, wheels, steer/gear, last actions, route
  lookahead, 41 extras (effects/runtime/ghost/line offset), track preview (10 pts), short history (5 steps),
  40-point 3D line preview every 10 m (Linesight), 3 older actions, lidar (32 fan + 4 down + 5 up).
  All route points of a step are interpolated in one batch (`_route_points`, `to_car_frame_many`): ~5x faster obs.
  **New features must be appended at the end of the non-lidar block** so `migrate()` / `widen_columns()` can
  convert old PPO checkpoints (new input weights start at 0 = same behaviour). IQN checkpoints don't convert.
- **Input normalisation** (`obs_norm`, default on, PPO + IQN): `ppo_ext.ObsNorm` = buffers mean/var/count/active INSIDE
  the network (PrivPolicy.obs_norm before the features; IQNNet.norm on obs + time-left), so it is saved with the run
  and travels with EMA/tester weights, IQN's flat weight copies, packs (+ surface remap of mean/var) and the IQN warm
  start. (x - mean) / max(std, 0.1), clipped +-5; pass-through (exact raw inputs) until the first update. Models /
  checkpoints / packs from before get identity stats with count = `ObsNorm.PRIOR` (2e5) via `norm_keys_missing`
  (PrivPolicy.load_state_dict override, `iqn.load_net`): identical behaviour on resume (byte-identical probe), stats
  blend in slowly. PPO: rollout obs stashed at rollout end, applied at the next rollout START (stats constant within
  collect + update); EMA copies the stats exactly. IQN: learner updates from the replay batch every `NORM_EVERY` (10)
  updates, tgt gets the same stats.
- **IQN exploitation laps** (`iqn_greedy_share` 0.2 for NEW IQN runs, iqn.json `greedy`; runs from before keep 0):
  that share of training laps (global lap counter, so round-robin over the cars; `next_greedy`) drives pure argmax:
  no epsilon, no Boltzmann (`choose(greedy=...)`). They go into the replay and count for best_run / sections like any
  lap. Lap events carry `greedy: 1` and `rnd` (random / Boltzmann actions this lap, 0 on greedy laps); the 5-min
  profile adds "exploitation laps: N, X % finished, best Y s"; the dashboard lap chart draws them green
  ("No-random laps", `glaps`). Champion tests unchanged.
- **IQN prioritized replay** (`iqn_per`, default on, NEW IQN runs only: iqn.json `per`; runs from before keep
  uniform): replay field `p` (= (|TD| + 1e-3)^0.6, new samples get the max `pm`), stratified sampling over a cumsum
  refreshed every 10 updates (`per_sample`), IS weights (n P)^-beta / max with beta 0.4 -> 1 over 3M steps
  (`PER_BETA`), per-sample quantile loss weighted. Logged at learner start.
- **Mid-start focus** (`start_focus` 0.5): `_pick_start` p = (1 - f)/n + f x mean(normalised fails, normalised
  `tloss`); `tloss` = smoothed seconds lost to the ghost between consecutive start points (`_track_loss`, per car).
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
  - **Start block among the waypoints** ("Better Training - 14": 5873/5873 laps stuck ~8-30 m from the spawn,
    gas 22 %): the start block's waypoint group became "checkpoint 1", so progress was capped right after the spawn
    (fallback route and `cp_arc`), driving on paid nothing and the AI learned to stop. `map_targets()` drops groups
    tagged start (not start/finish) and, on one-lap maps, any group < `START_WP_M` (30 m) from the spawn; `cp_arc`
    now comes from the PB trace's own checkpoint crossings when pb_trace.npy exists. selftest prints the waypoint
    groups (+ tag) and what was dropped. IQN random actions favour gas (`iqn.GAS_P`: 70 % gas). Test:
    `python tests/test_start.py` (fake `TICKGYM_FAKE_STARTWP=<tag>` adds a start group at z=8; fresh PPO + IQN must
    hold gas > 60 % in the first 30 laps and get past the start).
  - **Progress off the target line** (the WR line can take a cut the AI doesn't): the matched arc length is clamped
    between the last taken and next checkpoint (`cp_arc`, waypoints mapped onto the line); > `NEAR_LINE` 15 m off,
    the line is re-acquired anywhere in that section. Progress already paid is `_s`: a skipped stretch is paid out at
    <= 10 m per step when the car rejoins further on, a backward jump costs once (-10 m max).
  - **IQN lap cuts:** the stall rule (on the line, moving, no progress for 4 s) and "far off line" do NOT end IQN laps
    (they cut mini-races short); the car keeps paying the time cost each step. Standing still / < 3 m driven in 2 s
    still ends the lap "stuck" for both algorithms.
  - **Stuck rule:** "<2 m along the line in 2 s" only if the car was within 15 m of the line the whole 2 s, and then
    only after 4 more seconds (`STALL_TICKS`) if the car still moves (line over a gap/wall it can't take); otherwise
    stuck = < 3 m actually driven in 2 s (sum of movements). Speed < 1.5 m/s for 2 s as before. "far off line" ends a
    lap after 8 s continuously > 40 m off. `StuckWatch` logs a hint when >80% of the last 200 laps end stuck at one spot.
  - Laps are cut only against the AI's **own champion** pace (`champ_trace`), never before a champion exists.
  - A dense positive "while driving" reward is required for PPO; dashboard warns when it goes negative.
- **Reward simplification (reward_v 5):** Settings / DEFAULTS only have the minimal set (`config.WEIGHTS`): progress
  0.05, time 0.02, checkpoint 3, finish 50, give-up cost (stuck 20 / off-zone 20 + the remaining metres), plus the
  optional ghost (`ghost_weight`, 0 = off, + `ghost_window_s`) and wall hit (`w_hit`, 0 = off, + `w_hit_ramp`). Break-even
  speed = 20 x w_time / w_progress = 8 m/s at any tick rate (the "driving earns" warning prints it; fake minimal runs
  earn +0.14/step at 50 ms, +0.09 at 30 ms). The reward code still knows every older term (`config.ALL_TERMS`:
  guide, vcp/vsplit, finish_time, steer_smooth, crash, brake, coast, release, slip, wall, air, speed, novelty/known/
  focus), all 0 unless a run has frozen weights. Migration (first `config.load()` of a config with reward_v < 5):
  `freeze_legacy` writes every existing run with a model (runs/<run>, runs/_base/<base>) a `rewards.json` {preset
  "legacy", weights = all terms as that config meant them (`OLD_DEFAULTS` for missing keys) + `vcp_every_s`, `cov_len`,
  `cov_passes`}, then resets the kept weights to the minimal defaults for new runs. A config without reward_v (hand-
  written) counts as current. `common.run_rewards(run_dir)` finds rewards.json in the run folder or a parent (base
  maps, pool maps, transfer tests), used by TMEnv (W, VCP spacing, coverage stretch), live reload, Coverage, IQN's
  time cost, `break_even`. `reward_notice` logs at every start which weights a run uses and which legacy terms are on.
  Old minimal-preset runs keep their rewards.json too. Proof: the byte-identical probe on an old run with 13 legacy
  terms. Deleted (only reachable by removed settings): Settings rows of all legacy terms + vcp/coverage settings, the
  `reward_preset` setting + `freeze_preset` + `MINIMAL` + "Use minimal rewards" button, the `LEAN` preset + button,
  diagnose's novelty trial (e). Kept (old runs need them): all term code in rewards.py, VCPs, `Coverage` (now only
  created when the run's weights use novel/known/focus; `uses_coverage`).
- **Novelty -> speed focus (legacy term, rewards.json only):** `Coverage` in rewards.py, owned by `CarPool`; line cut into
  `cov_len` (25) m stretches; a lap counts stretch k only if it drove through all of it. known = 1 - exp(-passes /
  `cov_passes`). `novel` = progress x (w_novel x (1 - known) - w_known x wks x known); `focus` = ghost x w_focus x known.
- **Reward presets:** dashboard buttons "Linesight preset" and "Reset to minimal" fill the weights. Zero-weight terms
  are left out of the reward breakdown.
- **Lap-time chart markers:** G = ghost rebuilt (event k=target), S = section target switched (k=section); dashboard
  `targets` (lap number when it happened).
- **Own line once close to the PB** (`own_line_s` 2.0, 0 = never; PPO + IQN, runs with a `pb_file`): the target line,
  ghost, VCPs and mid-track starts follow the PB until the champion is within own_line_s of it; then `own_line.json`
  {on} and the WHOLE target is the champion's own lap (`build_own_line` from champ_trace), rebuilt with every new
  champion (`champion_ghost`). `own_line_update` (both testers after each test, and once at training start, so
  existing runs qualify on their next start, e.g. caelid 45.19 vs 44.05): back to the PB line (`build_guide`) when the
  champion is > own_line_s behind the PB (e.g. pb_seconds changed) or on a rollback while the test is slower than
  that, and then only again with a new champion (`blocked_champ`). TMEnv: `has_pb` + `own` (re-read every reset),
  `guide` = has_pb and not own (PB pull, PB mid-starts, guide ghost), mid-starts rebuild on a switch. Section fallback
  is off while on own line. Logged "Now using its own line (champion X s, PB Y s)" / "Back to the PB line (...)",
  events k=target what=own_line / pb_line (G marker), dashboard lap card "Line: its own (...)".
- **IQN horizon beyond the mini-race:** before, a sample whose mini-race ended inside its 3 steps (tau <= k) stopped
  there (no bootstrap): anything after the mini-race was worth 0, so IQN only optimised the next minirace_s (caelid:
  every section at PB pace, never trading "CP a bit later" for more exit speed). Now such a sample uses its full k
  steps and bootstraps from the target net at its final state with a FULL mini-race left (`trunc`, `left2`), so value
  continues past the mini-race, discounted by gamma (0.999/step = ~50 s at 50 ms, -> 1 = whole lap after 2.5M steps;
  logged at start). Failed races unchanged (time cost for the rest of the mini-race). `minirace_s` 1..15. Fake fork
  test (`TICKGYM_FAKE_FORK=1`: tight branch 0.2 s sooner to z=500 but 30 % slower after): last third of 3 min, 61 %
  fast-exit laps vs 48 % with the old target. Lateral offset: progress paid and lap rules are identical with the line
  0 / 15 / 20 / 25 m to the side (no change needed).
- **Section fallback** (`SectionTracker`, PPO + IQN, only with a `pb_file`): per checkpoint section, outcome of
  every lap that reaches it (fail = stuck / off-zone / far off line / behind ghost / timeout), window 200. >70%
  failing (>=50 laps) and a REAL AI route exists (`common.section_route`: sec_best_<k> that starts at CP k and ends on
  CP k+1, or a finished lap with the PB's checkpoint count, and <= `AI_SECTION_MAX_X` (2) x the PB's section time;
  an 84 s "AI route" once gave a 84 s target pace) -> mode "ai"; `build_guide` applies the same rule, else the PB: `build_guide` uses the AI's
  fastest route there (line + ghost + VCPs). Back to "pb" when 3 laps drive the whole section within 15 m of the PB
  route (`secs_wr` in lap info), or a periodic "try" (after 10 min, doubling to 1 h; 50 laps, <=50% fail = keep).
  Every switch is an event (k=section). Hysteresis: after a switch the target is kept >= 10 min (`HOLD_S`), to AI
  needs > 80 % fail, back to PB needs 5 laps on the PB route or a retry with <= 40 % fail; at most one log line per
  section per 10 min. `sec_fallback` setting turns it off.
- **Difficulty input** (`DIFF_COL`, right before the surface block): failure rate of the current section. PPO weights each
  sample's entropy bonus by `1 + 3 * difficulty` (`focus_entropy`, wraps `policy.evaluate_actions`): more
  exploration only where laps keep failing. Old PPO checkpoints are widened by `migrate()`.
- **PB guide:** `pb_file` (TICK input text of Kris's PB) is replayed once to trace *where/when* it drove; it becomes
  the target line + ghost; per checkpoint section the faster of PB vs AI best run is used (`build_guide`). The AI
  never drives the PB inputs (Kris was explicit about this).
- **Virtual checkpoints** every `vcp_every_s` of ghost pace: flat bonus + split vs ghost.
- **Mid-track starts** (snapshots along champion lap / PB) after the first finished lap; WITH a `pb_file` they start
  right away from the PB's states only (PPO + IQN; Shemesh: 12887 IQN laps, none past CP1, so later sections never got
  practice), the champion lap joins once it exists (PPO). IQN uses
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
  the test event logs tries + winning seed, Play champion replays `champion_seed`. **Ghost follows the champion
  only** (`champion_ghost`, called by both testers when the champion improves or no ghost exists; PB-guide runs
  rebuild the guide there): training laps only update best_run.txt/best_trace (no ghost/guide rebuild). Event
  k=target {what: ghost, ms, ghost_ms}. PPO uses EMA weights and
  `target_kl=0.02`; rollback after `rollback_after` bad tests (IQN only if `iqn_rollback`, default off). Both algos test in a spawn process (busy flag,
  restarted if it dies); the tester writes `ROLLBACK`, the learner loads the champion and deletes the flag.
- **PPO is the main algorithm** (Kris, after IQN stalled on Training - 22). IQN code + settings stay; the dashboard
  shows IQN settings only when IQN is chosen. Defaults = recommended PPO setup: minimal rewards, 8 cars, 50 ms,
  mid-starts 0.25 (after the first finish), best-of-3 tests.
- **Kris rejected:** input bruteforce / TAS polishing ("pure AI learning is best");  random switching of steering layouts mid-training.

## Speed
- Profiling every 5 min (`TMAI_PROF_S`): per car step TICK / inputs / rewards / waiting-for-action ms, learner
  updates/s, busy %, achieved reuse, tester seconds per test; dashboard card "Where the time goes".
- All telemetry groups requested are used by inputs (SHAPE is not requested). `lidar_every` > 1 opens the gym
  without lidar and calls `g.raycast()` every N steps (an extra round trip, so it only pays if the step's own
  scan is expensive: measure with bench, which compares both).
- PPO uses `make_pool_vecenv` (lock-step VecEnv on `CarPool`, PPO's rollouts need lock-step) instead of SubprocVecEnv.

## Checkpoints + play
- `atomic_save()`: tmp + rename, previous file kept as `<name>.prev.<ext>` (PPO latest.zip, IQN latest.pt +
  champion.pt). IQN resume: latest.pt, else latest.prev.pt (broken latest.pt -> latest.corrupt.pt), the real
  exception is logged; both broken -> the learner stops with a clear message (never starts fresh silently).
- Play champion (PPO + IQN, `replay_champion`): replays `champion_actions.json` (the test's own lap, now with
  `ticks`) -> play.txt; warns if play differs from champion_ms by > 0.01 s. Re-driving the network did not reproduce
  the time (27.7 vs 20.6 s): its inputs (ghost, line, section rates) changed since the test and IQN samples its
  quantiles with unseeded torch.rand (tests + network fallback now `torch.manual_seed`). Network drive only when no
  recording exists (or `--model`).
- IQN device: `pick_device` test-allocates on the GPU; unavailable/busy/OOM -> CPU with one message (main process
  decides, learner gets `_iqn_dev`; falls back again if moving the net to the GPU fails).
- Log hygiene: mid-start build logged once per rebuild (`starts_logged.json`, cleared at train start); shared
  replay unlinked exactly once and child processes no longer unregister it (Python 3.12 resource_tracker KeyError);
  SB3 schedule deprecation warnings filtered, PPO lr decay uses `ppo_ext.ConstLR`.

## Freezes
- tickgym calls time out (gym 20 s, HTTP 60 s); timeouts end the lap like a `TransportError`.
- **Out of memory (Windows error 1455, paging file too small; also MemoryError):** cars and the IQN learner are started
  through `boot.py` (imports torch/tickgym inside a try, so an import that dies on 1455 is reported as text);
  `tm_ai.py` skips its imports when re-imported as `__mp_main__`. `is_low_memory` on a car / learner error ->
  `LowMemory` -> status "out of memory ...", one message (`LOW_MEM_MSG`: raise Windows virtual memory or use fewer
  cars), exit code 4, no retries. Fake: `TICKGYM_FAKE_FAIL` file containing `1455`.
- **Cars default:** `envs` = min(8, CPU threads - 4) (8 on a 12-thread Ryzen 5 5600). IQN warns in the 5-min profile
  when the achieved reuse < 80 % of `iqn_reuse` (learner can't keep up) and suggests a car count.
- `CarPool`: a car whose process fails (e.g. `TransportError: timed out`, EOF) is replaced like a hung one, its
  lap ends "car restarted"; replacement start-ups are retried; the same car failing 3 times in 5 min stops the run.
- `CarPool`: a car not answering within max(120 s, 3 x max_lap_seconds) (`TMAI_CAR_DEADLINE`) is killed and replaced;
  its lap ends "car restarted". Car start-up waits feed the watchdog.
- PPO: freeze guard max(180 s, 2 x max_lap_seconds + 60); a champion test running longer than
  max tries x (1.3 x max_lap_seconds + 20) + 180 s restarts the tester.
- `watchdog()` re-arms faulthandler (stacks -> `hang.txt`, kept open). `freeze_guard()` thread in the trainer: 180 s
  without `watchdog()` -> dump all threads, status.json "froze", kill children, exit 3. The dashboard restarts runs
  marked "froze" or with a stale heartbeat (pid from status.json), also runs it didn't start; max 5 per hour.

## Testing (no game needed)
```
mkdir /tmp/t && cp *.py /tmp/t && cd /tmp/t                 # all modules (tm_ai.py is split into several)
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
Diagnose test: `TMAI_DIAG_S=25 python tm_ai.py diagnose` (~2.5 min for 4 trials). `TICKGYM_FAKE_FAIL=<file>`: resets raise TransportError while the file exists (content = how many in a row; 2 kills
a car, 100 = keeps failing -> run stops after 3). `TICKGYM_FAKE_WALL=150`: a wall at z=150 wrecks the car (speed 0 for the rest of the lap) unless it is <= 20 m/s there
(hard spot after the start); a PB that brakes for it: "0 accel 1 / 2500 accel 0 / 2500 brake 1 / 3500 brake 0 /
3500 accel 1" (fake input files are in ms) = 14.85 s. `TICKGYM_FAKE_HANG=<file>`: the next car to step after the file appears hangs forever (car replacement test).
Fake surfaces: material 16 (asphalt), dirt (6) for z 600..800, gameplay id 3 for z 300..320, ice 0; the down rays
hit and see the material 20 m ahead. `pb_seconds` 13.8 triggers best-of-8 (fake champion 13.75 s).
`time_limit_min` is an int setting (0.7 -> 0 = no limit). Always also:
`python -m py_compile *.py` and extract the `<script>` from dashboard.html and `node --check` it.
Test checkpoint conversion by training with the previous git version first, then resuming with the new one.
Refactor proof (used for the module split): train a run with every reward term on (PB guide + TICKGYM_FAKE_CORNER),
then on copies of that run drive fixed-seed scripted laps + the champion model with old and new code and compare the
per-step rewards, reward terms and lap times byte for byte. `python3 -m pyflakes` can't see through the star imports:
check that every global a function uses exists in its module (dis LOAD_GLOBAL vs module dict).
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
- After the freeze (Kris: "not yet"): local IQN knowledge library `brain/` (brain add <run> / brain train, curated
  memories of the AI's own driving, physics.json per surface, `start_from_brain` for new IQN runs, dashboard card).
