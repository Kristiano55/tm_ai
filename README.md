# tm_ai — Trackmania 2020 AI on tickgym

A reinforcement-learning driver for Trackmania 2020. It trains on the game's own physics through [tickgym](https://tm.achepta.com/tickgym/) and TICK, and comes with a browser dashboard to start, stop and watch training.

## Requirements
- Windows with Trackmania 2020
- TICK 3.0, installed and run once with the game
- Python 3.11 or newer

## Setup
1. Download this repo (green **Code** button → **Download ZIP**) and unzip it, e.g. to `Documents\tm-ai`.
2. Double-click **`setup.bat`**. It installs the Python packages, including a CPU-only torch.
3. Open TICK, start its services, and launch Trackmania from TICK.
4. Double-click **`start_dashboard.bat`**. The dashboard opens at http://127.0.0.1:8765.
5. Go to **Settings** and enter the map path (`...\Maps\YourMap.Map.Gbx`), a run name, the number of cars and your PB. Press **Save**, then **Start training**.

Optional check: `python tm_ai.py selftest --map "C:\path\to\map.Map.Gbx"` prints the car's observations and lists any telemetry fields that are missing.

## What it does
- **Two learning algorithms:** PPO, or **IQN** modelled on Linesight (the TMNF AI that beat world records): replay memory, distributional Q-learning, 3-step returns, 7 s mini-races, epsilon/Boltzmann exploration, champion tests in a background process. Pick it under *Learning algorithm* and use the *Linesight preset* for the rewards.
- **Observation (~290 values):** 40 line points ahead in 3D (400 m), last 5 actions, recent slide/rotation history, car-frame velocity, wheels, route lookahead, ghost pace, offset from the racing line, effects, and 41 lidar rays.
- **Actions:** 21 progressive steering levels × gas, coast or brake, deciding every 50 ms.
- **Champion tests:** every 50 laps the AI drives one lap without exploration. If it's faster, it becomes the new champion (`champion.zip` + `champion.txt`); if training gets worse, it rolls back to the champion.
- **Ghost and racing line:** reward for time gained on the fastest run so far. The racing line is rebuilt from each new champion.
- **Mid-track starts:** most training laps start from snapshots along the champion's lap, with more weight on the sections where laps fail.
- **Target line from your PB:** set *Your PB as a TICK input file* in Settings. The run is played once in the simulation to trace where it drove and when; that becomes the target line and pace. The AI is pulled toward it only where it is slower, and any checkpoint section the AI drives faster replaces yours, so it can find its own lines and strats.
- **Stable training:** limited policy step size (target_kl) and champion tests on smoothed weights; slow laps are cut against the champion's pace.
- **Pointless release penalty:** releasing the throttle on a clear straight is penalised; in corners it's free.
- **Reward weights** can be changed live from the dashboard.

## Output (runs/&lt;run name&gt;/)
- `champion.txt`: the best clean lap, as a TICK input file (load it in TICK to watch)
- `best_run.txt`: the fastest finished lap of any kind
- `play.txt`: the result of **Play champion**

Each map should get its own run name.
