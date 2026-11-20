"""
Trackmania 2020 RL agent on tickgym (PPO, runs on CPU).

    python tm_ai.py selftest --map "C:\\Maps\\X.Map.Gbx"
    python tm_ai.py train                (settings come from config.json / the dashboard)
    python tm_ai.py drive                (plays runs/<run>/champion.zip, writes play.txt for TICK)

Normally you start/stop all of this from the dashboard: python dashboard.py
"""
# The code lives in common.py, rewards.py, env.py, cars.py, train_ppo.py and cli.py; this file is the entry point and
# re-exports everything, so `import tm_ai as T` (iqn, packs, hub, tests) and the dashboard's commands keep working.
if __name__ != "__mp_main__":   # a spawned car / learner re-imports this file as __mp_main__: skip the heavy imports
    from common import *  # noqa: F401,F403
    from common import _f, _cp_cuts, _wp_kind, _vocab_load, _VOCAB, _hang_file, _HANG, _P  # noqa: F401
    from rewards import *  # noqa: F401,F403
    from env import *  # noqa: F401,F403
    from env import _make_env  # noqa: F401
    from cars import *  # noqa: F401,F403
    from train_ppo import *  # noqa: F401,F403
    from train_ppo import _make_callback  # noqa: F401
    from cli import *  # noqa: F401,F403
    from cli import main

if __name__ == "__main__":
    main()
