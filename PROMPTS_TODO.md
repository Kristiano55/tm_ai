# TM20 AI — prompts status

Updated 2026-11-20. **Rule: fix issues, no new features until one real map beats its PB.**

## Done (in git)
- Log fixes · PPO real-map fixes + `diagnose` · novelty reward · minimal reward preset ·
  ghost only on new champion · stuck-after-spawn fix · code split into modules
- Input normalisation, IQN prioritized replay (`iqn_per`), focused mid-track starts (`start_focus`)
- Feature freeze rule in CLAUDE.md
- Rewards cut to the minimal set (+ optional ghost, wall hit); legacy terms only for old runs
- IQN fixes: no lap cuts inside mini-races, reuse warning, default cars from thread count,
  clean stop on Windows error 1455
- `python tm_ai.py check <map>` real-game check

## Next: no prompts — test on the real game
1. `python tm_ai.py check "<map path>"` → all lines PASS?
2. New run (new name), IQN, default cars, minimal rewards, leave it 1–2 h.
3. Send: best lap vs PB, % laps stuck and where, driving-earns value.
Then decide the next fix from the real numbers.

## LATER (features — only after a map beats its PB)
- IQN as a `diagnose` trial
- 30 ms "new run from champion" button
- LSTM memory for PPO
