"""
Entry points of the spawned car / learner processes. They import the heavy modules (torch, SB3, tickgym) INSIDE a
try, so an import that fails on Windows (e.g. "[WinError 1455] The paging file is too small" while torch loads its
CUDA DLLs) is reported back to the main process as text instead of a silently dead process.
This module itself imports nothing heavy (tm_ai.py also skips its imports when it is re-imported as __mp_main__).
"""
import traceback


def car(cfg, run_dir, conn, rot=None):
    try:
        from cars import car_worker
    except BaseException as e:                      # noqa: B036 - report anything, the process ends anyway
        try:
            conn.send(("error", f"{type(e).__name__}: {str(e)[:200]}"))
        except Exception:
            pass
        return
    car_worker(cfg, run_dir, conn, rot)


def learner(*args):
    errq = args[-1]
    try:
        from iqn import learner_proc
    except BaseException as e:                      # noqa: B036
        try:
            errq.put(f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-1000:])
            args[-2].set()                          # ready_evt: the main process stops waiting
        except Exception:
            pass
        return
    learner_proc(*args)
