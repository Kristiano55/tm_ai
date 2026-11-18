"""
Brain packs: a trained PPO driver as a data-only zip that anyone can share (see packcheck.py for the safety rules).

    manifest.json        pack id, author, kind (map | base), code version, input fingerprint, base_id / lineage,
                         maps (UID, name, claimed champion / best times, PB / WR if known), sha256 of every file
    model.json           network layout: observation size, actions, net_arch, steering levels, policy keys
    model.npz            the policy weights as plain float arrays (no pickle)
    vecnorm.json         reward scaling (running mean / variance), so training can continue from the pack
    materials.json       the surface id -> input slot vocabulary the weights were trained with
    maps/<uid>/...       per map: champion.txt / best.txt (TICK input files of the claimed runs),
                         centerline.npy + profile.npy (+ .json) = the line and ghost the driver was trained with

Import never unpickles or runs anything from the pack: weights are read with numpy (allow_pickle=False), checked
against the local layout, the surface one-hot columns are remapped to the local vocabulary, and a fresh local
SB3 model is built and saved (latest.zip, champion.zip, levels.json, vecnorm.pkl are written by this code).
"""
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

import config as cfgmod
import packcheck as pc
import tm_ai as T

CODE_VERSION = "2026.11.19"          # bump when the pack / input format changes; the hub lists compatible versions
PACK_EXT = ".tmbrain.zip"


def input_fingerprint(env_cls=None):
    """Hash of everything that defines what each input column means (not the steering levels: those convert)."""
    E = env_cls or T.TMEnv
    spec = {"n_base0": T.N_BASE0, "extra": list(T.EXTRA_NAMES), "n_mem": E.N_MEM, "n_path": E.N_PATH,
            "n_diff": E.N_DIFF, "n_surf": T.N_SURF, "surf_rays": list(E.SURF_RAYS), "n_ice": E.N_ICE,
            "n_base": E.N_BASE, "look": list(E.LOOK), "preview": list(E.PREVIEW), "path3d": E.PATH3D,
            "lidar": [32, 4, 5], "n_priv": T.N_PRIV, "policy": "priv1"}
    return pc.sha256_bytes(pc.canonical(spec).encode())[:16]


# --- export ------------------------------------------------------------------------------------------------
def _policy_arrays(zip_path):
    from stable_baselines3 import PPO
    m = PPO.load(str(zip_path), device="cpu")                # our own local file
    sd = m.policy.state_dict()
    arrays = {k.replace(".", "__"): v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    analog = not hasattr(m.action_space, "nvec")
    lv = list(getattr(m.policy, "steer_levels", None) or T.STEER_LEVELS["progressive"].tolist())
    nvec = [len(lv), T.n_modes_of(m.action_space)] if analog else [int(x) for x in m.action_space.nvec]
    meta = {"obs_dim": int(m.observation_space.shape[0]), "nvec": nvec, "analog": analog,
            "steer_levels": [int(x) for x in lv],
            "net_arch": {"pi": [256, 256], "vf": [256, 256]}, "keys": list(sd.keys()),
            "num_timesteps": int(m.num_timesteps)}
    return arrays, meta


def _npz_bytes(arrays):
    import io
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()


def _npy_bytes(arr):
    import io
    b = io.BytesIO()
    np.save(b, np.asarray(arr, dtype=np.float64), allow_pickle=False)
    return b.getvalue()


def _map_files(md, uid, files):
    """Per-map files of a run / base map folder; returns the manifest map record."""
    md = Path(md)
    info, best, brun = T.read_json(md / "map_info.json"), T.read_json(md / "best.json"), T.read_json(md / "best_run.json")
    uid = "".join(c if c.isascii() and (c.isalnum() or c in "_-") else "_" for c in str(uid))[:40]
    rec = {"uid": uid, "name": info.get("name") or md.name, "map_sha256": info.get("sha256"),
           "champion_ms": best.get("champion_ms"), "champion_seed": best.get("champion_seed"),
           "best_ms": brun.get("ms"), "pb_ms": None, "wr_ms": None, "runs": {}}
    pbj = T.read_json(md / "pb_trace.json")
    if pbj.get("ms"):
        rec["pb_ms"] = int(pbj["ms"])
    for name, src in (("champion", md / "champion.txt"), ("best", md / "best_run.txt")):
        if src.exists():
            files[f"maps/{uid}/{name}.txt"] = src.read_bytes()
            rec["runs"][name] = {"file": f"maps/{uid}/{name}.txt",
                                 "claimed_ms": rec["champion_ms"] if name == "champion" else rec["best_ms"]}
    for f in ("centerline.npy", "profile.npy"):
        if (md / f).exists():
            files[f"maps/{uid}/{f}"] = _npy_bytes(np.load(md / f, allow_pickle=False))
    for f in ("centerline.json", "profile.json", "sections.json"):
        if (md / f).exists():
            d = T.read_json(md / f)
            if f == "sections.json":                         # only what the inputs use: failure rate per section
                d = {"n": d.get("n"), "rate": d.get("rate"), "mode": d.get("mode")}
            files[f"maps/{uid}/{f}"] = json.dumps(d).encode()
    return rec


def export_pack(cfg, source, out_dir, author="anonymous", wr=None, note=""):
    """source: a run name (runs/<run>) or 'base:<name>' (runs/_base/<name>). Writes <out_dir>/<pack_id>.tmbrain.zip."""
    is_base = source.startswith("base:")
    rd = cfgmod.base_dir(source[5:]) if is_base else cfgmod.ROOT / "runs" / source
    if not rd.exists():
        raise SystemExit(f"No such run folder: {rd}")
    model_zip = rd / "latest.zip" if is_base or not (rd / "champion.zip").exists() else rd / "champion.zip"
    if not model_zip.exists():
        raise SystemExit(f"No model in {rd} (latest.zip / champion.zip)")
    files, maps = {}, []
    arrays, mmeta = _policy_arrays(model_zip)
    lv = T.read_json(rd / "levels.json")
    mmeta.update(steer_levels=lv.get("steer"), obs_base=lv.get("obs_base"))
    if mmeta["obs_base"] != T.TMEnv.N_BASE or lv.get("obs_dim") != T.TMEnv.N_BASE + 41 + T.N_PRIV:
        raise SystemExit("This run's model uses an older input layout: train it once with this version, then export.")
    files["model.npz"] = _npz_bytes(arrays)
    files["model.json"] = json.dumps(mmeta).encode()
    try:
        import pickle
        with open(rd / "vecnorm.pkl", "rb") as fh:            # our own local file
            vn = pickle.load(fh)
        files["vecnorm.json"] = json.dumps({"mean": float(vn.ret_rms.mean), "var": float(vn.ret_rms.var),
                                            "count": float(vn.ret_rms.count)}).encode()
    except Exception:
        pass
    T._vocab_load()
    files["materials.json"] = json.dumps({k: {str(i): s for i, s in T._VOCAB[k].items()}
                                          for k in ("material", "gameplay")}).encode()
    if is_base:
        binfo = T.read_json(rd / "base.json")
        base_id, lineage = binfo.get("base_id"), [binfo.get("base_id")]
        for md in sorted((rd / "maps").glob("*")):
            uid = T.read_json(md / "map_info.json").get("uid")
            if uid:
                maps.append(_map_files(md, uid, files))
    else:
        binfo = T.read_json(rd / "base.json")
        base_id = binfo.get("base_id")
        lineage = [x for x in [base_id] + (T.read_json(rd / "pack.json").get("lineage") or []) if x]
        uid = T.read_json(rd / "map_info.json").get("uid")
        if not uid:
            raise SystemExit("The run has no map_info.json yet (train it once with this version).")
        maps.append(_map_files(rd, uid, files))
    if wr:
        for m in maps:
            m["wr_ms"] = int(float(wr) * 1000)
    pack_id = ("".join(c for c in author.lower() if c.isascii() and (c.isalnum() or c == "-"))[:20] or "anon") \
        + "-" + uuid.uuid4().hex[:8]
    man = {"pack_format": pc.PACK_FORMAT, "pack_id": pack_id, "kind": "base" if is_base else "map",
           "author": author, "created": round(time.time()), "code_version": CODE_VERSION,
           "input_fingerprint": input_fingerprint(), "base_id": base_id, "lineage": lineage,
           "steps": mmeta["num_timesteps"], "maps": maps, "note": note,
           "files": {n: pc.sha256_bytes(b) for n, b in files.items()}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{pack_id}{PACK_EXT}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(man, indent=1))
        for n, b in sorted(files.items()):
            z.writestr(n, b)
    pc.check_zip(path)                                      # never publish something our own check refuses
    return path, man


# --- import (safe) -------------------------------------------------------------------------------------------
class _SpacesEnv:
    """Spaces only, to build a local SB3 model for the pack's weights (no game, nothing else)."""

    def __init__(self, dim, nvec, analog=False):
        import gymnasium as gym
        from gymnasium import spaces

        class E(gym.Env):
            observation_space = spaces.Box(-10.0, 10.0, (dim,), np.float32)
            action_space = T.action_space(nvec, analog)

            def reset(self, *, seed=None, options=None):
                return np.zeros(dim, np.float32), {}

            def step(self, a):
                return np.zeros(dim, np.float32), 0.0, True, False, {}
        self.cls = E


def _remap_surface(w, pack_vocab):
    """First-layer weights [hidden, obs]: move each surface one-hot column from the pack's slot to the local slot
    of the same surface id (unknown locally -> added to the local vocabulary)."""
    E, K = T.TMEnv, T.N_SURF + 1
    c0 = E.DIFF_COL + 1
    groups = [("material", c0 + g * K) for g in range(4)] + [("gameplay", c0 + 4 * K + g * K) for g in range(4)] + \
             [("material", c0 + 8 * K + g * K) for g in range(len(E.SURF_RAYS))]
    out = w.copy()
    for kind, start in groups:
        old = w[:, start:start + K].copy()
        new = np.zeros_like(old)
        new[:, T.N_SURF] = old[:, T.N_SURF]                  # "other" stays "other"
        for sid, pslot in (pack_vocab.get(kind) or {}).items():
            pslot = int(pslot)
            if pslot < T.N_SURF:
                new[:, T.surface_slot(kind, int(sid))] += old[:, pslot]
        out[:, start:start + K] = new
    return out


def load_pack(path, expect_sha256=None, expect_size=None):
    """Checks (packcheck) + compatibility. Returns (manifest, files)."""
    man, files = pc.check_zip(path, expect_sha256, expect_size)
    if man.get("input_fingerprint") != input_fingerprint():
        raise pc.PackError("the pack was trained with a different input layout (input fingerprint "
                           f"{man.get('input_fingerprint')} vs this version's {input_fingerprint()})")
    return man, files


def build_model(man, files):
    """A fresh local PPO model holding the pack's weights (remapped to the local surface vocabulary)."""
    import io
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    mm = json.loads(files["model.json"])
    if mm.get("net_arch") != {"pi": [256, 256], "vf": [256, 256]}:
        raise pc.PackError("unsupported network layout")
    if mm["obs_dim"] != T.TMEnv.N_BASE + 41 + T.N_PRIV:
        raise pc.PackError("observation size does not match this version")
    analog = bool(mm.get("analog"))
    env = DummyVecEnv([_SpacesEnv(mm["obs_dim"], mm["nvec"], analog).cls])
    model = T.make_model(env, mm.get("steer_levels"))
    sd = model.policy.state_dict()
    with np.load(io.BytesIO(files["model.npz"]), allow_pickle=False) as z:
        arrays = {k.replace("__", "."): z[k] for k in z.files}
    if set(arrays) != set(sd):
        raise pc.PackError("the pack's weights do not fit this version's network")
    vocab = json.loads(files["materials.json"]) if "materials.json" in files else {}
    for k, v in arrays.items():
        a = np.asarray(v, dtype=np.float32)
        if a.shape != tuple(sd[k].shape):
            raise pc.PackError(f"weight {k} has shape {a.shape}, expected {tuple(sd[k].shape)}")
        if not np.all(np.isfinite(a)):
            raise pc.PackError(f"weight {k} contains NaN / inf")
        if a.ndim == 2 and a.shape[1] == mm["obs_dim"]:
            a = _remap_surface(a, vocab)
        sd[k] = torch.tensor(a, dtype=sd[k].dtype)
    model.policy.load_state_dict(sd)
    model.num_timesteps = int(mm.get("num_timesteps") or 0)
    return model, env, mm


def import_pack(path, dest, expect_sha256=None, expect_size=None, as_base=False):
    """Safe import into a run folder (dest = runs/<run>) or a base folder. Never overwrites an existing model."""
    man, files = load_pack(path, expect_sha256, expect_size)
    dest = Path(dest)
    if (dest / "latest.zip").exists():
        raise SystemExit(f"{dest} already has a model; import into a new run name.")
    dest.mkdir(parents=True, exist_ok=True)
    model, env, mm = build_model(man, files)
    model.save(str(dest / "latest"))
    if not as_base:
        model.save(str(dest / "champion"))
    T.write_json(dest / "levels.json", {"steer": mm.get("steer_levels") or T.STEER_LEVELS["progressive"].tolist(),
                                        "obs_base": T.TMEnv.N_BASE, "modes": int(mm["nvec"][1]),
                                        "obs_dim": int(mm["obs_dim"]),
                                        "policy": "analog1" if mm.get("analog") else "priv1"})
    if "vecnorm.json" in files:
        from stable_baselines3.common.vec_env import VecNormalize
        v = json.loads(files["vecnorm.json"])
        vn = VecNormalize(env, norm_obs=False, norm_reward=True)
        vn.ret_rms.mean, vn.ret_rms.var, vn.ret_rms.count = float(v["mean"]), float(v["var"]), float(v["count"])
        vn.save(str(dest / "vecnorm.pkl"))                   # written by this code, from plain numbers
    rec = {"pack_id": man["pack_id"], "author": man.get("author"), "sha256": pc.sha256_file(path),
           "lineage": man.get("lineage") or [], "imported": time.time()}
    T.write_json(dest / "pack.json", rec)
    if man.get("base_id"):
        T.write_json(dest / "base.json", {"base_name": f"pack:{man['pack_id']}", "base_id": man["base_id"],
                                          "base_steps": int(mm.get("num_timesteps") or 0), "t": time.time()}
                     if not as_base else {"base_id": man["base_id"], "name": dest.name, "from_pack": man["pack_id"],
                                          "created": time.time()})
    for n, b in files.items():                               # the pack's maps for reference (data only)
        if n.startswith("maps/"):
            p = dest / "pack_maps" / n[5:]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
    return man


def unpack_map(files, uid, md):
    """Line / ghost / sections the pack's driver was trained with -> a local map folder (for verify / benchmark)."""
    md = Path(md)
    md.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        if n.startswith(f"maps/{uid}/") and n.rsplit(".", 1)[-1] in ("npy", "json"):
            (md / n.rsplit("/", 1)[-1]).write_bytes(b)
