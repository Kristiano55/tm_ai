"""
Brain packs + hub tests (no game: uses the fake tickgym in this folder).

    python tests/test_hub.py

Works in a throw-away copy of the code (temp folder), so nothing in your runs/ is touched. Covers: pack safety
checks (good pack, tampered hash, path traversal, pickles), the hub's pull-request validator, the index builder
(verifiers only, quarantine on reports), hub list/import through a file:// index, and verify (claimed time replays
exactly; a faked time is rejected).
"""
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="tmai_hub_"))
for f in ("tm_ai.py", "config.py", "packs.py", "packcheck.py", "hub.py", "iqn.py", "ppo_ext.py"):
    shutil.copy(REPO / f, TMP / f)
shutil.copytree(REPO / "hub_template", TMP / "hub_template")
sys.path[:0] = [str(TMP), str(REPO / "tests")]
(TMP / "config.json").write_text(json.dumps({"map": "fake:1000:asphalt", "run": "fx", "github_user": "kris"}))

import numpy as np  # noqa: E402
import config as C  # noqa: E402
import hub  # noqa: E402
import packcheck as pc  # noqa: E402
import packs as P  # noqa: E402
import tm_ai as T  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def expect_error(name, fn, text=""):
    try:
        fn()
    except (pc.PackError, SystemExit) as e:
        check(name, text.lower() in str(e).lower(), f"error was: {e}")
        return str(e)
    check(name, False, "no error raised")


def make_fixture():
    """A run with a champion lap (full gas, recorded by the fake game) and a model that always drives full gas."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    cfg = C.load()
    rd = C.run_dir(cfg)
    env = T.TMEnv(cfg, rd, shorten=False, role="champion test")
    env.reset()
    while True:
        _, _, term, trunc, info = env.step(np.array([10, 0]))
        if term or trunc:
            break
    rec = env.g.export()
    T.InputText.write(str(rd / "champion.txt"), rec.actions)
    shutil.copy(rd / "champion.txt", rd / "best_run.txt")
    T.write_json(rd / "best.json", {"champion_ms": int(rec.finish_ms), "champion_seed": None})
    T.write_json(rd / "best_run.json", {"ms": int(rec.finish_ms)})
    dim = env.observation_space.shape[0]
    venv = DummyVecEnv([P._SpacesEnv(dim, [21, 3]).cls])
    model = T.make_model(venv)
    import torch
    with torch.no_grad():
        b = model.policy.action_net.bias
        b.zero_()
        b[10] = 30.0                                       # straight
        b[21] = 30.0                                       # gas
        model.policy.action_net.weight.zero_()
    model.save(str(rd / "champion"))
    model.save(str(rd / "latest"))
    VecNormalize(venv, norm_obs=False).save(str(rd / "vecnorm.pkl"))
    T.write_json(rd / "levels.json", {"steer": T.STEER_LEVELS["progressive"].tolist(), "obs_base": T.TMEnv.N_BASE,
                                      "modes": 3, "obs_dim": dim, "policy": "priv1"})
    env.close()
    return cfg, int(rec.finish_ms)


def rezip(src, dst, change=None, add=None, drop_hash=False):
    """Copy a pack, optionally changing members (and fixing the manifest hashes, like a careful attacker)."""
    zin = zipfile.ZipFile(src)
    files = {n: zin.read(n) for n in zin.namelist()}
    for n, b in (change or {}).items():
        files[n] = b
    for n, b in (add or {}).items():
        files[n] = b
    man = json.loads(files["manifest.json"])
    if not drop_hash:
        man["files"] = {n: pc.sha256_bytes(b) for n, b in files.items() if n != "manifest.json"}
    files["manifest.json"] = json.dumps(man).encode()
    with zipfile.ZipFile(dst, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    return dst


def main():
    cfg, ms = make_fixture()
    path, man = P.export_pack(cfg, "fx", TMP / "out", author="kris")
    print(f"fixture: champion {ms} ms, pack {path.name}")

    # --- packcheck -----------------------------------------------------------------------------------------
    m2, files = pc.check_zip(path, pc.sha256_file(path), path.stat().st_size)
    check("good pack passes the safety checks", m2["pack_id"] == man["pack_id"])
    expect_error("tampered hash is rejected", lambda: pc.check_zip(path, "0" * 64), "sha256 mismatch")
    bad = TMP / "trav.zip"
    rezip(path, bad, add={"../evil.txt": b"hello"})
    expect_error("path traversal is rejected", lambda: pc.check_zip(bad), "unsafe file name")
    rezip(path, bad, add={"maps/../../x.json": b"{}"})
    expect_error("nested path traversal is rejected", lambda: pc.check_zip(bad), "unsafe file name")
    b = io.BytesIO()
    np.savez(b, w=np.array([object(), 1], dtype=object))
    rezip(path, bad, change={"model.npz": b.getvalue()})
    expect_error("pickled object array in model.npz is rejected", lambda: pc.check_zip(bad), "not a plain numpy array")
    import pickle
    rezip(path, bad, add={"notes.txt": pickle.dumps({"x": 1})})
    expect_error("pickle disguised as .txt is rejected", lambda: pc.check_zip(bad), "pickle")
    rezip(path, bad, add={"model.pkl": b"x"})
    expect_error(".pkl file is rejected", lambda: pc.check_zip(bad), "file type not allowed")
    rezip(path, bad, add={"run.exe.txt": b"MZ\x90\x00"})
    expect_error("executable disguised as .txt is rejected", lambda: pc.check_zip(bad), "executable")
    rezip(path, bad, change={"model.json": b'{"obs_dim": 1}'}, drop_hash=True)
    expect_error("file changed after export (manifest hash) is rejected", lambda: pc.check_zip(bad), "sha256")
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", "{}")
        z.writestr("big.txt", "0" * (60 * 1024 * 1024))
    expect_error("zip bomb (compression ratio) is rejected", lambda: pc.check_zip(bad), "ratio")
    check("hub copy of packcheck.py is identical", (REPO / "packcheck.py").read_bytes() ==
          (REPO / "hub_template" / "tools" / "packcheck.py").read_bytes())

    # --- the pull-request validator -------------------------------------------------------------------------
    _, entry, _ = hub.cmd_publish(dict(cfg, github_user="kris"), "fx", open_browser=False)
    ppath = TMP / "runs" / "_hub" / "out" / f"{entry['pack_id']}{P.PACK_EXT}"
    base = TMP / "hub_base"
    shutil.copytree(TMP / "hub_template", base)

    def pr(files, author="kris", base_dir=base):
        head = TMP / "hub_head"
        shutil.rmtree(head, ignore_errors=True)
        shutil.copytree(base_dir, head)
        for rel, content in files.items():
            p = head / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content if isinstance(content, str) else json.dumps(content, indent=1))
        r = subprocess.run([sys.executable, str(base_dir / "tools" / "validate_pr.py"), "--base", str(base_dir),
                            "--head", str(head), "--author", author, "--asset-dir", str(ppath.parent)],
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    rel = f"packs/{entry['pack_id']}.json"
    code, out = pr({rel: entry})
    check("validator: good submission passes", code == 0, out)
    code, out = pr({rel: dict(entry, sha256="1" * 64)})
    check("validator: tampered sha256 is rejected", code == 1 and "sha256 mismatch" in out, out)
    code, out = pr({rel: dict(entry, size=entry["size"] + 1)})
    check("validator: wrong size is rejected", code == 1 and "size mismatch" in out, out)
    code, out = pr({rel: dict(entry, status="verified")})
    check("validator: author setting 'verified' is rejected", code == 1 and "status" in out, out)
    code, out = pr({rel: entry, "packs/other-12345678.json": entry})
    check("validator: two files in one PR are rejected", code == 1 and "exactly one file" in out, out)
    code, out = pr({rel: entry}, author="mallory")
    check("validator: someone else submitting your entry is rejected", code == 1 and "author_github" in out, out)
    code, out = pr({rel: dict(entry, input_fingerprint="0" * 16)})
    check("validator: incompatible input layout is rejected", code == 1, out)
    code, out = pr({"tools/validate_pr.py": "print('owned')"})
    check("validator: changing hub tools is rejected", code == 1 and "may only add" in out, out)
    trav = rezip(ppath, ppath.parent / "trav.tmbrain.zip", add={"../x.txt": b"x"})
    code, out = pr({rel: dict(entry, sha256=pc.sha256_file(trav), size=trav.stat().st_size,
                              asset_url=entry["asset_url"].rsplit("/", 1)[0] + "/trav.tmbrain.zip")})
    check("validator: unsafe zip behind a matching hash is rejected", code == 1 and "unsafe file name" in out, out)

    # merge the pack, then a verification
    (base / "packs" / f"{entry['pack_id']}.json").write_text(json.dumps(entry))
    ver = hub.cmd_verify(dict(cfg, github_user="kris"), str(ppath))
    check("verify: claimed runs replay to the exact time", ver["verdict"] == "verified" and all(
        r["match"] for m in ver["maps"] for r in m["runs"].values()), json.dumps(ver["maps"])[:300])
    check("verify: the pack's driver finishes (best of 3)", ver["maps"][0]["drive_ms"] == ms, str(ver["maps"][0].get("drive_ms")))
    vrel = f"verifications/{entry['pack_id']}/kris.json"
    code, out = pr({vrel: ver})
    check("validator: verification by a non-verifier is rejected", code == 1 and "not in verifiers.txt" in out, out)
    (base / "verifiers.txt").write_text("Kristiano55\nkris\n")
    code, out = pr({vrel: ver})
    check("validator: verification by a listed verifier passes", code == 0, out)
    code, out = pr({vrel: dict(ver, verdict="failed")})
    check("validator: verification edited after signing is rejected", code == 1 and "digest" in out, out)

    # a faked claim: the champion time in the manifest is 1.75 s faster than the run in the pack
    zin = zipfile.ZipFile(ppath)
    mf = json.loads(zin.read("manifest.json"))
    mf["maps"][0]["champion_ms"] = ms - 1750
    mf["maps"][0]["runs"]["champion"]["claimed_ms"] = ms - 1750
    fake = rezip(ppath, TMP / "out" / "faked.tmbrain.zip", change={"manifest.json": json.dumps(mf).encode()})
    vf = hub.cmd_verify(dict(cfg, github_user="kris"), str(fake))
    r = vf["maps"][0]["runs"]["champion"]
    check("verify: a faked time is rejected", vf["verdict"] == "failed" and not r["match"] and r["replay_ms"] == ms,
          json.dumps(r))

    # --- index builder: verifiers only, quarantine on reports ------------------------------------------------------
    sys.path.insert(0, str(base / "tools"))
    import build_index as B
    vdir = base / "verifications" / entry["pack_id"]
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "mallory.json").write_text(json.dumps(dict(ver, verifier="mallory")))
    idx = B.build(base, [])
    check("index: a non-verifier's file does not verify", idx["packs"][0]["status"] == "unverified")
    (vdir / "kris.json").write_text(json.dumps(ver))
    idx = B.build(base, [])
    p0 = idx["packs"][0]
    check("index: a verifier's file verifies", p0["status"] == "verified" and p0["maps"][0]["verified"]["drive_ms"] == ms)
    rep = [{"title": f"Report: {entry['pack_id']}", "labels": [{"name": "report"}], "created_at": "2099-01-01T00:00:00Z"}]
    check("index: a report quarantines the pack", B.build(base, rep)["packs"][0]["status"] == "quarantined")
    old = [{"title": f"Report: {entry['pack_id']}", "labels": [{"name": "report"}], "created_at": "2001-01-01T00:00:00Z"}]
    check("index: a re-verification after the report lifts the quarantine", B.build(base, old)["packs"][0]["status"] == "verified")
    B.main(["--repo", str(base), "--out", str(TMP / "site")])
    check("leaderboard page is built", "tm_ai brains" in (TMP / "site" / "index.html").read_text())

    # --- hub list / import through a file:// index ----------------------------------------------------------
    local = json.loads((base / "index.json").read_text())
    for p in local["packs"]:
        p["asset_url"] = ppath.as_uri()
    (TMP / "index_local.json").write_text(json.dumps(local))
    hcfg = dict(cfg, hub_index_url=(TMP / "index_local.json").as_uri())
    rows = hub.cmd_list(hcfg, refresh=True)
    check("hub list reads the index", len(rows) == 1 and rows[0]["pack_id"] == entry["pack_id"])
    check("hub search finds by map name", len(hub.cmd_list(hcfg, "asphalt")) == 1 and not hub.cmd_list(hcfg, "nothing"))
    shutil.rmtree(TMP / "runs" / "_hub" / "packs", ignore_errors=True)
    hub.cmd_import(hcfg, entry["pack_id"], "imported")
    d = TMP / "runs" / "imported"
    check("hub import: downloaded, checked and imported", (d / "latest.zip").exists() and (d / "pack.json").exists()
          and json.loads((d / "pack.json").read_text())["sha256"] == entry["sha256"])
    local["packs"][0]["sha256"] = "2" * 64
    (TMP / "index_local.json").write_text(json.dumps(local))
    hub.index(hcfg, refresh=True)
    shutil.rmtree(TMP / "runs" / "_hub" / "packs", ignore_errors=True)
    expect_error("hub import refuses a file whose sha256 differs from the index",
                 lambda: hub.cmd_import(hcfg, entry["pack_id"], "imported2"), "sha256")
    local["packs"][0].update(sha256=entry["sha256"], status="quarantined")
    (TMP / "index_local.json").write_text(json.dumps(local))
    hub.index(hcfg, refresh=True)
    expect_error("hub import refuses a quarantined pack", lambda: hub.cmd_import(hcfg, entry["pack_id"], "imported3"),
                 "quarantined")

    # --- benchmark: the pack drives the benchmark maps (one on this PC, one missing) -------------------------------
    (TMP / "benchmark.json").write_text(json.dumps({"version": 1, "maps": [
        {"uid": man["maps"][0]["uid"], "name": "fake:1000:asphalt", "par_ms": 13000},
        {"uid": "NOTHERE1234", "name": "missing.Map.Gbx"}]}))
    bres = hub.cmd_benchmark(hcfg, str(ppath))
    want = round((100.0 * 13000 / ms + 0.0) / 2, 1)
    check("benchmark: score = average over maps, missing map = 0", bres["score"] == want and not bres["complete"],
          f"{bres['score']} vs {want}")
    v2 = hub.cmd_verify(dict(hcfg, github_user="kris"), str(ppath), bench=True)
    check("verify --benchmark adds the score inside the signed digest", v2["benchmark"]["score"] == want and
          pc.digest({k: x for k, x in v2.items() if k != "signed_off"}) == v2["digest"])

    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}   (temp folder {TMP})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
