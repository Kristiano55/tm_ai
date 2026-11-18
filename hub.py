"""
Brain hub client: the public pack index on GitHub (repo Kristiano55/tm_ai-brains, see hub_template/).

    python tm_ai.py hub list [--map TEXT] [--verified]      packs in the index (cached in runs/_hub/index.json)
    python tm_ai.py hub search TEXT
    python tm_ai.py hub import PACK_ID [--as NAME] [--base]  download, check size + sha256 + safety, safe import
    python tm_ai.py hub refresh
    python tm_ai.py publish RUN|base:NAME [--wr SECONDS]     export + sha256 + open the pre-filled GitHub pages
    python tm_ai.py verify PACK(.zip or id) [--map UID=PATH]  replay claimed runs + drive best-of-3 -> verification
    python tm_ai.py benchmark PACK|base:NAME                  benchmark maps, no learning -> score
    python tm_ai.py hub verify-pending                        verifiers: verify every unverified pack you have maps for

No token is needed or stored by default: publishing opens the GitHub pages with everything filled in. With the
optional github_token setting, the release asset is uploaded directly. Nothing from a pack is ever executed.
"""
import json
import os
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

import numpy as np

import config as cfgmod
import packcheck as pc
import packs as P
import tm_ai as T

HUB = cfgmod.ROOT / "runs" / "_hub"
VERIFY_SEEDS = (None, 1000, 1001)               # best-of-3, fixed: anyone gets the same laps


# --- index ---------------------------------------------------------------------------------------------------
def _fetch(url, dest=None, limit=pc.MAX_ZIP_BYTES):
    """http(s):// or file:// download (size-limited). Returns bytes, or writes to dest."""
    req = urllib.request.Request(url, headers={"User-Agent": "tm_ai-hub-client"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out, n = (open(dest, "wb") if dest else None), 0
        chunks = []
        try:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                n += len(b)
                if n > limit:
                    raise pc.PackError(f"download larger than {limit} bytes")
                (out.write(b) if out else chunks.append(b))
        finally:
            if out:
                out.close()
    return None if dest else b"".join(chunks)


def index(cfg, refresh=False):
    """The hub index (cached; refreshed when asked or older than an hour)."""
    HUB.mkdir(parents=True, exist_ok=True)
    f = HUB / "index.json"
    if refresh or not f.exists() or time.time() - f.stat().st_mtime > 3600:
        try:
            data = json.loads(_fetch(cfg["hub_index_url"], limit=50 * 1024 * 1024).decode("utf-8"))
            T.write_json(f, data)
        except Exception as e:
            if not f.exists():
                raise SystemExit(f"Could not load the hub index from {cfg['hub_index_url']}: {e}")
            T.say(f"Hub index refresh failed ({str(e)[:80]}); using the cached copy")
    return T.read_json(f)


def my_lineage():
    """base ids of this PC's bases and runs started from a base / pack (for "same lineage" filters)."""
    ids = set()
    for f in list((cfgmod.ROOT / "runs").glob("*/base.json")) + list((cfgmod.ROOT / "runs" / "_base").glob("*/base.json")):
        b = T.read_json(f).get("base_id")
        if b:
            ids.add(b)
    return sorted(ids)


def find(cfg, pack_id, refresh=False):
    for p in index(cfg, refresh).get("packs", []):
        if p.get("pack_id") == pack_id:
            return p
    return None


def _fmt_ms(ms):
    return f"{ms / 1000:.3f}" if ms else "-"


def cmd_list(cfg, text=None, verified=False, refresh=False):
    packs = index(cfg, refresh).get("packs", [])
    rows = []
    for p in packs:
        hay = " ".join([p.get("pack_id", ""), p.get("author", ""), p.get("note", "")] +
                       [f"{m.get('uid', '')} {m.get('name', '')}" for m in p.get("maps", [])]).lower()
        if text and text.lower() not in hay:
            continue
        if verified and p.get("status") != "verified":
            continue
        rows.append(p)
    if not rows:
        print("no packs match")
    for p in rows:
        maps = ", ".join(f"{m.get('name') or m.get('uid')} {_fmt_ms((m.get('verified') or {}).get('drive_ms') or m.get('champion_ms'))} s"
                         for m in p.get("maps", []))
        print(f"{p['pack_id']:30s} {p.get('status', '?'):11s} {p.get('kind', ''):4s} by {p.get('author', '?'):15s} "
              f"bench {p.get('benchmark_score') or '-':>5}  votes {p.get('votes', 0):3d}  {maps}")
    return rows


def download(cfg, entry):
    """The entry's release asset, checked against the index's size + sha256 and the safety rules."""
    d = HUB / "packs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{entry['pack_id']}{P.PACK_EXT}"
    if not (path.exists() and pc.sha256_file(path) == entry["sha256"]):
        tmp = path.with_suffix(".part")
        _fetch(entry["asset_url"], tmp)
        os.replace(tmp, path)
    pc.check_zip(path, entry["sha256"], entry["size"])
    return path


def cmd_import(cfg, pack_id, name=None, as_base=False):
    e = find(cfg, pack_id) or find(cfg, pack_id, refresh=True)
    if not e:
        raise SystemExit(f"Pack {pack_id} is not in the hub index.")
    if e.get("status") == "quarantined":
        raise SystemExit(f"Pack {pack_id} is quarantined (reported, waiting for re-verification); not importing.")
    path = download(cfg, e)
    name = name or pack_id
    dest = cfgmod.base_dir(name) if as_base else cfgmod.ROOT / "runs" / name
    man = P.import_pack(path, dest, e["sha256"], e["size"], as_base=as_base)
    T.say(f"Imported pack {pack_id} ({e.get('status')}) into {dest}. "
          + ("Use it as a base ('Start this map run from base' = " + name + ")" if as_base else
             f"Train it on its map with run name '{name}'"))
    return man


# --- publish ------------------------------------------------------------------------------------------------
def entry_for(man, path, user, repo_name):
    """The packs/<id>.json entry for the hub (status always "unverified": only verifiers change that)."""
    tag = f"pack-{man['pack_id']}"
    return {"pack_id": man["pack_id"], "kind": man["kind"], "author": man["author"], "author_github": user,
            "created": man["created"], "code_version": man["code_version"],
            "input_fingerprint": man["input_fingerprint"], "base_id": man.get("base_id"),
            "lineage": man.get("lineage") or [], "steps": man.get("steps"), "note": man.get("note", ""),
            "maps": [{"uid": m["uid"], "name": m.get("name"), "champion_ms": m.get("champion_ms"),
                      "best_ms": m.get("best_ms"), "pb_ms": m.get("pb_ms"), "wr_ms": m.get("wr_ms")}
                     for m in man["maps"]],
            "sha256": pc.sha256_file(path), "size": os.path.getsize(path),
            "asset_url": f"https://github.com/{user}/{repo_name}/releases/download/{tag}/{Path(path).name}",
            "status": "unverified"}


def _upload_release(cfg, user, repo_name, tag, path, body):
    """Optional (github_token set): create the release on your fork and upload the pack as its asset."""
    tok = cfg["github_token"]
    hdr = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "tm_ai-hub-client"}
    req = urllib.request.Request(f"https://api.github.com/repos/{user}/{repo_name}/releases", method="POST",
                                 data=json.dumps({"tag_name": tag, "name": tag, "body": body}).encode(), headers=hdr)
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.loads(r.read().decode())
    up = rel["upload_url"].split("{")[0] + "?name=" + urllib.parse.quote(Path(path).name)
    req = urllib.request.Request(up, method="POST", data=Path(path).read_bytes(),
                                 headers={**hdr, "Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())["browser_download_url"]


def new_file_url(repo, filename, content):
    """GitHub's "create new file" page, pre-filled (GitHub offers to fork + open a pull request)."""
    return (f"https://github.com/{repo}/new/main?filename={urllib.parse.quote(filename)}"
            f"&value={urllib.parse.quote(content)}")


def cmd_publish(cfg, source, wr=None, note="", open_browser=True):
    user = cfg["github_user"] or "your-github-name"
    path, man = P.export_pack(cfg, source, HUB / "out", author=cfg["github_user"] or "anonymous", wr=wr, note=note)
    repo_name = cfg["hub_repo"].split("/")[-1]
    entry = entry_for(man, path, user, repo_name)
    tag = f"pack-{man['pack_id']}"
    body = (f"tm_ai brain pack {man['pack_id']} ({man['kind']}), sha256 {entry['sha256']}, "
            f"{entry['size']} bytes. Maps: " + ", ".join(f"{m.get('name')} ({m['uid']})" for m in man["maps"]))
    if cfg["github_token"]:
        entry["asset_url"] = _upload_release(cfg, user, repo_name, tag, path, body)
        T.say(f"Uploaded {path.name} to the release {tag} on {user}/{repo_name}")
    entry_file = HUB / "out" / f"{man['pack_id']}.json"
    T.write_json(entry_file, entry)
    urls = []
    if not cfg["github_token"]:
        urls.append(f"https://github.com/{user}/{repo_name}/releases/new?" + urllib.parse.urlencode(
            {"tag": tag, "title": tag, "body": body + f"\n\nAttach this file: {path}"}))
    urls.append(new_file_url(cfg["hub_repo"], f"packs/{man['pack_id']}.json", json.dumps(entry, indent=2)))
    T.say(f"Pack {path} (sha256 {entry['sha256']}, {entry['size']} bytes).")
    T.say("Next: " + ("1) on the release page, attach the pack zip and publish; 2) " if not cfg["github_token"] else "")
          + "on the new-file page, propose the file (GitHub opens a pull request). The hub's check validates it.")
    for u in urls:
        T.say("  " + u[:160] + ("..." if len(u) > 160 else ""))
        if open_browser:
            webbrowser.open(u)
    return path, entry, urls


# --- verify -----------------------------------------------------------------------------------------------
def _local_map(cfg, uid, name, overrides):
    if uid in overrides:
        return overrides[uid]
    rec = T.read_json(T.MAPS_FILE).get(uid)
    if rec and (str(rec["path"]).startswith("fake:") or Path(rec["path"]).exists()):
        return rec["path"]
    folder = cfg.get("maps_folder") or ""
    if folder and name:                              # same file name in your maps folder, confirmed by UID below
        for p in Path(folder).rglob(name):
            return str(p)
    return None


def _open_pack(cfg, pack):
    """A local pack zip path, or a pack id from the hub (downloaded + checked)."""
    if Path(pack).exists():
        return Path(pack), None
    e = find(cfg, pack) or find(cfg, pack, refresh=True)
    if not e:
        raise SystemExit(f"{pack}: no such file and not in the hub index")
    return download(cfg, e), e


def drive_tries(cfg, model, man, files, uid, map_path, work):
    """The pack's driver on a local map, best-of-3 with fixed seeds, using the pack's line / ghost."""
    md = Path(work) / uid
    P.unpack_map(files, uid, md)
    c = cfgmod.map_cfg(cfg, {"map": map_path}, cfg["progress_norm_m"] if man.get("kind") == "base" or man.get("base_id") else 0)
    env = T.TMEnv(c, md, shorten=False, role="verify")
    if env.map_uid and env.map_uid != uid:
        env.close()
        raise pc.PackError(f"{map_path} has UID {env.map_uid}, not {uid}")
    tries = []
    for sd in VERIFY_SEEDS:
        inf = T.run_episode(env, model, sd)
        length = env.dense[2][-1] if env.dense is not None else env.lap_len
        tries.append({"seed": sd, "finished": bool(inf["finished"]),
                      "ms": int(inf["race_ms"]) if inf["finished"] else None, "end": inf.get("end", ""),
                      "cp": int(inf.get("cp", 0)),
                      "progress": round(min(float(getattr(env, "_s", 0.0)) / max(length, 1.0), 1.0), 3)})
    return env, tries


def replay(env, text_bytes, ticks):
    """Replay a TICK input file on the map: the exact finish time (deterministic physics), or None."""
    tmp = Path(env.run_dir) / "_replay.txt"
    tmp.write_bytes(text_bytes)
    acts = T.InputText.read(str(tmp))
    _, ms = T.trace_record(env.g, acts, ticks=ticks)
    return ms


def cmd_verify(cfg, pack, overrides=None, open_browser=False, bench=False):
    overrides = overrides or {}
    path, entry = _open_pack(cfg, pack)
    man, files = P.load_pack(path, entry and entry["sha256"], entry and entry["size"])
    model, _, _ = P.build_model(man, files)
    work = HUB / "verify" / man["pack_id"]
    results, ok = [], True
    for m in man["maps"]:
        uid = m["uid"]
        mp = _local_map(cfg, uid, m.get("name"), overrides)
        r = {"uid": uid, "name": m.get("name"), "runs": {}}
        if not mp:
            r["error"] = "map not found locally (train it once, or pass --map UID=PATH, or set maps_folder)"
            ok = False
            results.append(r)
            continue
        try:
            env, tries = drive_tries(cfg, model, man, files, uid, mp, work)
        except pc.PackError as e:
            r["error"] = str(e)
            ok = False
            results.append(r)
            continue
        for name, run in (m.get("runs") or {}).items():
            got = replay(env, files[run["file"]], cfg["ticks_per_step"])
            match = got is not None and run.get("claimed_ms") is not None and int(got) == int(run["claimed_ms"])
            r["runs"][name] = {"claimed_ms": run.get("claimed_ms"), "replay_ms": got, "match": match}
            ok = ok and match
            T.say(f"  {m.get('name')}: {name} run claimed {_fmt_ms(run.get('claimed_ms'))} s, replay "
                  f"{_fmt_ms(got)} s -> {'OK' if match else 'MISMATCH'}")
        env.close()
        fin = [t for t in tries if t["finished"]]
        r["drive"] = tries
        r["drive_ms"] = min(t["ms"] for t in fin) if fin else None
        if not fin:
            ok = False
        T.say(f"  {m.get('name')}: the pack's driver, best of 3: "
              + (f"{_fmt_ms(r['drive_ms'])} s" if fin else "no finish"))
        results.append(r)
    ver = {"pack_id": man["pack_id"], "sha256": pc.sha256_file(path), "verifier": cfg["github_user"] or "anonymous",
           "created": round(time.time()), "code_version": P.CODE_VERSION, "input_fingerprint": P.input_fingerprint(),
           "seeds": [s for s in VERIFY_SEEDS], "verdict": "verified" if ok else "failed", "maps": results}
    if bench:
        b = cmd_benchmark(cfg, str(path), overrides)
        ver["benchmark"] = {"score": b["score"], "complete": b["complete"], "maps": b["maps"]}
    ver["digest"] = pc.digest(ver)
    ver["signed_off"] = f"Signed-off-by: {ver['verifier']} (digest {ver['digest'][:16]})"
    out = HUB / "verifications" / f"{man['pack_id']}.json"
    T.write_json(out, ver)
    T.say(f"Verification {ver['verdict'].upper()} -> {out}")
    url = new_file_url(cfg["hub_repo"], f"verifications/{man['pack_id']}/{ver['verifier']}.json", json.dumps(ver, indent=2))
    T.say("Submit it (verifiers only): " + url[:140] + "...")
    if open_browser:
        webbrowser.open(url)
    return ver


def cmd_verify_pending(cfg):
    idx = index(cfg, refresh=True)
    user = cfg["github_user"]
    if user not in (idx.get("verifiers") or []):
        raise SystemExit(f"'{user or '(github_user not set)'}' is not in the hub's verifiers list.")
    done = []
    for e in idx.get("packs", []):
        if e.get("status") in ("unverified", "quarantined"):
            try:
                done.append(cmd_verify(cfg, e["pack_id"], open_browser=True, bench=True))
            except (SystemExit, pc.PackError) as ex:
                T.say(f"{e['pack_id']}: skipped ({ex})")
    return done


# --- benchmark ----------------------------------------------------------------------------------------------
def benchmark_maps(cfg):
    url = cfg.get("hub_benchmark_url") or cfg["hub_index_url"].rsplit("/", 1)[0] + "/benchmark.json"
    f = HUB / "benchmark.json"
    try:
        T.write_json(f, json.loads(_fetch(url, limit=5 * 1024 * 1024).decode("utf-8")))
    except Exception as e:
        if not f.exists():
            raise SystemExit(f"Could not load benchmark.json from {url}: {e}")
    return T.read_json(f).get("maps", [])


def map_score(par_ms, t):
    """100 = par time (e.g. the WR), more is faster (capped 150); no finish = up to 50 for the share driven."""
    if t["finished"] and t["ms"]:
        return min(150.0, 100.0 * float(par_ms) / t["ms"]) if par_ms else 100.0
    return 50.0 * float(t.get("progress") or 0.0)


def cmd_benchmark(cfg, pack, overrides=None):
    overrides = overrides or {}
    if pack.startswith("base:"):
        path, man = P.export_pack(cfg, pack, HUB / "bench_tmp", author="local")
    else:
        path, _ = _open_pack(cfg, pack)
        man = None
    man, files = P.load_pack(path)
    model, _, _ = P.build_model(man, files)
    rows, scores = [], []
    for bm in benchmark_maps(cfg):
        mp = _local_map(cfg, bm["uid"], bm.get("name"), overrides)
        if not mp:
            rows.append({"uid": bm["uid"], "name": bm.get("name"), "error": "not on this PC (see benchmark.json link)"})
            continue
        env, tries = drive_tries(cfg, model, man, files, bm["uid"], mp, HUB / "bench" / man["pack_id"])
        env.close()
        best = max(tries, key=lambda t: map_score(bm.get("par_ms"), t))
        s = map_score(bm.get("par_ms"), best)
        scores.append(s)
        rows.append({"uid": bm["uid"], "name": bm.get("name"), "best": best, "score": round(s, 1)})
        T.say(f"  {bm.get('name')}: " + (f"{_fmt_ms(best['ms'])} s" if best["finished"] else
              f"no finish ({best['progress']:.0%})") + f" -> {s:.1f}")
    n_all = len(rows)
    total = round(sum(scores) / n_all, 1) if n_all else None     # missing maps count as 0
    res = {"pack_id": man["pack_id"], "base_id": man.get("base_id"), "score": total, "maps": rows,
           "complete": len(scores) == n_all, "created": round(time.time()), "code_version": P.CODE_VERSION}
    T.write_json(HUB / "benchmarks" / f"{man['pack_id']}.json", res)
    T.say(f"Benchmark score {total} over {n_all} maps" + ("" if res["complete"] else " (some maps missing = 0)"))
    return res


def main(cfg, argv):
    import argparse
    ap = argparse.ArgumentParser(prog="tm_ai.py hub")
    ap.add_argument("action", choices=["list", "search", "import", "refresh", "verify-pending"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--map", default=None)
    ap.add_argument("--verified", action="store_true")
    ap.add_argument("--as", dest="name", default=None)
    ap.add_argument("--base", action="store_true", help="import as a base model (runs/_base/<name>)")
    a = ap.parse_args(argv)
    if a.action == "refresh":
        n = len(index(cfg, refresh=True).get("packs", []))
        T.say(f"Hub index refreshed: {n} packs")
    elif a.action in ("list", "search"):
        cmd_list(cfg, a.arg if a.action == "search" else a.map, a.verified)
    elif a.action == "import":
        if not a.arg:
            raise SystemExit("hub import PACK_ID")
        cmd_import(cfg, a.arg, a.name, a.base)
    elif a.action == "verify-pending":
        cmd_verify_pending(cfg)


def parse_overrides(items):
    out = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = v.strip()
    return out
