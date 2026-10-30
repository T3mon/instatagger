"""Tag every photo on an Instagram profile and build a page to search them by tag.

    python instatagger.py <handle or profile link> [N]   newest N posts (default all): list, fetch, tag, build the page
    python instatagger.py <handle> --retag               re-score after editing tags.txt (no Instagram calls)

Needs session.txt next to this file, holding your Instagram `sessionid` cookie value.
Full-size photos only pass through memory. out/<handle>/ keeps small previews,
the booru-tag scores (what makes --retag instant) and index.html.
"""
import csv
import ctypes
import json
import os
import random
import re
import secrets
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path

import instaloader
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from timm.data import create_transform, resolve_data_config

HERE = Path(__file__).parent
MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"  # booru tagger, trained on labels like underboob, pool, no_humans
THRESHOLD = 0.35  # default cutoff, the tagger's usual one; override per tag with "| 0.5" in tags.txt
RATINGS = {"general": "safe", "sensitive": "suggestive", "questionable": "questionable", "explicit": "explicit"}  # tagger's own levels
THUMB = 360
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def handle_from(text):
    handle = text.split("instagram.com/")[-1].split("/")[0].split("?")[0].lstrip("@")
    if not re.fullmatch(r"\w(?:[\w.]{0,28}\w)?", handle):  # also rejects "..": the handle becomes a folder name
        sys.exit(f"Not an Instagram handle: {text}")
    return handle


def parse_tags(text):
    """'[Group]' starts a group; 'tag: booru_tag ...' or 'tag: booru_tag ... | cutoff' adds a tag."""
    groups = []
    for line in map(str.strip, text.splitlines()):
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            groups.append((line.strip("[]"), []))
            continue
        if not groups:
            groups.append(("Tags", []))
        tag, _, rest = line.partition(":")
        booru, _, cutoff = rest.partition("|")
        groups[-1][1].append((tag.strip(), booru.split(), float(cutoff) if cutoff.strip() else THRESHOLD))
    return groups


class Polite(instaloader.RateController):
    def wait_before_query(self, query_type):
        super().wait_before_query(query_type)
        time.sleep(random.uniform(6, 12))  # one page of 12 posts per 6-12 s, so Instagram doesn't flag the account


def list_photos(handle, limit=None):
    """(url, item) for every photo slide of the newest `limit` posts (None = all); videos are skipped."""
    session_file = HERE / "session.txt"
    session = session_file.read_text(encoding="utf-8").strip() if session_file.exists() else ""
    if not session:
        sys.exit(f"Paste your Instagram sessionid cookie into {session_file}")
    L = instaloader.Instaloader(quiet=True, rate_controller=Polite)
    L.load_session("me", {"sessionid": session, "csrftoken": secrets.token_hex(16), "ds_user_id": session.split("%3A")[0]})
    photos = []
    try:
        profile = instaloader.Profile.from_username(L.context, handle)
        for i, post in enumerate(islice(profile.get_posts(), limit), 1):
            if i % 120 == 0:
                print(f"  {i}/{limit or profile.mediacount} posts listed", flush=True)
            nodes = (post.get_sidecar_nodes() if post.typename == "GraphSidecar"
                     else [instaloader.PostSidecarNode(post.is_video, post.url, None)])
            for num, node in enumerate(nodes, 1):
                if not node.is_video:
                    photos.append((node.display_url, {"c": post.shortcode, "n": num, "l": post.likes or 0,
                                                      "d": post.date_utc.date().isoformat(), "f": f"{post.shortcode}_{num}.jpg"}))
    except instaloader.exceptions.InstaloaderException as e:  # rate limit or expired login halfway: keep what we got
        print("Instagram stopped the listing:", e)
    if not photos:
        sys.exit("No photos found (expired session.txt? private profile? wrong handle?)")
    return photos


def prepare(job):
    """Fetch one photo into memory and save its preview; False if it couldn't be fetched."""
    url, thumb = job
    try:
        with Image.open(thumb) as im:  # fetched on an earlier run: reuse the preview, skip the download
            im.load()
        return True
    except OSError:  # not there yet, or half-written by a stopped run
        pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            img = Image.open(BytesIO(r.read())).convert("RGB")
    except Exception as e:  # one dead link shouldn't sink a 2,000-photo run
        print(f"skipped {thumb.name}: {e}")
        return False
    img.thumbnail((THUMB, THUMB))
    img.save(thumb, quality=80)
    return True


def booru_scores(thumbs):
    """Probability of every booru tag for each preview; the tagger wants white-padded squares in BGR."""
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = timm.create_model("hf-hub:" + MODEL, pretrained=True).eval().to(DEV, dtype)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    out = []
    for i in range(0, len(thumbs), 16):
        x = torch.stack([transform(ImageOps.pad(Image.open(f).convert("RGB"), (THUMB, THUMB), color="white"))
                         for f in thumbs[i:i + 16]])[:, [2, 1, 0]]
        with torch.no_grad():
            out.append(torch.sigmoid(model(x.to(DEV, dtype)).float()).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def booru_columns():
    with open(hf_hub_download(MODEL, "selected_tags.csv"), encoding="utf-8") as f:
        return {row["name"]: j for j, row in enumerate(csv.DictReader(f))}


def tag(wd, groups):
    """A photo gets a tag when any of the tag's booru tags scores >= its cutoff."""
    col = booru_columns()
    names, hits = [], []
    for _, tags in groups:
        for name, booru, cutoff in tags:
            missing = [b for b in booru if b not in col]
            if missing or not booru:
                sys.exit(f"{name}: the tagger has no tag {missing or '(none given)'}")
            hits.append(wd[:, [col[b] for b in booru]].max(axis=1) >= cutoff)
            names.append(name)
    return [[names[j] for j in np.flatnonzero(row)] for row in np.stack(hits, axis=1)]


def disk_size(files):
    """Bytes the files really take on disk: each one fills whole clusters (128 KB apiece on an exFAT drive)."""
    if hasattr(os.stat_result, "st_blocks"):
        return sum(f.stat().st_blocks * 512 for f in files)
    per_cluster, sector = ctypes.c_ulong(), ctypes.c_ulong()
    ctypes.windll.kernel32.GetDiskFreeSpaceW(HERE.resolve().anchor, ctypes.byref(per_cluster), ctypes.byref(sector), None, None)
    cluster = per_cluster.value * sector.value or 4096
    return sum(-(-f.stat().st_size // cluster) * cluster for f in files)


def stats_row(model, items, skipped, folder):
    """One line of the page's Stats panel."""
    dates, likes = sorted(it["d"] for it in items) or ["-"], sorted(it["l"] for it in items) or [0]
    top = Counter(t for it in items for t in it["t"] if not t.startswith("@")).most_common(3)
    files = [f for f in folder.rglob("*") if f.is_file()]
    thumbs = [f for f in files if f.parent.name == "thumbs"]
    return {"model": model, "posts": len({(it.get("h"), it["c"]) for it in items}), "photos": len(items), "skipped": skipped,
            "from": dates[0], "to": dates[-1], "likes": f"{likes[len(likes) // 2]:,}", "top": ", ".join(f"{t} {n}" for t, n in top),
            "thumbs": f"{disk_size(thumbs) / 2**20:,.0f} MB", "total": f"{disk_size(files) / 2**20:,.0f} MB"}


def add_rating(data, folder, cols):
    """Tag each shown photo with the tagger's most likely explicitness level, plus a 0-3 score for sorting."""
    if not (folder / "wd.npy").exists():
        return
    wd = np.load(folder / "wd.npy").astype(np.float32)[:, cols]
    row = {it["f"]: k for k, it in enumerate(json.loads((folder / "items.json").read_text(encoding="utf-8")))}
    for it in data["items"]:
        p = wd[row[it["f"]]]
        it["t"] = it["t"] + [list(RATINGS.values())[p.argmax()]]
        it["x"] = round(float(p @ np.arange(len(p)) / p.sum()), 3)
    data["groups"] = [["Rating", list(RATINGS.values())]] + data["groups"]


def build_pages():
    """Write every out/<handle>/index.html and the all-models out/index.html from the saved data.json files (no re-tagging)."""
    template = (HERE / "viewer.html").read_text(encoding="utf-8")

    def write(path, data):
        path.write_text(template.replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")

    rows, items, groups = [], [], {"Model": []}
    col = booru_columns()
    for f in sorted((HERE / "out").glob("*/data.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        add_rating(d, f.parent, [col[r] for r in RATINGS])
        rows.append(stats_row("@" + d["handle"], d["items"], d["skipped"], f.parent))
        write(f.parent / "index.html", {**d, "stats": rows[-1:]})
        groups["Model"].append("@" + d["handle"])
        items += [{**it, "h": d["handle"], "t": it["t"] + ["@" + d["handle"]]} for it in d["items"]]
        for g, tags in d["groups"]:  # profiles tagged with an older tags.txt keep their old tag names
            known = groups.setdefault(g, [])
            known += [t for t in tags if t not in known]
    rows.append(stats_row("All", items, sum(r["skipped"] for r in rows), HERE / "out"))
    write(HERE / "out" / "index.html", {"title": "All models", "all": True, "items": items, "groups": list(groups.items()), "stats": rows})


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    handle = handle_from(sys.argv[1])
    limit = next((int(a) for a in sys.argv[2:] if a.isdigit()), None)
    out = HERE / "out" / handle
    if "--retag" in sys.argv:
        items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    else:
        print("Listing posts, one page of 12 every 6-12 s...", flush=True)
        photos = {it["f"]: (url, it) for url, it in list_photos(handle, limit)}  # dedupes repeats
        (out / "thumbs").mkdir(parents=True, exist_ok=True)
        print(f"Fetching and tagging {len(photos)} photos...", flush=True)
        jobs = [(url, out / "thumbs" / f) for f, (url, _) in photos.items()]
        with ThreadPoolExecutor(8) as pool:
            ok = list(pool.map(prepare, jobs))
            retry = [k for k, good in enumerate(ok) if not good]  # timeouts come in bursts when the line is busy
            for k, good in zip(retry, pool.map(prepare, [jobs[k] for k in retry])):
                ok[k] = good
        items = [it for (_, it), good in zip(photos.values(), ok) if good]
        if len(items) < len(photos) / 2:  # e.g. Wi-Fi dropped mid-run: don't replace a good earlier run with a handful of photos
            sys.exit(f"Only {len(items)} of {len(photos)} photos could be downloaded (network down?). Nothing saved; run it again.")
        (out / "items.json").write_text(json.dumps(items), encoding="utf-8")
        (out / "wd.npy").unlink(missing_ok=True)  # new set of photos: scores are recomputed below
    if not (out / "wd.npy").exists():
        np.save(out / "wd.npy", booru_scores([out / "thumbs" / it["f"] for it in items]))
    wd = np.load(out / "wd.npy")
    groups = parse_tags((HERE / "tags.txt").read_text(encoding="utf-8"))
    for it, tags in zip(items, tag(wd, groups)):
        it["t"] = tags
    skip = {name for g, tags in groups if g.lower() == "skip" for name, _, _ in tags}
    shown = [it for it in items if not skip.intersection(it["t"])]  # posts with no person: scenery, text slides...
    data = {"handle": handle, "items": shown, "skipped": len(items) - len(shown),
            "groups": [[g, [name for name, _, _ in tags]] for g, tags in groups if g.lower() != "skip"]}
    (out / "data.json").write_text(json.dumps(data), encoding="utf-8")
    build_pages()
    for g, tags in groups:
        counted = items if g.lower() == "skip" else shown
        print(f"{g}: " + ", ".join(f"{name} {sum(name in it['t'] for it in counted)}" for name, _, _ in tags))
    print(f"{len(shown)} photos shown, {len(items) - len(shown)} skipped -> {out / 'index.html'}")
    print(f"All models: {HERE / 'out' / 'index.html'}")


if __name__ == "__main__":
    main()
