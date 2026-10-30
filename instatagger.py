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
GIRLS = ["1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls"]
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


def login(rate_controller=Polite):
    """Instaloader logged in as you, from the sessionid cookie in session.txt."""
    session_file = HERE / "session.txt"
    session = session_file.read_text(encoding="utf-8").strip() if session_file.exists() else ""
    if not session:
        sys.exit(f"Paste your Instagram sessionid cookie into {session_file}")
    L = instaloader.Instaloader(quiet=True, rate_controller=rate_controller)
    L.load_session("me", {"sessionid": session, "csrftoken": secrets.token_hex(16), "ds_user_id": session.split("%3A")[0]})
    return L


def list_photos(handle, limit=None):
    """(url, item) for every photo slide of the newest `limit` posts (None = all); videos are skipped."""
    L = login()
    photos = []
    try:
        profile = instaloader.Profile.from_username(L.context, handle)
        total = min(limit or profile.mediacount, profile.mediacount)
        for i, post in enumerate(islice(profile.get_posts(), limit), 1):
            nodes = (post.get_sidecar_nodes() if post.typename == "GraphSidecar"
                     else [instaloader.PostSidecarNode(post.is_video, post.url, None)])
            for num, node in enumerate(nodes, 1):
                if not node.is_video:
                    photos.append((node.display_url, {"c": post.shortcode, "n": num, "l": post.likes or 0,
                                                      "d": post.date_utc.date().isoformat(), "f": f"{post.shortcode}_{num}.jpg"}))
            if i % 12 == 0 or i == total:
                print(f"  {i}/{total} posts listed, {len(photos)} photos found", flush=True)
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
    print("  loading the tagger", flush=True)
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = timm.create_model("hf-hub:" + MODEL, pretrained=True).eval().to(DEV, dtype)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    out = []
    for i in range(0, len(thumbs), 16):
        x = torch.stack([transform(ImageOps.pad(Image.open(f).convert("RGB"), (THUMB, THUMB), color="white"))
                         for f in thumbs[i:i + 16]])[:, [2, 1, 0]]
        with torch.no_grad():
            out.append(torch.sigmoid(model(x.to(DEV, dtype)).float()).cpu())
        if (i // 16) % 10 == 9 or i + 16 >= len(thumbs):
            print(f"  {min(i + 16, len(thumbs))}/{len(thumbs)} photos scored", flush=True)
    return torch.cat(out).numpy().astype(np.float16)


def booru_columns():
    with open(hf_hub_download(MODEL, "selected_tags.csv"), encoding="utf-8") as f:
        return {row["name"]: j for j, row in enumerate(csv.DictReader(f))}


def no_girl(wd):
    """Pre-screen: nobody in the photo, or people but no girl and no single subject (men, crowds, stages).
    A lone figure the tagger calls male is kept: it takes women in suits or with slicked-back hair for men."""
    col = booru_columns()
    girl = wd[:, [col[t] for t in GIRLS]].max(axis=1)
    return (wd[:, col["no_humans"]] >= 0.35) | ((girl < 0.45) & (wd[:, col["solo"]] < 0.5))


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
    skipped_file = out / "skipped.json"  # photos the pre-screen threw out: never downloaded again
    skipped = set(json.loads(skipped_file.read_text(encoding="utf-8"))) if skipped_file.exists() else set()
    if "--retag" in sys.argv:
        items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    else:
        print("Step 1/3: listing posts, one page of 12 every 6-12 s", flush=True)
        photos = {it["f"]: (url, it) for url, it in list_photos(handle, limit) if it["f"] not in skipped}  # dedupes repeats
        (out / "thumbs").mkdir(parents=True, exist_ok=True)
        print(f"Step 2/3: downloading {len(photos)} photos", flush=True)
        jobs = [(url, out / "thumbs" / f) for f, (url, _) in photos.items()]
        with ThreadPoolExecutor(8) as pool:
            ok = []
            for k, good in enumerate(pool.map(prepare, jobs), 1):
                ok.append(good)
                if k % 25 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} photos downloaded", flush=True)
            retry = [k for k, good in enumerate(ok) if not good]  # timeouts come in bursts when the line is busy
            if retry:
                print(f"  retrying {len(retry)} failed downloads", flush=True)
            for k, good in zip(retry, pool.map(prepare, [jobs[k] for k in retry])):
                ok[k] = good
        print(f"  {sum(ok)}/{len(jobs)} photos downloaded, {len(jobs) - sum(ok)} failed", flush=True)
        items = [it for (_, it), good in zip(photos.values(), ok) if good]
        if len(items) < len(photos) / 2:  # e.g. Wi-Fi dropped mid-run: don't replace a good earlier run with a handful of photos
            sys.exit(f"Only {len(items)} of {len(photos)} photos could be downloaded (network down?). Nothing saved; run it again.")
        if (out / "items.json").exists():  # keep earlier photos this run didn't reach, e.g. a re-run limited to the newest posts
            new = {it["f"] for it in items}
            items += [it for it in json.loads((out / "items.json").read_text(encoding="utf-8"))
                      if it["f"] not in new and (out / "thumbs" / it["f"]).exists()]
        (out / "items.json").write_text(json.dumps(items), encoding="utf-8")
        (out / "wd.npy").unlink(missing_ok=True)  # new set of photos: scores are recomputed below
    print(f"Step 3/3: scoring and tagging {len(items)} photos", flush=True)
    if not (out / "wd.npy").exists():
        np.save(out / "wd.npy", booru_scores([out / "thumbs" / it["f"] for it in items]))
    wd = np.load(out / "wd.npy")
    groups = parse_tags((HERE / "tags.txt").read_text(encoding="utf-8"))
    print("  applying your tags", flush=True)
    tags = tag(wd, groups)
    extra = {name for g, group in groups if g.lower() == "skip" for name, _, _ in group}
    drop = no_girl(wd) | np.array([bool(extra.intersection(t)) for t in tags])
    print(f"  skipping photos without a girl: {int(drop.sum())}", flush=True)
    if drop.any():  # skipped photos lose their preview and scores, and are remembered so they're never fetched again
        for it in [it for it, d in zip(items, drop) if d]:
            (out / "thumbs" / it["f"]).unlink(missing_ok=True)
            skipped.add(it["f"])
        skipped_file.write_text(json.dumps(sorted(skipped)), encoding="utf-8")
        items, wd, tags = [it for it, d in zip(items, drop) if not d], wd[~drop], [t for t, d in zip(tags, drop) if not d]
        (out / "items.json").write_text(json.dumps(items), encoding="utf-8")
        np.save(out / "wd.npy", wd)
    for it, t in zip(items, tags):
        it["t"] = t
    groups = [(g, group) for g, group in groups if g.lower() != "skip"]
    data = {"handle": handle, "items": items, "skipped": len(skipped), "groups": [[g, [name for name, _, _ in group]] for g, group in groups]}
    (out / "data.json").write_text(json.dumps(data), encoding="utf-8")
    print("  building pages", flush=True)
    build_pages()
    for g, group in groups:
        print(f"{g}: " + ", ".join(f"{name} {sum(name in it['t'] for it in items)}" for name, _, _ in group))
    print(f"{len(items)} photos -> {out / 'index.html'} | {int(drop.sum())} without a girl skipped and deleted now, {len(skipped)} in total")
    print(f"All models: {HERE / 'out' / 'index.html'}")


if __name__ == "__main__":
    main()
