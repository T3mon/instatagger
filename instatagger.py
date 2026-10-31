"""Tag every photo on an Instagram profile and build a page to search them by tag.

    python instatagger.py <handle or profile link> [N]   newest N posts (default all): list, fetch, tag, build the page
    python instatagger.py <handle> --retag               re-score after editing tags.txt (no Instagram calls)

Needs session.txt next to this file, holding your Instagram `sessionid` cookie value.
Full-size photos only pass through memory. out/<handle>/ keeps small previews,
the booru-tag scores (what makes --retag instant), the image features and index.html.
"""
import base64
import csv
import ctypes
import json
import os
import random
import re
import secrets
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path
from queue import Queue

import instaloader
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

HERE = Path(__file__).parent
MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"  # booru tagger, trained on labels like underboob, pool, no_humans
THRESHOLD = 0.35  # default cutoff, the tagger's usual one; override per tag with "| 0.5" in tags.txt
RATINGS = {"general": "safe", "sensitive": "suggestive", "questionable": "questionable", "explicit": "explicit"}  # tagger's own levels
GIRLS = ["1girl", "2girls", "3girls", "4girls", "5girls", "6+girls", "multiple_girls"]
THUMB = 360
PAGE = 33  # posts per listing request after the first (always 12); the web app asks for 12, Instagram also answers 33
_say = threading.Lock()


def say(line):
    """Print a progress line. Listing, downloads and tagging print at the same time; one write each keeps lines whole."""
    with _say:
        print(line, flush=True)


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


def small_url(media):
    """The smallest copy Instagram sent whose long side still covers the tagger's 448 px input (previews are 360 px).
    It sends ~14 copies, from the original upload (often 2500+ px, ~300 KB) down, plus square crops to leave out."""
    sizes = (media.get("image_versions2") or {}).get("candidates", [])
    if not sizes:
        return None
    big = max(sizes, key=lambda c: c["width"] * c["height"])
    shape = big["width"] / big["height"]
    fits = [c for c in sizes if max(c["width"], c["height"]) >= 448 and abs(c["width"] / c["height"] - shape) < 0.02 * shape]
    return min(fits, key=lambda c: c["width"])["url"]


def list_photos(handle, limit=None, known=(), state=None):
    """(url, item) for every photo slide of the newest `limit` posts (None = all) as their page arrives; videos are skipped.
    known: shortcodes already in her library. state: what earlier runs listed, {"listed": posts gone through,
    "to_end": reached her first post}; when that covers what's asked, listing stops once it's back among known posts.
    It's updated in place."""
    state = state if state is not None else {"listed": 0, "to_end": False}
    L = login()
    found = i = 0
    try:
        profile = instaloader.Profile.from_username(L.context, handle)
        total = min(limit or profile.mediacount, profile.mediacount)
        covered = state["to_end"] if limit is None else state["listed"] >= limit
        streak = 0  # known posts in a row: at most 3 are pinned at the top, so 4 means we're back among old posts
        posts = profile.get_posts()
        # shortcut: Instaloader has no page-size setting, so these are its internals; same request, bigger answer
        instaloader.NodeIterator._graphql_page_length = PAGE
        if hasattr(posts, "_query_variables"):
            posts._query_variables["data"]["count"] = PAGE
        for i, post in enumerate(islice(posts, limit), 1):
            if post.typename != "GraphVideo":  # videos aren't kept, so they neither count nor break a streak
                streak = streak + 1 if post.shortcode in known else 0
            # shortcut: Instaloader keeps the post as Instagram sent it, with every image size, only in this internal field
            media = getattr(post, "_node", {}).get("iphone_struct") or {}
            slides = media.get("carousel_media") or [media]
            nodes = (post.get_sidecar_nodes() if post.typename == "GraphSidecar"
                     else [instaloader.PostSidecarNode(post.is_video, post.url, None)])
            for num, node in enumerate(nodes, 1):
                if not node.is_video:
                    found += 1
                    yield (small_url(slides[num - 1] if num <= len(slides) else {}) or node.display_url,
                           {"c": post.shortcode, "n": num, "l": post.likes or 0, "d": post.date_utc.date().isoformat(),
                            "f": f"{post.shortcode}_{num}.jpg"})
            if covered and streak >= 4:
                say(f"  {i}/{total} posts listed, {found} photos found: the rest are already in her library")
                break
            if i % 12 == 0 or i == total:
                say(f"  {i}/{total} posts listed, {found} photos found")
        else:
            state["to_end"] |= limit is None or i < limit  # the profile ran out, not the limit
    except instaloader.exceptions.InstaloaderException as e:  # rate limit or expired login halfway: keep what we got
        say(f"Instagram stopped the listing: {e}")
    state["listed"] = max(state["listed"], i)
    if not found and not known:
        sys.exit("No photos found (expired session.txt? private profile? wrong handle?)")


_http = threading.local()  # one kept-open connection per download thread, instead of a new handshake per photo


def prepare(job):
    """Fetch one photo into memory and save its preview: True, or the error text if it couldn't be fetched."""
    url, thumb = job
    try:
        with Image.open(thumb) as im:  # fetched on an earlier run: reuse the preview, skip the download
            im.load()
        return True
    except OSError:  # not there yet, or half-written by a stopped run
        pass
    try:
        if not hasattr(_http, "session"):
            _http.session = requests.Session()
        r = _http.session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:  # one dead link shouldn't sink a 2,000-photo run
        say(f"skipped {thumb.name}: {e}")
        return str(e) or type(e).__name__
    img.thumbnail((THUMB, THUMB))
    img.save(thumb, quality=80)
    return True


def failure(error):
    """A download error in a few words, for the run's summary."""
    if m := re.match(r"(\d{3}) (?:Client|Server) Error", error):  # requests' wording, e.g. "403 Client Error: Forbidden"
        return f"HTTP {m[1]}"
    for needle, words in [("10060", "timed out"), ("timed out", "timed out"), ("10054", "dropped by the server"),
                          ("reset", "dropped by the server"), ("getaddrinfo", "no internet"), ("HTTP Error", None)]:
        if needle in error:
            return words or error.split(":")[0].replace("HTTP Error", "HTTP")
    return error[:40]


class Tagger:
    """The booru tagger. It starts loading the moment it's made, in the background: PyTorch and the model take
    25-35 s to load, and listing posts doesn't need them."""

    def __init__(self):
        self.prep = ThreadPoolExecutor(8)  # decoding and resizing photos one by one kept the GPU waiting
        self.loaded = ThreadPoolExecutor(1).submit(self._load)

    @staticmethod
    def _load():
        import timm  # imported here rather than at the top, for the same reason
        import torch
        from timm.data import create_transform, resolve_data_config
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model = timm.create_model("hf-hub:" + MODEL, pretrained=True).eval().to(dev, dtype)
        return torch, model, create_transform(**resolve_data_config(model.pretrained_cfg, model=model)), dev, dtype

    def score(self, thumbs):
        """Probability of every booru tag for each preview, and its image features: the 1024 numbers the tagger's last
        layer turns into tags, which "same look" matching compares. The tagger wants white-padded squares in BGR."""
        torch, model, transform, dev, dtype = self.loaded.result()
        x = torch.stack(list(self.prep.map(
            lambda f: transform(ImageOps.pad(Image.open(f).convert("RGB"), (THUMB, THUMB), color="white"))[[2, 1, 0]], thumbs)))
        with torch.no_grad():
            seen = model.forward_features(x.to(dev, dtype))
            return (torch.sigmoid(model.forward_head(seen).float()).cpu().numpy().astype(np.float16),
                    model.forward_head(seen, pre_logits=True).float().cpu().numpy().astype(np.float16))


def score_all(tagger, thumbs):
    """Tag scores and image features of many previews, 16 at a time."""
    wd, feat = [], []
    for i in range(0, len(thumbs), 16):
        w, f = tagger.score(thumbs[i:i + 16])
        wd.append(w)
        feat.append(f)
        if (i // 16) % 10 == 9 or i + 16 >= len(thumbs):
            say(f"  {min(i + 16, len(thumbs))}/{len(thumbs)} photos scored")
    return np.concatenate(wd), np.concatenate(feat)


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


def saved_set(path):
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def remove_photos(folder, files):
    """Delete photos from a model's library (preview, scores, entry). They're remembered in removed.json,
    so a re-run never downloads them again. Returns how many were removed."""
    items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
    keep = [k for k, it in enumerate(items) if it["f"] not in files]
    gone = [it["f"] for it in items if it["f"] in files]
    for f in gone:
        (folder / "thumbs" / f).unlink(missing_ok=True)
    (folder / "removed.json").write_text(json.dumps(sorted(saved_set(folder / "removed.json") | set(gone))), encoding="utf-8")
    for name in ["wd.npy", "feat.npy"]:  # one row per items.json entry, so they all shrink together
        if (folder / name).exists():
            np.save(folder / name, np.load(folder / name)[keep])
    (folder / "items.json").write_text(json.dumps([items[k] for k in keep]), encoding="utf-8")
    data = json.loads((folder / "data.json").read_text(encoding="utf-8"))
    data["items"] = [it for it in data["items"] if it["f"] not in files]
    (folder / "data.json").write_text(json.dumps(data), encoding="utf-8")
    return len(gone)


def add_scores(data, folder, cols, general):
    """Per shown photo: the tagger's most likely explicitness level as a tag, a 0-3 score for sorting,
    "b": every general booru tag it saw (score >= THRESHOLD) as [column, percent], strongest first,
    and "e": its image features at unit length as int8 in base64, for "same look" matching on the page."""
    if not (folder / "wd.npy").exists():
        return
    wd = np.load(folder / "wd.npy").astype(np.float32)
    feat = np.load(folder / "feat.npy").astype(np.float32) if (folder / "feat.npy").exists() else None
    row = {it["f"]: k for k, it in enumerate(json.loads((folder / "items.json").read_text(encoding="utf-8")))}
    for it in data["items"]:
        w = wd[row[it["f"]]]
        p = w[cols]
        it["t"] = it["t"] + [list(RATINGS.values())[p.argmax()]]
        it["x"] = round(float(p @ np.arange(len(p)) / p.sum()), 3)
        seen = general[w[general] >= THRESHOLD]
        it["b"] = [[int(j), round(float(w[j]) * 100)] for j in seen[np.argsort(-w[seen])]]
        if feat is not None:
            e = feat[row[it["f"]]]
            it["e"] = base64.b64encode(np.round(e / np.linalg.norm(e) * 127).astype(np.int8).tobytes()).decode()
    data["groups"] = [["Rating", list(RATINGS.values())]] + data["groups"]


def build_pages():
    """Write every out/<handle>/index.html and the all-models out/index.html from the saved data.json files (no re-tagging)."""
    template = (HERE / "viewer.html").read_text(encoding="utf-8")

    def write(path, data):
        path.write_text(template.replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")

    items, groups = [], {"Model": []}
    with open(hf_hub_download(MODEL, "selected_tags.csv"), encoding="utf-8") as f:
        tags_csv = list(csv.DictReader(f))
    vocab = [r["name"] for r in tags_csv]  # the page looks up each photo's "b" columns here
    general = np.array([j for j, r in enumerate(tags_csv) if r["category"] == "0"])  # not ratings or anime characters
    col = {name: j for j, name in enumerate(vocab)}
    for f in sorted((HERE / "out").glob("*/data.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        add_scores(d, f.parent, [col[r] for r in RATINGS], general)
        write(f.parent / "index.html", {**d, "vocab": vocab})
        groups["Model"].append("@" + d["handle"])
        items += [{**it, "h": d["handle"], "t": it["t"] + ["@" + d["handle"]]} for it in d["items"]]
        for g, tags in d["groups"]:  # profiles tagged with an older tags.txt keep their old tag names
            known = groups.setdefault(g, [])
            known += [t for t in tags if t not in known]
    write(HERE / "out" / "index.html", {"title": "All models", "all": True, "items": items, "groups": list(groups.items()), "vocab": vocab})


def fetch(handle, limit, out, known, never, state, tagger, tagged, scores):
    """List her posts and, while the listing waits between pages, download each photo it finds and tag it once it's
    in, so the three steps overlap instead of taking turns. Returns the photos fetched, newest first.
    never: files not to fetch again. tagged: files whose scores are saved already. scores gets the new ones."""
    say("Step 1/2: listing posts; each photo is downloaded and tagged as soon as it's found")
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    photos, results, waiting = {}, {}, Queue()  # file -> (url, item); file -> True or why it failed; downloaded, to tag
    lock, queued = threading.Lock(), Counter()

    def keep(f, result):  # record a download and queue it for tagging; returns how many are done
        with lock:
            results[f] = result
            if result is True and f not in tagged:
                queued["photos"] += 1
                waiting.put(f)
            return len(results)

    def downloaded(future, f):
        if keep(f, future.result()) % 25 == 0:
            say(f"  {len(results)}/{len(photos)} photos downloaded")

    def tag_downloads():  # on its own thread: tags what has downloaded, up to 16 at a time, until it gets None
        while True:
            batch = [waiting.get()]
            while batch[-1] is not None and len(batch) < 16 and not waiting.empty():
                batch.append(waiting.get())
            files = [f for f in batch if f is not None]
            if files:
                scores.update(zip(files, zip(*tagger.score([out / "thumbs" / f for f in files]))))
            if files and len(scores) // 16 > (len(scores) - len(files)) // 16 or batch[-1] is None and scores:
                say(f"  {len(scores)}/{queued['photos']} photos scored")
            if batch[-1] is None:
                return

    tagging = threading.Thread(target=tag_downloads, daemon=True)  # daemon: a run that stops early doesn't wait for it
    tagging.start()
    with ThreadPoolExecutor(8) as pool:
        for url, it in list_photos(handle, limit, known, state):
            if it["f"] not in never and it["f"] not in photos:  # also dedupes repeats
                photos[it["f"]] = url, it
                pool.submit(prepare, (url, out / "thumbs" / it["f"])).add_done_callback(lambda fut, f=it["f"]: downloaded(fut, f))
    say(f"  {len(results)}/{len(photos)} photos downloaded")
    retry = [f for f, r in results.items() if r is not True]
    if retry:  # failures come in bursts when Instagram's image servers push back: pause, then go slower
        say(f"  retrying {len(retry)} failed downloads after a 15 s pause, 4 at a time")
        time.sleep(15)
        with ThreadPoolExecutor(4) as pool:
            for done, (f, r) in enumerate(zip(retry, pool.map(prepare, [(photos[f][0], out / "thumbs" / f) for f in retry])), 1):
                keep(f, r)
                if done % 10 == 0 or done == len(retry):
                    say(f"  {done}/{len(retry)} failed photos retried")
    ok = [f for f in photos if results.get(f) is True]
    why = Counter(failure(r) for r in results.values() if r is not True)
    say(f"  {len(ok)}/{len(photos)} photos downloaded, {len(photos) - len(ok)} failed"
        + (": " + ", ".join(f"{n} {w}" for w, n in why.most_common()) if why else ""))
    waiting.put(None)  # nothing more to tag after what's queued
    if len(ok) < len(photos) / 2:  # e.g. Wi-Fi dropped mid-run: don't replace a good earlier run with a handful of photos
        sys.exit(f"Only {len(ok)} of {len(photos)} photos could be downloaded (network down?). Nothing saved; run it again.")
    tagging.join()
    return [photos[f][1] for f in ok]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    handle = handle_from(sys.argv[1])
    limit = next((int(a) for a in sys.argv[2:] if a.isdigit()), None)
    out = HERE / "out" / handle
    skipped_file = out / "skipped.json"  # photos the pre-screen threw out: never downloaded again
    skipped = saved_set(skipped_file)
    removed = saved_set(out / "removed.json")  # photos you deleted: never downloaded again either
    old = json.loads((out / "items.json").read_text(encoding="utf-8")) if (out / "items.json").exists() else []
    saved = json.loads((out / "data.json").read_text(encoding="utf-8")) if (out / "data.json").exists() else {}
    state = {"listed": saved.get("listed", 0), "to_end": saved.get("to_end", False)}
    rows = {}  # file -> its row in the saved scores and features: a re-run keeps those instead of tagging again
    if (out / "wd.npy").exists() and (out / "feat.npy").exists():
        wd_old, feat_old = np.load(out / "wd.npy"), np.load(out / "feat.npy")
        if len(wd_old) == len(feat_old) == len(old):
            rows = {it["f"]: k for k, it in enumerate(old)}
    scores, tagger, items = {}, None, old  # scores: file -> (tag scores, image features) of photos tagged on this run
    if "--retag" not in sys.argv:
        tagger = Tagger()  # starts loading now, while posts are listed
        known = {f.rsplit("_", 1)[0] for f in [it["f"] for it in old] + sorted(skipped | removed)}
        new = fetch(handle, limit, out, known, skipped | removed, state, tagger, rows, scores)
        fresh = {it["f"] for it in new}  # keep earlier photos this run didn't reach, e.g. a re-run limited to the newest posts
        items = new + [it for it in old if it["f"] not in fresh and (out / "thumbs" / it["f"]).exists()]
    if not items:
        sys.exit("No photos in her library")
    say("Step 2/2: tagging the last photos, skipping photos without a girl, building pages")
    todo = [it["f"] for it in items if it["f"] not in scores and it["f"] not in rows]
    if todo:
        scores.update(zip(todo, zip(*score_all(tagger or Tagger(), [out / "thumbs" / f for f in todo]))))
    wd = np.stack([scores[it["f"]][0] if it["f"] in scores else wd_old[rows[it["f"]]] for it in items])
    feat = np.stack([scores[it["f"]][1] if it["f"] in scores else feat_old[rows[it["f"]]] for it in items])
    (out / "items.json").write_text(json.dumps(items), encoding="utf-8")
    np.save(out / "wd.npy", wd)
    np.save(out / "feat.npy", feat)
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
        np.save(out / "feat.npy", feat[~drop])
    for it, t in zip(items, tags):
        it["t"] = t
    groups = [(g, group) for g, group in groups if g.lower() != "skip"]
    data = {"handle": handle, "items": items, "skipped": len(skipped), "groups": [[g, [name for name, _, _ in group]] for g, group in groups],
            **state}
    (out / "data.json").write_text(json.dumps(data), encoding="utf-8")
    print("  building pages", flush=True)
    build_pages()
    for g, group in groups:
        print(f"{g}: " + ", ".join(f"{name} {sum(name in it['t'] for it in items)}" for name, _, _ in group))
    print(f"{len(items)} photos -> {out / 'index.html'} | {int(drop.sum())} without a girl skipped and deleted now, {len(skipped)} in total")
    print(f"All models: {HERE / 'out' / 'index.html'}")


if __name__ == "__main__":
    main()
