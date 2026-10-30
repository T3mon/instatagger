"""Tag every photo on an Instagram profile and build a page to search them by tag.

    python instatagger.py <handle or profile link> [N]   newest N posts (default all): list, fetch, tag, build the page
    python instatagger.py <handle> --retag               re-score after editing tags.txt (no Instagram calls)

Needs session.txt next to this file, holding your Instagram `sessionid` cookie value.
Full-size photos only pass through memory. out/<handle>/ keeps small previews,
the CLIP embeddings and booru-tag scores (what makes --retag instant) and index.html.
"""
import csv
import json
import random
import re
import secrets
import sys
import time
import urllib.request
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
from transformers import CLIPModel, CLIPProcessor

HERE = Path(__file__).parent
MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # LAION data kept swimwear/lingerie, so it knows these tags
BASELINE = "a photo of a woman"  # the "none of these" option every group competes with
THRESHOLD = 0.25  # share of its group a tag must win; override per tag with "| 0.4" in tags.txt
WD_MODEL = "SmilingWolf/wd-eva02-large-tagger-v3"  # booru tagger: trained on labels like underboob that CLIP can't tell apart
WD_THRESHOLD = 0.35  # default cutoff for "wd:" tags, the tagger's usual one
THUMB = 360
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def handle_from(text):
    handle = text.split("instagram.com/")[-1].split("/")[0].split("?")[0].lstrip("@")
    if not re.fullmatch(r"\w(?:[\w.]{0,28}\w)?", handle):  # also rejects "..": the handle becomes a folder name
        sys.exit(f"Not an Instagram handle: {text}")
    return handle


def parse_tags(text):
    """'[Group]' starts a group; 'tag: description' or 'tag: description | cutoff' adds a tag."""
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
        prompt, _, cutoff = rest.partition("|")
        prompt = prompt.strip()
        default = WD_THRESHOLD if prompt.startswith("wd:") else THRESHOLD
        groups[-1][1].append((tag.strip(), prompt, float(cutoff) if cutoff.strip() else default))
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
    """Fetch one photo into memory, save its preview, return it padded square for the model."""
    url, thumb = job
    try:
        if thumb.exists():  # fetched on an earlier run: reuse the preview, skip the download
            return ImageOps.pad(Image.open(thumb).convert("RGB"), (224, 224))
    except OSError:  # half-written by a stopped run: fetch it again
        pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            img = Image.open(BytesIO(r.read())).convert("RGB")
    except Exception as e:  # one dead link shouldn't sink a 2,000-photo run
        print(f"skipped {thumb.name}: {e}")
        return None
    square = ImageOps.pad(img, (224, 224))  # pad instead of CLIP's center crop, so heads and feet stay in
    img.thumbnail((THUMB, THUMB))
    img.save(thumb, quality=80)
    return square


def embed(model, proc, squares):
    out = []
    for i in range(0, len(squares), 64):
        px = proc(images=squares[i:i + 64], return_tensors="pt")["pixel_values"].to(DEV, model.dtype)
        with torch.no_grad():
            out.append(torch.nn.functional.normalize(model.get_image_features(pixel_values=px).pooler_output, dim=-1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def booru_scores(thumbs):
    """Probability of every booru tag for each preview; the WD tagger wants white-padded squares in BGR."""
    dtype = torch.float16 if DEV == "cuda" else torch.float32
    model = timm.create_model("hf-hub:" + WD_MODEL, pretrained=True).eval().to(DEV, dtype)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    out = []
    for i in range(0, len(thumbs), 16):
        x = torch.stack([transform(ImageOps.pad(Image.open(f).convert("RGB"), (THUMB, THUMB), color="white"))
                         for f in thumbs[i:i + 16]])[:, [2, 1, 0]]
        with torch.no_grad():
            out.append(torch.sigmoid(model(x.to(DEV, dtype)).float()).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def tag(model, proc, emb, wd, groups):
    """CLIP tags compete within their group and with BASELINE, firing when their share >= cutoff.
    "wd:" tags fire on their own when any of their booru tags scores >= cutoff."""
    img = torch.from_numpy(emb).float()
    with open(hf_hub_download(WD_MODEL, "selected_tags.csv"), encoding="utf-8") as f:
        col = {row["name"]: j for j, row in enumerate(csv.DictReader(f))}
    names, hits = [], []
    for _, tags in groups:
        clip = [(name, p, c) for name, p, c in tags if not p.startswith("wd:")]
        if clip:
            txt = proc(text=[BASELINE] + [p for _, p, _ in clip], return_tensors="pt", padding=True).to(DEV)
            with torch.no_grad():
                t = torch.nn.functional.normalize(model.get_text_features(**txt).pooler_output, dim=-1).float().cpu()
            share = (model.logit_scale.exp().item() * img @ t.T).softmax(dim=-1)[:, 1:]
            clip_hits = dict(zip([name for name, _, _ in clip], (share >= torch.tensor([c for _, _, c in clip])).T))
        for name, p, c in tags:
            if p.startswith("wd:"):
                booru = p[3:].split()
                if missing := [b for b in booru if b not in col]:
                    sys.exit(f"{name}: the booru tagger has no tag {missing}")
                hits.append(torch.from_numpy(wd[:, [col[b] for b in booru]].max(axis=1) >= c))
            else:
                hits.append(clip_hits[name])
            names.append(name)
    return [[names[j] for j in row.nonzero().flatten().tolist()] for row in torch.stack(hits, dim=1)]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    handle = handle_from(sys.argv[1])
    limit = next((int(a) for a in sys.argv[2:] if a.isdigit()), None)
    out = HERE / "out" / handle
    model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
    proc = CLIPProcessor.from_pretrained(MODEL)
    if "--retag" in sys.argv:
        items = json.loads((out / "items.json").read_text(encoding="utf-8"))
        emb = np.load(out / "emb.npy")
    else:
        print("Listing posts, one page of 12 every 6-12 s...", flush=True)
        photos = {it["f"]: (url, it) for url, it in list_photos(handle, limit)}  # dedupes repeats
        (out / "thumbs").mkdir(parents=True, exist_ok=True)
        items = [it for _, it in photos.values()]
        print(f"Fetching and tagging {len(items)} photos...", flush=True)
        jobs = [(url, out / "thumbs" / f) for f, (url, _) in photos.items()]
        with ThreadPoolExecutor(8) as pool:
            squares = list(pool.map(prepare, jobs))
            retry = [k for k, sq in enumerate(squares) if sq is None]  # timeouts come in bursts when the line is busy
            for k, sq in zip(retry, pool.map(prepare, [jobs[k] for k in retry])):
                squares[k] = sq
        items = [it for it, sq in zip(items, squares) if sq is not None]
        emb = embed(model, proc, [sq for sq in squares if sq is not None])
        (out / "items.json").write_text(json.dumps(items), encoding="utf-8")
        np.save(out / "emb.npy", emb)
        (out / "wd.npy").unlink(missing_ok=True)  # new set of photos: booru scores are recomputed below
    if not (out / "wd.npy").exists():
        np.save(out / "wd.npy", booru_scores([out / "thumbs" / it["f"] for it in items]))
    wd = np.load(out / "wd.npy")
    groups = parse_tags((HERE / "tags.txt").read_text(encoding="utf-8"))
    for it, tags in zip(items, tag(model, proc, emb, wd, groups)):
        it["t"] = tags
    skip = {name for g, tags in groups if g.lower() == "skip" for name, _, _ in tags}
    shown = [it for it in items if not skip.intersection(it["t"])]  # posts with no person: text slides, products...
    data = {"handle": handle, "items": shown,
            "groups": [[g, [name for name, _, _ in tags]] for g, tags in groups if g.lower() != "skip"]}
    page = (HERE / "viewer.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(page.replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    for g, tags in groups:
        counted = items if g.lower() == "skip" else shown
        print(f"{g}: " + ", ".join(f"{name} {sum(name in it['t'] for it in counted)}" for name, _, _ in tags))
    print(f"{len(shown)} photos shown, {len(items) - len(shown)} skipped -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
