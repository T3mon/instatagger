"""Serve the pages with the Add model and Models pages:  python serve.py, then open http://localhost:8765"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import date, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from statistics import mean, median
from urllib.parse import parse_qs, urlparse

import instaloader
import numpy as np

import instatagger

HERE = Path(__file__).parent
PORT = 8765
run = {"handle": None, "proc": None, "log": [], "started": 0, "ended": None}  # one run at a time: Instagram sees one listing
files_lock, stale = threading.Lock(), threading.Event()  # deletes and page rebuilds take turns on the library files


def recent_stats(posts, followers):
    """What her newest posts say. Only fields that come with the feed page: reading comments costs a request per post,
    and Instagram answered that with "feedback_required"."""
    posts = [x for x in posts if not getattr(x, "is_pinned", False)]  # pinned posts can be years old
    if not posts:
        return {}
    photo = [x for x in posts if not x.is_video]
    days = max((posts[0].date_utc - posts[-1].date_utc).total_seconds() / 86400, 1)
    likes = int(median(x.likes for x in posts))
    return {"recent": len(posts), "last_post": posts[0].date_utc.date().isoformat(), "per_week": round(len(posts) / days * 7, 1),
            "photo_posts": len(photo), "photos_per_post": round(sum(x.mediacount for x in photo) / len(photo), 1) if photo else 0,
            "likes": None if likes <= 3 else likes,  # Instagram reports 3 when likes are hidden
            "like_rate": None if likes <= 3 else round(likes / max(followers, 1) * 100, 2)}


def profile(handle):
    """Her account (one Instagram request), her newest page of posts (one more) and how much of her is already here."""
    p = instaloader.Profile.from_username(instatagger.login(None).context, handle)
    data = HERE / "out" / p.username / "data.json"
    saved = json.loads(data.read_text(encoding="utf-8")) if data.exists() else {}
    info = {"handle": p.username, "name": p.full_name, "posts": p.mediacount, "followers": p.followers, "verified": p.is_verified,
            "private": p.is_private and not p.followed_by_viewer, "have": len(saved.get("items", [])),
            "listed": saved.get("listed", 0), "to_end": saved.get("to_end", False)}  # how far her earlier runs listed
    if not info["private"] and p.mediacount:
        info.update(recent_stats(list(islice(p.get_posts(), 12)), p.followers))
    return info


def walk(folder):
    """Every file under folder. On Windows the folder listing already carries each file's size,
    so this skips a disk check per file (8.5 s of the Models page's 10 on an exFAT drive)."""
    for e in os.scandir(folder):
        yield from walk(e.path) if e.is_dir() else [e]


def model_stats(folder, cols):
    """Everything the Models page shows about one model, from her saved files (no Instagram calls)."""
    d = json.loads((folder / "data.json").read_text(encoding="utf-8"))
    instatagger.add_scores(d, folder, cols, np.array([], int))  # adds the rating; no booru tag list needed here
    items, levels = d["items"], list(instatagger.RATINGS.values())
    seen = [it for it in items if it["l"] > 3]  # Instagram reports 3 when likes are hidden
    mid = median(it["l"] for it in seen) if seen else None
    dates = sorted(it["d"] for it in items)
    posts = len({it["c"] for it in items})
    weeks = max((date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 7, 1) if dates else 1
    tags = Counter(t for it in items for t in it["t"] if t not in levels)
    liked = Counter(t for it in seen for t in it["t"] if t not in levels)
    lift = {t: median(it["l"] for it in seen if t in it["t"]) / mid for t, n in liked.items() if n >= 10}  # vs her median
    scored = bool(items) and all("x" in it for it in items)
    files = list(walk(folder))
    by_post = {it["c"]: it for it in sorted(seen, key=lambda it: (it["l"], -it["n"]))}  # a post's first slide, least liked first
    show = list(by_post.values())[::-1][:4] or sorted(items, key=lambda it: it["d"])[-4:]
    return {"handle": d["handle"], "photos": len(items), "posts": posts, "skipped": d["skipped"],
            "oldest": dates[0] if dates else None, "newest": dates[-1] if dates else None, "per_week": round(posts / weeks, 1),
            "likes": mid, "likes_avg": round(mean(it["l"] for it in seen)) if seen else None,
            "likes_max": max((it["l"] for it in seen), default=None), "hidden": len(items) - len(seen),
            "rating": {r: sum(r in it["t"] for it in items) for r in levels} if scored else None,
            "spice": round(mean(it["x"] for it in items) / 3 * 100) if scored else None,
            "top": tags.most_common(8), "lift": sorted(([t, round(x, 2)] for t, x in lift.items()), key=lambda r: -r[1])[:5],
            "best": [{k: it[k] for k in "cnfl"} for it in show], "years": dict(sorted(Counter(d[:4] for d in dates).items())),
            "removed": len(instatagger.saved_set(folder / "removed.json")),
            "thumbs": instatagger.disk_size([f for f in files if Path(f.path).parent.name == "thumbs"]), "disk": instatagger.disk_size(files),
            "files": len(files), "updated": datetime.fromtimestamp((folder / "data.json").stat().st_mtime).isoformat(timespec="minutes")}


def settings():
    session = HERE / "session.txt"  # only its date: the file holds your Instagram login
    return {"tagger": instatagger.MODEL, "cutoff": instatagger.THRESHOLD, "folder": str(HERE / "out"),
            "groups": [[g, len(t)] for g, t in instatagger.parse_tags((HERE / "tags.txt").read_text(encoding="utf-8")) if g != "Skip"],
            "session": datetime.fromtimestamp(session.stat().st_mtime).date().isoformat() if session.exists() else None,
            "free": shutil.disk_usage(HERE).free}


def rebuild_later():
    """Rebuild the pages in the background after photo deletes; deletes in a row share one rebuild."""
    def go():
        with files_lock:
            if stale.is_set():
                stale.clear()
                instatagger.build_pages()
    stale.set()
    threading.Thread(target=go, daemon=True).start()


def busy():
    return bool(run["proc"] and run["proc"].poll() is None)


def start(handle, cmd):
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    run.update(handle=handle, proc=proc, log=[], started=time.time(), ended=None)

    def read():
        for line in proc.stdout:
            if not line.startswith("skipped "):  # per-photo download errors; the run prints a total
                run["log"].append([round(time.time() - run["started"]), line.rstrip()])
        run["ended"] = time.time()

    threading.Thread(target=read, daemon=True).start()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE / "out"), **kwargs)

    def log_message(self, *args):  # the page polls every 2 s; keep the console quiet
        pass

    def reply(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        # Other sites can't send this header without a CORS preflight, which this server never answers,
        # so they can't make it look someone up or start a run with your Instagram login
        return self.headers.get("X-Instatagger") == "1"

    def do_GET(self):
        url = urlparse(self.path)
        pages = {"/add": "add.html", "/models": "models.html", "/favicon.ico": "favicon.ico"}  # separate pages, so a run never blocks browsing
        if url.path in pages:
            body = (HERE / pages[url.path]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/x-icon" if url.path == "/favicon.ico" else "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        if not self.allowed():
            return self.reply(403, {"error": "Forbidden"})
        if url.path == "/api/status":
            proc = run["proc"]
            elapsed = (run["ended"] or time.time()) - run["started"] if proc else 0
            return self.reply(200, {"handle": run["handle"], "running": busy(),
                                    "code": proc.poll() if proc else None, "elapsed": round(elapsed), "log": run["log"]})
        if url.path == "/api/models":
            cols = [instatagger.booru_columns()[r] for r in instatagger.RATINGS]
            return self.reply(200, {"models": [model_stats(f.parent, cols) for f in sorted((HERE / "out").glob("*/data.json"))],
                                    "settings": settings(), "running": run["handle"] if busy() else None})
        if url.path == "/api/profile":
            try:
                return self.reply(200, profile(instatagger.handle_from(parse_qs(url.query).get("u", [""])[0])))
            except SystemExit as e:  # bad handle or no session.txt
                return self.reply(400, {"error": str(e)})
            except instaloader.exceptions.InstaloaderException as e:
                return self.reply(400, {"error": f"Instagram: {e}"})
        self.reply(404, {"error": "Not found"})

    def do_POST(self):
        if not self.allowed():
            return self.reply(403, {"error": "Forbidden"})
        if self.path not in ("/api/add", "/api/delete"):
            return self.reply(404, {"error": "Not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or "{}")
        try:
            handle = instatagger.handle_from(str(body.get("u", "")))
        except SystemExit as e:
            return self.reply(400, {"error": str(e)})
        if self.path == "/api/delete":
            folder = HERE / "out" / handle
            if not (folder / "data.json").exists():
                return self.reply(404, {"error": f"@{handle} isn't in the library"})
            years, files = body.get("years"), body.get("files")
            if busy() and (run["handle"] == handle or years is None and files is None):  # a run's last step reads every model
                return self.reply(409, {"error": f"Still adding @{run['handle']}: wait for it to finish"})
            if years is None and files is None:  # the whole model
                with files_lock:
                    try:
                        shutil.rmtree(folder)
                        error = None
                    except OSError as e:  # e.g. a file still open in another program
                        error = f"Some of her files couldn't be deleted: {e}"
                    instatagger.build_pages()  # her photos leave the all-models page
                return self.reply(500, {"error": error}) if error else self.reply(200, {"deleted": handle})
            if years is not None:  # [first, last] year, both included
                if not (isinstance(years, list) and len(years) == 2 and all(isinstance(y, int) for y in years)):
                    return self.reply(400, {"error": "years must be [first, last]"})
                items = json.loads((folder / "data.json").read_text(encoding="utf-8"))["items"]
                files = [it["f"] for it in items if years[0] <= int(it["d"][:4]) <= years[1]]
            elif not (isinstance(files, list) and all(isinstance(f, str) for f in files)):
                return self.reply(400, {"error": "files must be a list of photo file names"})
            with files_lock:
                n = instatagger.remove_photos(folder, set(files))  # only names in her library are touched
            rebuild_later()
            return self.reply(200, {"deleted": n})
        limit = body.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            return self.reply(400, {"error": "The post limit must be a whole number, 1 or more"})
        if busy():
            return self.reply(409, {"error": f"Still adding @{run['handle']}: wait for it to finish"})
        start(handle, [sys.executable, "-u", str(HERE / "instatagger.py"), handle] + ([str(limit)] if limit else []))
        self.reply(200, {"started": handle})


if __name__ == "__main__":
    ThreadingHTTPServer.allow_reuse_address = sys.platform != "win32"  # on Windows it lets a second copy listen on the same port, and pages talk to either one
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        sys.exit(f"Port {PORT} is already in use, probably by instatagger in another window: open http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT}  (Ctrl+C to stop)")
    server.serve_forever()
