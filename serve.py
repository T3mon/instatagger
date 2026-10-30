"""Serve the pages with an "Add model" button:  python serve.py, then open http://localhost:8765"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from statistics import median
from urllib.parse import parse_qs, urlparse

import instaloader

import instatagger

HERE = Path(__file__).parent
PORT = 8765
run = {"handle": None, "proc": None, "log": [], "started": 0, "ended": None}  # one run at a time: Instagram sees one listing


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
    info = {"handle": p.username, "name": p.full_name, "posts": p.mediacount, "followers": p.followers, "verified": p.is_verified,
            "private": p.is_private and not p.followed_by_viewer,
            "have": len(json.loads(data.read_text(encoding="utf-8"))["items"]) if data.exists() else 0}
    if not info["private"] and p.mediacount:
        info.update(recent_stats(list(islice(p.get_posts(), 12)), p.followers))
    return info


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
        if url.path == "/add":  # the Add model page, separate so a run never blocks browsing
            body = (HERE / "add.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
            return self.reply(200, {"handle": run["handle"], "running": bool(proc and proc.poll() is None),
                                    "code": proc.poll() if proc else None, "elapsed": round(elapsed), "log": run["log"]})
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
        if self.path != "/api/add":
            return self.reply(404, {"error": "Not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or "{}")
        try:
            handle = instatagger.handle_from(str(body.get("u", "")))
        except SystemExit as e:
            return self.reply(400, {"error": str(e)})
        limit = body.get("limit")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            return self.reply(400, {"error": "The post limit must be a whole number, 1 or more"})
        if run["proc"] and run["proc"].poll() is None:
            return self.reply(409, {"error": f"Still adding @{run['handle']}: wait for it to finish"})
        start(handle, [sys.executable, "-u", str(HERE / "instatagger.py"), handle] + ([str(limit)] if limit else []))
        self.reply(200, {"started": handle})


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
