"""Serve the pages with an "Add model" button:  python serve.py, then open http://localhost:8765"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import instaloader

import instatagger

HERE = Path(__file__).parent
PORT = 8765
run = {"handle": None, "proc": None, "log": []}  # one run at a time, so Instagram only ever sees one slow listing


def profile(handle):
    """Public stats of an account (one Instagram request), plus how many of her photos are already here."""
    p = instaloader.Profile.from_username(instatagger.login(None).context, handle)
    data = HERE / "out" / p.username / "data.json"
    return {"handle": p.username, "name": p.full_name, "posts": p.mediacount, "followers": p.followers,
            "following": p.followees, "verified": p.is_verified, "bio": p.biography,
            "private": p.is_private and not p.followed_by_viewer,
            "have": len(json.loads(data.read_text(encoding="utf-8"))["items"]) if data.exists() else 0}


def start(handle, cmd):
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    run.update(handle=handle, proc=proc, log=[], started=time.time(), ended=None)

    def read():
        for line in proc.stdout:
            if not line.startswith("skipped "):  # per-photo download errors would push the progress lines out
                run["log"].append(line.rstrip())
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
        if not url.path.startswith("/api/"):
            return super().do_GET()
        if not self.allowed():
            return self.reply(403, {"error": "Forbidden"})
        if url.path == "/api/status":
            proc = run["proc"]
            elapsed = (run["ended"] or time.time()) - run["started"] if proc else 0
            return self.reply(200, {"handle": run["handle"], "running": bool(proc and proc.poll() is None),
                                    "code": proc.poll() if proc else None, "elapsed": round(elapsed), "log": run["log"][-40:]})
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
