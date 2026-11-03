"""Serve the pages with the Add model and Models pages:  python serve.py, then open http://localhost:8765"""
import json
import os
import re
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
import requests

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
    p = instatagger.find_profile(instatagger.login(None).context, handle)
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
    # only the ratings: add_scores also builds every photo's tag list and image features, 0.2-0.7 s a model
    rated = instatagger.ratings(folder, cols) if (folder / "wd.npy").exists() else {}
    for it in d["items"]:
        if it["f"] in rated:
            it["t"], it["x"] = it["t"] + [rated[it["f"]][0]], rated[it["f"]][1]
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
            "groups": [[g, len(t)] for g, t in instatagger.parse_tags((HERE / "tags.txt").read_text(encoding="utf-8")) if g.lower() != "skip"],
            "skip": skip_section()[1], "rules": instatagger.skip_rules(), "defaults": instatagger.SKIPS, "girls": instatagger.GIRLS,
            "session": datetime.fromtimestamp(session.stat().st_mtime).date().isoformat() if session.exists() else None,
            "free": shutil.disk_usage(HERE).free}


def skip_section():
    """tags.txt split around its [Skip] rules: (lines up to [Skip], the rules, the lines after)."""
    lines = (HERE / "tags.txt").read_text(encoding="utf-8").splitlines()
    start = next((k for k, l in enumerate(lines) if l.strip().lower() == "[skip]"), None)
    if start is None:
        return lines + ["", "[Skip]"], [], []
    end = next((k for k in range(start + 1, len(lines)) if lines[k].strip().startswith("[")), len(lines))
    return lines[:start + 1], [l.strip() for l in lines[start + 1:end] if l.strip() and not l.strip().startswith("#")], lines[end:]


def parse_skip(text):
    """Your own skip rules from the page, checked against the tagger's tag list: (lines for tags.txt, parsed rules).
    Raises ValueError saying what's wrong."""
    rules = [l.strip() for l in str(text).splitlines() if l.strip()]
    if any(l.startswith(("[", "#")) or ":" not in l for l in rules):
        raise ValueError("Write one rule per line: name: booru_tag ... | cutoff, e.g. boys: 1boy multiple_boys | 0.5")
    try:
        parsed = instatagger.parse_tags("\n".join(["[Skip]"] + rules))[0][1] if rules else []
    except ValueError:
        raise ValueError("The cutoff after | must be a number, e.g. 0.5") from None
    cols = instatagger.booru_columns()
    unknown = sorted({b for _, booru, _ in parsed for b in booru if b not in cols})
    if unknown:
        raise ValueError(f"Not in the tagger's tag list: {', '.join(unknown)}. Its tags use underscores, like multiple_boys")
    if any(not name or not booru for name, booru, _ in parsed):
        raise ValueError("Each rule needs a name and at least one booru tag")
    if any(not 0 < cutoff <= 1 for _, _, cutoff in parsed):
        raise ValueError("The cutoff must be between 0 and 1")
    return rules, parsed


def check_rules(rules):
    """The built-in skip rules sent by the page: every rule and field of instatagger.SKIPS, on/off and cut-offs 0-1."""
    if not (isinstance(rules, dict) and all(isinstance(rules.get(k), dict) and all(
            isinstance(rules[k].get(f), bool) if f == "on" else type(rules[k].get(f)) in (int, float) and 0 <= rules[k][f] <= 1
            for f in rule) for k, rule in instatagger.SKIPS.items())):
        raise ValueError("Each built-in rule needs on or off, and cut-offs from 0 to 1")
    return {k: {f: rules[k][f] for f in rule} for k, rule in instatagger.SKIPS.items()}


def skip_preview(rules, custom):
    """What these skip rules catch in each model's library now: [(her folder, her photos, file names caught)]."""
    found = []
    for f in sorted((HERE / "out").glob("*/items.json")):
        if busy() and f.parent.name == run["handle"]:
            continue  # her files are being rewritten
        items = json.loads(f.read_text(encoding="utf-8"))
        wd = np.load(f.parent / "wd.npy") if (f.parent / "wd.npy").exists() else None
        if wd is None or len(wd) != len(items):
            continue  # no saved scores to check
        drop = instatagger.no_girl(wd, rules)
        if custom:
            drop |= np.array([bool(t) for t in instatagger.tag(wd, [("Skip", custom)])], bool)
        found.append((f.parent, len(items), [it["f"] for it, d in zip(items, drop) if d]))
    return found


def save_skips(rules, lines, custom):
    """Save the skip rules (built-in ones in settings.json, yours in tags.txt) and delete the photos they catch now.
    Returns how many photos were deleted."""
    remember(skip=rules)
    before, _, after = skip_section()
    (HERE / "tags.txt").write_text("\n".join(before + lines + ([""] if after else []) + after) + "\n", encoding="utf-8")
    deleted = sum(instatagger.remove_photos(folder, set(files), "skipped") for folder, _, files in skip_preview(rules, custom) if files)
    if deleted:
        instatagger.build_pages()
    return deleted


def saved_settings():
    path = HERE / "settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def remember(**values):
    """Keep values in settings.json, next to what's there already."""
    (HERE / "settings.json").write_text(json.dumps({**saved_settings(), **values}, indent=1), encoding="utf-8")


save_window = threading.Lock()  # one Save As window at a time


def ask_where(name):
    """Where to save a download, picked in a Save As window the server opens on your desktop: a page can't make the
    browser ask, and Brave turns off the file-picker API pages could use. Starts in the folder used last; None if cancelled."""
    import tkinter  # only here: the server needs no window otherwise
    from tkinter import filedialog
    with save_window:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # in front of the browser
        try:
            return filedialog.asksaveasfilename(
                parent=root, title="Save the full-size photo", initialfile=name, defaultextension=".jpg",
                initialdir=str(saved_settings().get("download_dir") or Path.home() / "Downloads"),
                filetypes=[("JPEG image", "*.jpg"), ("All files", "*.*")]) or None
        finally:
            root.destroy()


def original(code, n):
    """Slide n of a post at the size it was uploaded, as (url, width, height). The gallery only keeps 360 px previews;
    Instagram's media info lists every size it has (one Instagram request)."""
    ctx = instatagger.login(None).context
    media = ctx.get_iphone_json(f"api/v1/media/{instaloader.Post.shortcode_to_mediaid(code)}/info/", {})["items"][0]
    slide = (media.get("carousel_media") or [media])[n - 1]
    big = max(slide["image_versions2"]["candidates"], key=lambda c: c["width"] * c["height"])
    return big["url"], big["width"], big["height"]


def check_login(session=None):
    """Your Instagram username if the sessionid works (one Instagram request), else None."""
    return instatagger.login(None, session=session).test_login()


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
        pages = {"/add": "add.html", "/models": "models.html", "/settings": "settings.html", "/favicon.ico": "favicon.ico"}  # separate pages, so a run never blocks browsing
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
            cols, models = [instatagger.booru_columns()[r] for r in instatagger.RATINGS], []
            for f in sorted((HERE / "out").glob("*/data.json")):
                try:
                    models.append(model_stats(f.parent, cols))
                except (OSError, ValueError, KeyError, EOFError):  # her files are being rewritten by a run or a delete
                    pass  # shortcut: a model whose files are broken for good disappears too; show it as an error card if that happens
            return self.reply(200, {"models": models, "settings": settings(), "running": run["handle"] if busy() else None})
        if url.path == "/api/settings":
            return self.reply(200, settings())
        if url.path == "/api/login":
            try:
                return self.reply(200, {"user": check_login()})
            except SystemExit as e:  # no session.txt
                return self.reply(400, {"error": str(e)})
            except instaloader.exceptions.InstaloaderException as e:
                return self.reply(400, {"error": f"Instagram: {e}"})
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
        if self.path not in ("/api/add", "/api/delete", "/api/download", "/api/settings", "/api/skip", "/api/skip/preview"):
            return self.reply(404, {"error": "Not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or "{}")
        if self.path.startswith("/api/skip"):
            try:
                rules, (lines, custom) = check_rules(body.get("rules")), parse_skip(body.get("custom", ""))
            except ValueError as e:
                return self.reply(400, {"error": str(e)})
            if self.path == "/api/skip/preview":
                return self.reply(200, {"models": [{"handle": folder.name, "photos": n, "drop": len(files), "sample": files[:12]}
                                                   for folder, n, files in skip_preview(rules, custom)]})
            if busy():  # the run re-checks and rewrites her files at its end
                return self.reply(409, {"error": f"Still adding @{run['handle']}: wait for it to finish"})
            with files_lock:
                deleted = save_skips(rules, lines, custom)
            return self.reply(200, {**settings(), "deleted": deleted})
        if self.path == "/api/settings":
            # pasted as the cookie's value, or as "sessionid=...;"; decoded colons are encoded back like the browser shows them
            session = str(body.get("session", "")).strip().strip('"').removeprefix("sessionid=").rstrip(";").replace(":", "%3A")
            if not re.fullmatch(r"[\w%.-]{20,500}", session):
                return self.reply(400, {"error": "That doesn't look like a sessionid: copy the Value of the sessionid cookie"})
            try:
                user = check_login(session)
            except instaloader.exceptions.InstaloaderException as e:
                return self.reply(400, {"error": f"Instagram: {e}"})
            if not user:
                return self.reply(400, {"error": "Instagram didn't accept it: copy a fresh one while logged in to instagram.com"})
            (HERE / "session.txt").write_text(session, encoding="utf-8")
            return self.reply(200, {"user": user})
        try:
            handle = instatagger.handle_from(str(body.get("u", "")))
        except SystemExit as e:
            return self.reply(400, {"error": str(e)})
        if self.path == "/api/download":
            code, n = str(body.get("c", "")), body.get("n")
            if not re.fullmatch(r"[\w-]{5,40}", code) or type(n) is not int or n < 1:
                return self.reply(400, {"error": "Not a photo of a post"})
            where = ask_where(f"{handle}_{code}_{n}.jpg")
            if not where:
                return self.reply(200, {"cancelled": True})
            try:
                url, width, height = original(code, n)
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
                r.raise_for_status()
            except (SystemExit, instaloader.exceptions.InstaloaderException, KeyError, IndexError) as e:  # no login, post gone
                return self.reply(502, {"error": f"Instagram didn't give the full-size photo: {e}"})
            except requests.RequestException as e:
                return self.reply(502, {"error": f"The download failed: {instatagger.failure(str(e))}"})
            path = Path(where)
            if "webp" in r.headers.get("Content-Type", ""):  # Instagram sends JPEG; keep the name honest if it ever doesn't
                path = path.with_suffix(".webp")
            try:
                path.write_bytes(r.content)
            except OSError as e:
                return self.reply(500, {"error": f"Couldn't save it there: {e}"})
            remember(download_dir=str(path.parent))
            return self.reply(200, {"saved": str(path), "width": width, "height": height})
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
