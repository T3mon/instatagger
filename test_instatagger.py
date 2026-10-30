"""Run: python test_instatagger.py"""
import http.client
import json
import sys
import tempfile
import threading
import time
import types
from datetime import datetime
from pathlib import Path

import instaloader
import numpy as np

import instatagger
from instatagger import THRESHOLD, handle_from, parse_tags

assert handle_from("https://www.instagram.com/brennahblack/?hl=en") == "brennahblack"
assert handle_from("@meganblakeirwin") == "meganblakeirwin"
try:
    handle_from("../secrets")
    raise AssertionError("path-like handle accepted")
except SystemExit:
    pass
assert [instatagger.failure(e) for e in ["<urlopen error [WinError 10060] A connection attempt failed>", "[WinError 10054] An existing connection was forcibly closed",
        "HTTP Error 404: Not Found", "<urlopen error [Errno 11001] getaddrinfo failed>"]] == ["timed out", "dropped by the server", "HTTP 404", "no internet"]
assert parse_tags("# note\n[Place]\nbeach: beach\npool: pool poolside | 0.5\n") == [
    ("Place", [("beach", ["beach"], THRESHOLD), ("pool", ["pool", "poolside"], 0.5)])]

# list_photos on a fake profile (no Instagram calls): newest-N limit, one entry per carousel slide, videos skipped
day = datetime(2026, 10, 1)
posts = [
    types.SimpleNamespace(shortcode="A", typename="GraphImage", is_video=False, url="u/A", likes=5, date_utc=day),
    types.SimpleNamespace(shortcode="B", typename="GraphSidecar", likes=7, date_utc=day, get_sidecar_nodes=lambda: iter(
        [instaloader.PostSidecarNode(False, "u/B1", None), instaloader.PostSidecarNode(True, "u/B2", "v/B2")])),
    types.SimpleNamespace(shortcode="C", typename="GraphImage", is_video=False, url="u/C", likes=1, date_utc=day),
]
instaloader.Profile.from_username = lambda context, handle: types.SimpleNamespace(mediacount=3, get_posts=lambda: iter(posts))
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "session.txt").write_text("1%3Afake%3A2%3Asig")
    instatagger.HERE = Path(d)
    got = instatagger.list_photos("someone", 2)
assert [(url, it["f"]) for url, it in got] == [("u/A", "A_1.jpg"), ("u/B1", "B_1.jpg")], got

# no_girl: nobody, or a group without a girl, is skipped; a lone "man" is kept (the tagger misreads women in suits)
col = instatagger.booru_columns()
wd = np.zeros((4, len(col)), np.float16)
wd[0, col["no_humans"]] = 0.9                                    # food, scenery
wd[1, col["1boy"]], wd[1, col["multiple_boys"]] = 0.9, 0.9      # two men
wd[2, col["1boy"]], wd[2, col["solo"]] = 0.95, 0.95             # her in a suit, read as a man
wd[3, col["1girl"]], wd[3, col["solo"]] = 0.99, 0.97             # her
assert instatagger.no_girl(wd).tolist() == [True, True, False, False]

# build_pages: one all-models page from each profile's data.json, with a Model row
with tempfile.TemporaryDirectory() as d:
    instatagger.HERE = Path(d)
    (Path(d) / "viewer.html").write_text("const D = /*DATA*/null;")
    for h in ["a", "b"]:
        (Path(d) / "out" / h / "thumbs").mkdir(parents=True)
        (Path(d) / "out" / h / "thumbs" / "X_1.jpg").write_bytes(b"x")
        (Path(d) / "out" / h / "data.json").write_text(json.dumps({"handle": h, "skipped": 1, "groups": [["Body", ["ass"]]],
            "items": [{"c": "X", "n": 1, "l": 5, "d": "2026-01-01", "f": "X_1.jpg", "t": ["ass"]}]}))
    col = instatagger.booru_columns()  # profile a has saved scores, so it gets a rating; b doesn't
    wd = np.zeros((1, len(col)), np.float16)
    wd[0, col["sensitive"]], wd[0, col["questionable"]] = 0.1, 0.9
    wd[0, col["beach"]], wd[0, col["1girl"]], wd[0, col["pool"]] = 0.8, 0.99, 0.2  # pool is under the cutoff
    np.save(Path(d) / "out" / "a" / "wd.npy", wd)
    (Path(d) / "out" / "a" / "items.json").write_text(json.dumps([{"f": "X_1.jpg"}]))
    instatagger.build_pages()
    page = json.loads((Path(d) / "out" / "index.html").read_text().removeprefix("const D = ").removesuffix(";"))
assert [g for g, _ in page["groups"]] == ["Model", "Rating", "Body"], page["groups"]
assert page["groups"][0] == ["Model", ["@a", "@b"]]
assert [it["h"] for it in page["items"]] == ["a", "b"] and page["items"][0]["t"] == ["ass", "questionable", "@a"]
assert abs(page["items"][0]["x"] - 1.9) < 0.01 and "x" not in page["items"][1]
assert [(page["vocab"][j], pct) for j, pct in page["items"][0]["b"]] == [("1girl", 99), ("beach", 80)]  # no ratings, strongest first

# serve.py: API needs the X-Instatagger header, checks input, and reports a run's output (a fake run, no Instagram)
import serve  # noqa: E402

srv = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method, path, body=None, header=True):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port)
    c.request(method, path, body=json.dumps(body) if body else None, headers={"X-Instatagger": "1"} if header else {})
    r = c.getresponse()
    return r.status, json.loads(r.read())


assert call("GET", "/api/status", header=False)[0] == 403
page = http.client.HTTPConnection("127.0.0.1", srv.server_port)
page.request("GET", "/add")
assert b"Add model" in page.getresponse().read()
page.request("GET", "/models")
assert b"Your models" in page.getresponse().read()
assert call("POST", "/api/add", {"u": "../x"})[0] == 400
assert call("POST", "/api/add", {"u": "someone", "limit": 0})[0] == 400
serve.start("someone", [sys.executable, "-c", "print('Listing posts'); print('skipped x.jpg: timeout'); print('3 photos -> done')"])
serve.run["proc"].wait()
for _ in range(50):
    if serve.run["ended"]:
        break
    time.sleep(0.05)
code, status = call("GET", "/api/status")
assert code == 200 and status["handle"] == "someone" and not status["running"] and status["code"] == 0, status
assert [line for _, line in status["log"]] == ["Listing posts", "3 photos -> done"], status  # per-photo errors are left out

# Models page data and delete, on a fake library: 10 photos tagged ass at 300 likes, 19 others at 100, 1 with likes hidden
with tempfile.TemporaryDirectory() as d:
    serve.HERE = instatagger.HERE = Path(d)
    (Path(d) / "viewer.html").write_text("const D = /*DATA*/null;")
    (Path(d) / "tags.txt").write_text("[Body]\nass: ass\n")
    folder = Path(d) / "out" / "a"
    (folder / "thumbs").mkdir(parents=True)
    (folder / "thumbs" / "P0_1.jpg").write_bytes(b"x")
    items = [{"c": f"P{k}", "n": 1, "l": 3 if k == 0 else 300 if k <= 10 else 100, "d": f"2026-01-{k + 1:02}", "f": f"P{k}_1.jpg",
              "t": ["ass"] if 0 < k <= 10 else []} for k in range(30)]
    (folder / "data.json").write_text(json.dumps({"handle": "a", "skipped": 2, "groups": [["Body", ["ass"]]], "items": items}))
    code, got = call("GET", "/api/models")
    m = got["models"][0]
    assert code == 200 and (m["handle"], m["photos"], m["posts"], m["skipped"], m["hidden"], m["likes"], m["likes_max"]) == ("a", 30, 30, 2, 1, 100, 300), m
    assert (m["top"], m["lift"], m["rating"], m["per_week"]) == ([["ass", 10]], [["ass", 3.0]], None, 7.2), m
    assert [it["l"] for it in m["best"]] == [300] * 4 and m["files"] == 2, m
    assert got["settings"]["groups"] == [["Body", 1]] and got["settings"]["session"] is None, got["settings"]
    assert call("POST", "/api/delete", {"u": "nobody"})[0] == 404
    # deleting photos: preview, entry and score row go together, and they're remembered so a re-run skips them
    (folder / "items.json").write_text(json.dumps([{k: v for k, v in it.items() if k != "t"} for it in items]))
    wd = np.zeros((30, len(instatagger.booru_columns())), np.float16)
    wd[:, 0] = np.arange(30)
    np.save(folder / "wd.npy", wd)
    assert call("POST", "/api/delete", {"u": "a", "files": ["P0_1.jpg", "nope.jpg"]}) == (200, {"deleted": 1})
    assert call("POST", "/api/delete", {"u": "a", "years": "2026"})[0] == 400
    with serve.files_lock:  # waits for the page rebuild
        assert not (folder / "thumbs" / "P0_1.jpg").exists() and json.loads((folder / "removed.json").read_text()) == ["P0_1.jpg"]
        assert np.load(folder / "wd.npy")[:, 0].tolist() == list(range(1, 30)) and len(json.loads((folder / "items.json").read_text())) == 29
        assert len(json.loads((folder / "data.json").read_text())["items"]) == 29
    serve.start("b", [sys.executable, "-c", "import time; time.sleep(30)"])
    assert call("POST", "/api/delete", {"u": "a"})[0] == 409 and folder.exists()  # not while a run is going
    assert call("POST", "/api/delete", {"u": "a", "years": [2027, 2030]}) == (200, {"deleted": 0})  # photos of another model are fine
    serve.run["handle"] = "a"
    assert call("POST", "/api/delete", {"u": "a", "years": [2026, 2026]})[0] == 409  # but not hers while she's being added
    serve.run["proc"].kill()
    serve.run["proc"].wait()
    assert call("POST", "/api/delete", {"u": "a"}) == (200, {"deleted": "a"}) and not folder.exists()
    page = json.loads((Path(d) / "out" / "index.html").read_text().removeprefix("const D = ").removesuffix(";"))
    assert page["items"] == [] and call("GET", "/api/models")[1]["models"] == []
srv.shutdown()

# recent_stats: newest posts without pinned ones; photo share, slides per photo post, likes (3 = hidden)
day = lambda d: datetime(2026, 10, d)
newest = [types.SimpleNamespace(is_pinned=True, is_video=False, mediacount=1, likes=9, date_utc=day(1)),
          types.SimpleNamespace(is_pinned=False, is_video=False, mediacount=4, likes=100, date_utc=day(29)),
          types.SimpleNamespace(is_pinned=False, is_video=True, mediacount=1, likes=300, date_utc=day(22)),
          types.SimpleNamespace(is_pinned=False, is_video=False, mediacount=2, likes=200, date_utc=day(15))]
got = serve.recent_stats(newest, 10_000)
assert (got["recent"], got["last_post"], got["photo_posts"], got["photos_per_post"], got["likes"], got["like_rate"]) == \
    (3, "2026-10-29", 2, 3.0, 200, 2.0), got
assert got["per_week"] == 1.5, got  # 3 posts over 14 days
assert serve.recent_stats([types.SimpleNamespace(is_pinned=False, is_video=False, mediacount=1, likes=3, date_utc=day(2))], 10)["likes"] is None
print("ok")
