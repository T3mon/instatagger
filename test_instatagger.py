"""Run: python test_instatagger.py"""
import base64
import contextlib
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
        "HTTP Error 404: Not Found", "<urlopen error [Errno 11001] getaddrinfo failed>", "403 Client Error: Forbidden for url: x",
        "HTTPSConnectionPool(host='x', port=443): Read timed out. (read timeout=30)"]] == [
    "timed out", "dropped by the server", "HTTP 404", "no internet", "HTTP 403", "timed out"]
# the smallest copy Instagram sent that still covers the tagger's 448 px input, never a square crop of a portrait photo
sizes = {"image_versions2": {"candidates": [{"width": w, "height": w * 4 // 3, "url": f"u{w}"} for w in [2565, 1080, 640, 481, 320]] +
                                          [{"width": w, "height": w, "url": f"square{w}"} for w in [1080, 480]]}}
assert instatagger.small_url(sizes) == "u481" and instatagger.small_url({}) is None
assert parse_tags("# note\n[Place]\nbeach: beach\npool: pool poolside | 0.5\n") == [
    ("Place", [("beach", ["beach"], THRESHOLD), ("pool", ["pool", "poolside"], 0.5)])]

# find_profile: when Instagram leaves the profile out of her page, she's found by the user id the page still names
# (stubbed, no Instagram calls); a page without an id, or with someone else's, still means she doesn't exist
def gone(context, handle):
    raise instaloader.exceptions.ProfileNotExistsException(f"Profile {handle} does not exist.")


real_from_id = instaloader.Profile.from_id
instaloader.Profile.from_username = gone
instaloader.Profile.from_id = lambda context, pk: types.SimpleNamespace(username={42: "her", 7: "someone_else"}[pk])
ctx = types.SimpleNamespace(is_logged_in=True, get_anonymous_session=contextlib.nullcontext)
for page, found in [('"page_id":"profilePage_42","profile_id":"42"', "her"), ("no id here", None), ('"profile_id":"7"', None)]:
    ctx._get_page = lambda session, path, page=page: page
    try:
        assert instatagger.find_profile(ctx, "Her").username == found
    except instaloader.exceptions.ProfileNotExistsException:
        assert found is None, page
instaloader.Profile.from_id = real_from_id

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
    got = list(instatagger.list_photos("someone", 2))
    assert [(url, it["f"]) for url, it in got] == [("u/A", "A_1.jpg"), ("u/B1", "B_1.jpg")], got
    # a re-run stops once it's back among known posts, but not at known pinned ones (3 at most) above new posts,
    # and only when earlier runs reached as far as this one asks
    post = lambda c: types.SimpleNamespace(shortcode=c, typename="GraphImage", is_video=False, url="u/" + c, likes=1, date_utc=day)
    posts[:] = [post(c) for c in ["P1", "P2", "N", "K1", "K2", "K3", "K4", "OLD"]]
    known = {"P1", "P2", "K1", "K2", "K3", "K4"}
    state = {"listed": 7, "to_end": True}
    assert [it["c"] for _, it in instatagger.list_photos("someone", None, known, state)] == ["P1", "P2", "N", "K1", "K2", "K3", "K4"]
    assert state == {"listed": 7, "to_end": True}
    state = {"listed": 7, "to_end": False}  # earlier runs stopped short of her first post: list it all this time
    assert len(list(instatagger.list_photos("someone", None, known, state))) == 8 and state == {"listed": 8, "to_end": True}

# fetch: listing, downloading and tagging overlap; a failed download is retried; photos tagged before aren't tagged again
tries, batches = {}, []


def fake_list(handle, limit, known, state):
    for c in ["A", "B", "C", "A"]:  # A twice: Instagram can repeat a post across pages
        yield "u/" + c, {"c": c, "n": 1, "l": 1, "d": "2026-01-01", "f": c + "_1.jpg"}


def fake_prepare(job):
    tries[job[1].name] = tries.get(job[1].name, 0) + 1
    return "timed out" if job[1].name == "B_1.jpg" and tries["B_1.jpg"] == 1 else True  # B fails once, then works


class FakeTagger:
    def score(self, thumbs):
        batches.append([t.name for t in thumbs])
        return np.zeros((len(thumbs), 3), np.float16), np.ones((len(thumbs), 2), np.float16)


real = instatagger.list_photos, instatagger.prepare, instatagger.time
instatagger.list_photos, instatagger.prepare, instatagger.time = fake_list, fake_prepare, types.SimpleNamespace(sleep=lambda s: None)
with tempfile.TemporaryDirectory() as d:
    scores = {}
    got = instatagger.fetch("x", None, Path(d), set(), {"C_1.jpg"}, {}, FakeTagger(), {"A_1.jpg": 0}, scores)
assert [it["f"] for it in got] == ["A_1.jpg", "B_1.jpg"] and tries == {"A_1.jpg": 1, "B_1.jpg": 2}, (got, tries)
assert batches == [["B_1.jpg"]] and list(scores) == ["B_1.jpg"], batches
instatagger.list_photos, instatagger.prepare, instatagger.time = real

# a re-run keeps the saved scores of earlier photos, tags only the new one, and saves how far the listing got
col = instatagger.booru_columns()
girl = np.zeros(len(col), np.float16)
girl[col["1girl"]] = girl[col["solo"]] = 0.99


def fake_fetch(handle, limit, out, known, never, state, tagger, tagged, scores):
    assert known == {"O"} and state == {"listed": 50, "to_end": True} and set(tagged) == {"O_1.jpg"}, (known, state, tagged)
    state["listed"] = 60
    scores["N_1.jpg"] = girl, np.full(2, 3, np.float16)
    return [{"c": "N", "n": 1, "l": 9, "d": "2026-01-01", "f": "N_1.jpg"}]


real = instatagger.fetch, instatagger.Tagger, instatagger.build_pages, sys.argv
instatagger.fetch, instatagger.Tagger, instatagger.build_pages, sys.argv = fake_fetch, FakeTagger, lambda: None, ["instatagger.py", "x"]
with tempfile.TemporaryDirectory() as d:
    instatagger.HERE = Path(d)
    out = Path(d) / "out" / "x"
    (out / "thumbs").mkdir(parents=True)
    (out / "thumbs" / "O_1.jpg").write_bytes(b"x")
    (Path(d) / "tags.txt").write_text("[Body]\nass: ass\n")
    (out / "items.json").write_text(json.dumps([{"c": "O", "n": 1, "l": 5, "d": "2025-01-01", "f": "O_1.jpg"}]))
    np.save(out / "wd.npy", girl[None])
    np.save(out / "feat.npy", np.full((1, 2), 7, np.float16))
    (out / "data.json").write_text(json.dumps({"handle": "x", "items": [], "skipped": 0, "groups": [], "listed": 50, "to_end": True}))
    instatagger.main()
    data = json.loads((out / "data.json").read_text())
    assert [it["f"] for it in data["items"]] == ["N_1.jpg", "O_1.jpg"] and (data["listed"], data["to_end"]) == (60, True), data
    assert np.load(out / "feat.npy")[:, 0].tolist() == [3, 7] and batches == [["B_1.jpg"]]  # nothing tagged again
instatagger.fetch, instatagger.Tagger, instatagger.build_pages, sys.argv = real

# no_girl: nobody, a group without a girl, or a cartoon without a girl is skipped; a lone "man" in a photo is kept
# (the tagger misreads women in suits), and so is a girl in a dark or edited photo that scores as not realistic
col = instatagger.booru_columns()
wd = np.zeros((6, len(col)), np.float16)
wd[:, col["realistic"]] = 0.8
wd[0, col["no_humans"]] = 0.9                                    # food, scenery
wd[1, col["1boy"]], wd[1, col["multiple_boys"]] = 0.9, 0.9      # two men
wd[2, col["1boy"]], wd[2, col["solo"]] = 0.95, 0.95             # her in a suit, read as a man
wd[3, col["1girl"]], wd[3, col["solo"]] = 0.99, 0.97             # her
wd[4, col["1boy"]], wd[4, col["solo"]], wd[4, col["realistic"]] = 0.95, 0.95, 0.01  # a cartoon meme
wd[5, col["1girl"]], wd[5, col["solo"]], wd[5, col["realistic"]] = 0.9, 0.9, 0.01   # her at night, in neon
assert instatagger.no_girl(wd, instatagger.SKIPS).tolist() == [True, True, False, False, True, False]
less = {**instatagger.SKIPS, "cartoon": {**instatagger.SKIPS["cartoon"], "on": False}, "nobody": {"on": True, "no_humans": 0.95}}
assert instatagger.no_girl(wd, less).tolist() == [True, True, False, False, False, False]  # the food photo has no girl either

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
    np.save(Path(d) / "out" / "a" / "feat.npy", np.array([[3, 0, -4, 0]], np.float16))  # image features, stored at unit length x 127
    (Path(d) / "out" / "a" / "items.json").write_text(json.dumps([{"f": "X_1.jpg"}]))
    instatagger.build_pages()
    page = json.loads((Path(d) / "out" / "index.html").read_text().removeprefix("const D = ").removesuffix(";"))
assert [g for g, _ in page["groups"]] == ["Rating", "Body"], page["groups"]  # models aren't a tag group: the page has a bar for them
assert [it["h"] for it in page["items"]] == ["a", "b"] and page["items"][0]["t"] == ["ass", "questionable"]
assert abs(page["items"][0]["x"] - 1.9) < 0.01 and "x" not in page["items"][1]
assert [(page["vocab"][j], pct) for j, pct in page["items"][0]["b"]] == [("1girl", 99), ("beach", 80)]  # no ratings, strongest first
assert list(np.frombuffer(base64.b64decode(page["items"][0]["e"]), np.int8)) == [76, 0, -102, 0] and "e" not in page["items"][1]

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
page.request("GET", "/settings")
assert b"Skipped photos" in page.getresponse().read()
page.request("GET", "/favicon.ico")
assert page.getresponse().read()[:4] == b"\0\0\1\0"  # an .ico file
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
    broken = Path(d) / "out" / "zz"
    broken.mkdir()
    (broken / "data.json").write_text("{")  # half written, e.g. by a run finishing right now
    assert [m["handle"] for m in call("GET", "/api/models")[1]["models"]] == ["a"]  # the page still loads, without her
    (broken / "data.json").unlink()
    broken.rmdir()
    assert call("POST", "/api/delete", {"u": "nobody"})[0] == 404
    # Settings: the sessionid is checked with Instagram (stubbed) before it's saved
    serve.check_login = lambda session=None: "me" if session and session.startswith("12345%3A") else None
    assert call("POST", "/api/settings", {"session": "short"})[0] == 400
    assert call("POST", "/api/settings", {"session": "99999%3Aabcdefghijklmnopqrstuvwxyz"})[0] == 400 and not (Path(d) / "session.txt").exists()
    assert call("POST", "/api/settings", {"session": "sessionid=12345:abcdefghijklmnopqrstuvwxyz;"}) == (200, {"user": "me"})
    assert (Path(d) / "session.txt").read_text() == "12345%3Aabcdefghijklmnopqrstuvwxyz"
    # deleting photos: preview, entry and score row go together, and they're remembered so a re-run skips them
    (folder / "items.json").write_text(json.dumps([{k: v for k, v in it.items() if k != "t"} for it in items]))
    wd = np.zeros((30, len(instatagger.booru_columns())), np.float16)
    wd[:, 0] = np.arange(30)
    np.save(folder / "wd.npy", wd)
    np.save(folder / "feat.npy", np.arange(30, dtype=np.float16)[:, None] * np.ones((1, 8), np.float16))
    assert call("POST", "/api/delete", {"u": "a", "files": ["P0_1.jpg", "nope.jpg"]}) == (200, {"deleted": 1})
    assert call("POST", "/api/delete", {"u": "a", "years": "2026"})[0] == 400
    with serve.files_lock:  # waits for the page rebuild
        assert not (folder / "thumbs" / "P0_1.jpg").exists() and json.loads((folder / "removed.json").read_text()) == ["P0_1.jpg"]
        assert np.load(folder / "wd.npy")[:, 0].tolist() == list(range(1, 30)) and len(json.loads((folder / "items.json").read_text())) == 29
        assert np.load(folder / "feat.npy")[:, 0].tolist() == list(range(1, 30))  # features shrink with them
        assert len(json.loads((folder / "data.json").read_text())["items"]) == 29
    # skip rules: the preview counts what saving would delete; saving checks them, writes them and deletes those photos as skipped
    wd = np.load(folder / "wd.npy")
    wd[:, col["1girl"]], wd[:, col["solo"]], wd[:, col["realistic"]] = 0.9, 0.9, 0.8
    wd[:3, col["no_humans"]] = 0.6  # three photos of nobody
    np.save(folder / "wd.npy", wd)
    off = {k: {**r, "on": False} for k, r in instatagger.SKIPS.items()}
    assert call("POST", "/api/skip/preview", {"rules": off})[1] == {"models": [{"handle": "a", "photos": 29, "drop": 0, "sample": []}]}
    assert call("POST", "/api/skip/preview", {"rules": instatagger.SKIPS})[1]["models"][0]["drop"] == 3
    assert call("POST", "/api/skip/preview", {"rules": {**off, "nobody": {"on": True, "no_humans": 0.7}}})[1]["models"][0]["drop"] == 0
    assert call("POST", "/api/skip/preview", {"rules": {**off, "nobody": {"on": True, "no_humans": 2}}})[0] == 400  # cut-offs are 0-1
    assert call("POST", "/api/skip/preview", {"rules": {**off, "nobody": {"on": "yes", "no_humans": 0.5}}})[0] == 400
    assert call("POST", "/api/skip/preview", {"rules": off, "custom": "boys: 1boyz"})[0] == 400  # not a tagger tag
    assert call("POST", "/api/skip/preview", {"rules": off, "custom": "boys 1boy"})[0] == 400
    assert call("POST", "/api/skip/preview", {"rules": off, "custom": "her: 1girl | 0.95"})[1]["models"][0]["drop"] == 0
    code, got = call("POST", "/api/skip", {"rules": {**off, "nobody": {"on": True, "no_humans": 0.5}}, "custom": "boys: 1boy multiple_boys | 0.5\n\n"})
    assert code == 200 and got["deleted"] == 3 and got["skip"] == ["boys: 1boy multiple_boys | 0.5"], got
    assert got["rules"]["nobody"] == {"on": True, "no_humans": 0.5} and got["groups"] == [["Body", 1]], got
    assert instatagger.skip_rules() == got["rules"] and not instatagger.skip_rules()["cartoon"]["on"]  # read back from settings.json
    assert instatagger.parse_tags((Path(d) / "tags.txt").read_text())[1] == ("Skip", [("boys", ["1boy", "multiple_boys"], 0.5)])
    assert len(json.loads((folder / "skipped.json").read_text())) == 3 and json.loads((folder / "data.json").read_text())["skipped"] == 3
    assert len(np.load(folder / "wd.npy")) == len(json.loads((folder / "items.json").read_text())) == 26
    assert call("POST", "/api/skip", {"rules": off, "custom": ""})[1]["skip"] == []
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
