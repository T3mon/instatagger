"""Run: python test_instatagger.py"""
import json
import tempfile
import types
from datetime import datetime
from pathlib import Path

import instaloader

import instatagger
from instatagger import THRESHOLD, handle_from, parse_tags

assert handle_from("https://www.instagram.com/brennahblack/?hl=en") == "brennahblack"
assert handle_from("@meganblakeirwin") == "meganblakeirwin"
try:
    handle_from("../secrets")
    raise AssertionError("path-like handle accepted")
except SystemExit:
    pass
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

# build_pages: one all-models page from each profile's data.json, with a Model row and a stats line per profile + total
with tempfile.TemporaryDirectory() as d:
    instatagger.HERE = Path(d)
    (Path(d) / "viewer.html").write_text("const D = /*DATA*/null;")
    for h in ["a", "b"]:
        (Path(d) / "out" / h / "thumbs").mkdir(parents=True)
        (Path(d) / "out" / h / "thumbs" / "X_1.jpg").write_bytes(b"x")
        (Path(d) / "out" / h / "data.json").write_text(json.dumps({"handle": h, "skipped": 1, "groups": [["Body", ["ass"]]],
            "items": [{"c": "X", "n": 1, "l": 5, "d": "2026-01-01", "f": "X_1.jpg", "t": ["ass"]}]}))
    instatagger.build_pages()
    page = json.loads((Path(d) / "out" / "index.html").read_text().removeprefix("const D = ").removesuffix(";"))
assert page["groups"] == [["Model", ["@a", "@b"]], ["Body", ["ass"]]], page["groups"]
assert [it["h"] for it in page["items"]] == ["a", "b"] and page["items"][0]["t"] == ["ass", "@a"]
assert [(r["model"], r["posts"], r["skipped"]) for r in page["stats"]] == [("@a", 1, 1), ("@b", 1, 1), ("All", 2, 2)]
print("ok")
