# instatagger

Tags every photo on an Instagram profile and builds a local page to search them by tag: `ass` + `beach`, exclude `lingerie`, sort by likes. Each result links to the exact post and slide.

- Lists posts with your own Instagram login through [Instaloader](https://instaloader.github.io/), one page of 12 every 6–12 s.
- Tags come from the [WD EVA02 booru tagger](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3), which runs locally; a GPU is recommended. `tags.txt` maps your tag names to its tags.
- Photos without a girl in them (scenery, food, text slides, crowds, groups of men) are skipped before tagging: their previews are deleted and they're never downloaded again.
- Full-size photos are never saved. `out/<handle>/` keeps small previews, cached scores and `index.html`.

## Setup

    pip install -r requirements.txt

For a GPU, install torch with CUDA from pytorch.org first. Put your Instagram `sessionid` cookie value (browser DevTools → Application → Cookies) in `session.txt`. Keep it private: anyone with it is logged in as you.

## Use

    python instatagger.py <handle or profile link> [N]   # newest N posts, default all
    python instatagger.py <handle> --retag               # after editing tags.txt, no Instagram calls

Open `out/<handle>/index.html` in a browser, or `out/index.html` for every model on one page (filter by model, sort by likes or date). **More like this** on a photo sorts the grid by the tags other posts share with it (outfit, pose, setting), with your filters still on.

Or run `python serve.py` and open http://localhost:8765. The pages then get an **Add model** button: paste a profile link to see the account's stats and a time estimate, optionally limit it to the newest N posts, and follow the run's progress. The **Models** page shows each model's stats (likes, rating mix, top tags, the tags that get her the most likes, disk use) and settings. It deletes a whole model or her photos from a range of years; the photo panel in the gallery deletes single photos. Deleted photos are never downloaded again. Edit `tags.txt` to add or tune tags. `python test_instatagger.py` runs the tests.

Scraping breaks Instagram's terms and can get the account rate-limited. Use it for private research and don't repost the photos.
