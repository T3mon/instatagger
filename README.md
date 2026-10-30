# instatagger

Tags every photo on an Instagram profile and builds a local page to search them by tag: `ass` + `beach`, exclude `lingerie`, sort by likes. Each result links to the exact post and slide.

- Lists posts with your own Instagram login through [Instaloader](https://instaloader.github.io/), one page of 12 every 6–12 s.
- Body, pose and framing tags come from the [WD EVA02 booru tagger](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3), outfit and place tags from [LAION CLIP ViT-H/14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K). Both run locally; a GPU is recommended.
- Photos with no person in them (text slides, food, packaging) are skipped.
- Full-size photos are never saved. `out/<handle>/` keeps small previews, cached scores and `index.html`.

## Setup

    pip install -r requirements.txt

For a GPU, install torch with CUDA from pytorch.org first. Put your Instagram `sessionid` cookie value (browser DevTools → Application → Cookies) in `session.txt`. Keep it private: anyone with it is logged in as you.

## Use

    python instatagger.py <handle or profile link> [N]   # newest N posts, default all
    python instatagger.py <handle> --retag               # after editing tags.txt, no Instagram calls

Open `out/<handle>/index.html` in a browser. Edit `tags.txt` to add or tune tags. `python test_instatagger.py` runs the tests.

Scraping breaks Instagram's terms and can get the account rate-limited. Use it for private research and don't repost the photos.
