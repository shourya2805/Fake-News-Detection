import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from yt_dlp import YoutubeDL

URL_FILES = [Path("data/youtube_urls.txt"), Path("data/twitter_urls.txt")]
DATA_JSON = Path("data/data.json")
OUT_DIR = Path("videos")
FAIL_LOG = OUT_DIR / "failed.csv"
COVERAGE = OUT_DIR / "coverage.json"
CATEGORY = {"": "REAL", "ft": "CD", "fv": "CE", "fa": "SV", "fc": "CA"}


def video_id(url):
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url) or re.search(r"status/(\d+)", url)
    return match.group(1)


def load_urls():
    urls = []
    for path in URL_FILES:
        urls += [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return urls


def options(vid, browser):
    opts = {
        "outtmpl": str(OUT_DIR / f"{vid}.%(ext)s"),
        "format": "bv*[height<=480]+ba/b[height<=480]/b",
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "noplaylist": True,
        "playlist_items": "1",
        "retries": 3,
        "sleep_interval": 1,
        "max_sleep_interval": 3,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def download(urls, browser):
    failures = []
    for i, url in enumerate(urls, 1):
        vid = video_id(url)
        if (OUT_DIR / f"{vid}.mp4").exists():
            continue
        try:
            with YoutubeDL(options(vid, browser)) as ydl:
                ydl.download([url])
            print(f"[{i}/{len(urls)}] ok    {vid}")
        except Exception as e:
            reason = str(e).splitlines()[0][:200]
            failures.append((vid, url, reason))
            print(f"[{i}/{len(urls)}] FAIL  {vid}  {reason}")

    with FAIL_LOG.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "url", "reason"])
        writer.writerows(failures)
    return failures


def report():
    records = json.loads(DATA_JSON.read_text())
    have, total = Counter(), Counter()
    for r in records:
        keys = [CATEGORY[r["class"]], "YouTube" if len(r["video_id"]) == 11 else "X", "ALL"]
        present = (OUT_DIR / f"{r['video_id']}.mp4").exists()
        for k in keys:
            total[k] += 1
            have[k] += present

    rows = {k: {"have": have[k], "total": total[k], "pct": round(100 * have[k] / total[k], 1)}
            for k in ["REAL", "CD", "CE", "SV", "CA", "YouTube", "X", "ALL"]}
    COVERAGE.write_text(json.dumps(rows, indent=2))

    print(f"\n{'group':<8}{'have':>7}{'total':>7}{'pct':>8}")
    for k, v in rows.items():
        print(f"{k:<8}{v['have']:>7}{v['total']:>7}{v['pct']:>7}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookies-from-browser", dest="browser")
    parser.add_argument("--source", choices=["youtube", "x", "all"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    urls = load_urls()
    if args.source == "youtube":
        urls = [u for u in urls if "youtube.com" in u]
    elif args.source == "x":
        urls = [u for u in urls if "x.com" in u]
    if args.limit:
        urls = urls[: args.limit]

    failures = download(urls, args.browser)
    print(f"\n{len(failures)} failed this run -> {FAIL_LOG}")
    report()


if __name__ == "__main__":
    main()
