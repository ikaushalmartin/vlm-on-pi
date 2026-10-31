"""Run the real two-question pipeline over frames of a video (or still frames) and score it.

Runs inside the app container, next to the model:

    docker compose run --rm --no-deps -T app python tools/video_test.py \
        --video /data/supermarket.mp4 --times 0.5 3.5 7 10.5 \
        --items "shopping basket, vegetables, cardboard box, price tag, backpack, bicycle, dog, wet floor sign" \
        --truth "shopping basket, vegetables, cardboard box, price tag, backpack" \
        --out /data/video-test

Use --images a.jpg b.jpg instead of --video/--times for frames already saved.
--truth lists the items that really are in every tested frame; everything
else in --items is really absent. Saves each frame and results.json in --out.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlm_on_pi.detector import Detector  # noqa: E402


def grab(source, size, seconds=None):
    """One frame as a JPEG, longest side `size`: from a video at `seconds`, or from an image."""
    seek = ["-ss", str(seconds)] if seconds is not None else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + seek + ["-i", source,
           "-frames:v", "1", "-vf", "scale=w=%d:h=%d:force_original_aspect_ratio=decrease" % (size, size),
           "-f", "image2", "-vcodec", "mjpeg", "-q:v", "3", "pipe:1"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def split(text):
    return [s.strip() for s in text.split(",") if s.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video")
    p.add_argument("--times", type=float, nargs="+", help="seconds into the video")
    p.add_argument("--images", nargs="+", help="still frames to test instead of a video")
    p.add_argument("--items", required=True)
    p.add_argument("--truth", default="", help="items that really are in the frames")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--llm", default=os.environ.get("LLM_URL", "http://llm:8080"))
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.images:
        frames = [(os.path.basename(path), lambda path=path: grab(path, args.size)) for path in args.images]
    elif args.video and args.times:
        frames = [("frame_%05.1fs.jpg" % t, lambda t=t: grab(args.video, args.size, t)) for t in args.times]
    else:
        p.error("give --images, or --video with --times")

    items, truth = split(args.items), set(split(args.truth))
    os.makedirs(args.out, exist_ok=True)
    detector = Detector(args.llm)
    results, right, asked = [], 0, 0
    for name, load in frames:
        jpeg = load()
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(jpeg)
        found, info = detector.detect(jpeg, items)
        said = set(info["present"])
        mistakes = sorted(i for i in items if (i in said) != (i in truth))
        right += len(items) - len(mistakes)
        asked += len(items)
        results.append({"frame": name, "seconds": info["seconds"], "present": info["present"],
                        "boxes": found, "wrong": mistakes, "raw": info["raw"]})
        print("%-22s %5.1f s  present=%s  boxes=%d  wrong=%s" % (
            name, info["seconds"], info["present"], len(found), mistakes or "none"), flush=True)

    summary = {"frames": len(results), "items": items, "truth": sorted(truth),
               "presence_correct": right, "presence_asked": asked,
               "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 1)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"summary": summary, "frames": results}, f, indent=2)
    print("presence answers correct: %d of %d | average %.1f s per frame" % (
        right, asked, summary["avg_seconds"]))


if __name__ == "__main__":
    main()
