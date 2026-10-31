"""Time one check: how long the model takes to find named items in one image.

Runs on the Pi's stock Python (3.7+, standard library only) against the
llama.cpp server. Example:

    python3 tools/bench.py --image sample.jpg --items "cat, remote control" --runs 3
"""
import argparse
import base64
import json
import time
import urllib.request

# The grounding prompt format this model expects.
PROMPT = ('Detect all instances of: {items}. Response must be a JSON array: '
          '[{{"label": ..., "bbox": [x1, y1, x2, y2]}}, ...]. '
          'Coordinates are normalized to [0,1].')


def check(server, image_path, items, max_tokens):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
            {"type": "text", "text": PROMPT.format(items=items)},
        ]}],
        # Recommended sampling settings for this model.
        "temperature": 0.1,
        "min_p": 0.15,
        "repeat_penalty": 1.05,
        "max_tokens": max_tokens,
        # Every real check sees a new photo, so never reuse the cached image.
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        server + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    return time.time() - started, result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--image", required=True)
    parser.add_argument("--items", required=True, help='comma-separated, e.g. "keys, wallet"')
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    totals = []
    for run in range(1, args.runs + 1):
        seconds, result = check(args.server, args.image, args.items, args.max_tokens)
        totals.append(seconds)
        t = result.get("timings", {})
        print("run %d: %.1f s total | prompt %s tokens in %.1f s | output %s tokens in %.1f s (%.1f tok/s)" % (
            run, seconds,
            t.get("prompt_n"), t.get("prompt_ms", 0) / 1000.0,
            t.get("predicted_n"), t.get("predicted_ms", 0) / 1000.0,
            t.get("predicted_per_second", 0)))
        print("  answer:", result["choices"][0]["message"]["content"].strip())
    print("average: %.1f s per check" % (sum(totals) / len(totals)))


if __name__ == "__main__":
    main()
