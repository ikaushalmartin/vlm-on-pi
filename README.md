# VLM on Pi

Runs the LFM2.5-VL-450M vision-language model on a Raspberry Pi with llama.cpp,
CPU only. Name the items to watch for in plain text; it checks the camera every
few minutes and shows the live view, what it found, and where.

## Results on a Raspberry Pi 4 (2 GB)

Raspberry Pi 4 Model B, 2 GB, 4 cores at 1.5 GHz, CPU only. Model in 4-bit
(Q4_0) with its vision encoder (Q8_0), served by llama.cpp in Docker. Frames
are 512 px on the long side.

### Speed and resources

| | |
|---|---|
| Model files | 309 MB (210 MB language model + 99 MB vision encoder) |
| RAM, whole system | ~500 MB of 1.8 GB (OS, model, app, live video), no swap |
| Reading the frame + prompt | 14–17 tokens/s |
| Writing the answer | 10–13 tokens/s |
| Tokens per question | ~200 |
| Time per question | 19–23 s |
| Time per look (what's there + where) | 36–54 s |
| Zoom-in pass for tighter boxes | +12–15 s per item found |
| CPU during a look | llama.cpp 358% (all 4 cores), live video 15–23% of one core, web app ~2.5% |
| CPU temperature under load | 64–71 °C |
| Power | needs a 5.1 V / 3 A supply to hold 1.5 GHz |

### Accuracy on real CCTV

Each frame was labelled by hand (items present and absent), then
`tools/video_test.py` scored every yes/no answer against the labels.

| Footage | Answers right | False alarms |
|---|---|---|
| School lobby camera (5 frames, 10 items) | 40 / 50 | 0 of 20 |
| Bus station camera (3 frames, 10 items) | 23 / 30 | 0 of 15 |
| Supermarket clip (4 frames, 8 items) | 28 / 32 | 0 of 12 |
| **Total** | **91 / 112 (81%)** | **0 of 47** |

Specificity 100% (47/47), recall 68% (44/65). Every error was a miss, mostly
small or distant objects such as a couch at the back of a room, trash cans
and an ATM.

### What I learned

- **Ask what's there before asking where.** Asked only to draw boxes, the
  model marks every item it's asked about, even absent ones; it put a cat and
  a remote on a person. Asking "which of these are in the image?" first, then
  boxes only for those, removed the false alarms.
- **Reading the image is most of the cost.** A 256 px frame took about half
  as long as 512 px. Lowering llama.cpp's image-token limit made it almost 6×
  slower instead: the frame got split into tiles (1,725 tokens instead of 299).
- **Boxes on wide frames are coarse.** A 16:9 frame at 512×286 is only about
  144 image tokens. Re-asking on a crop around each box doubled the overlap
  with the true position on a test frame (IoU 0.15 → 0.31).
- **Benchmark with fresh frames.** Repeating the same frame looked 6× faster
  because llama.cpp reused the cached image. A cached image isn't reused
  across different questions for this model, so every question re-reads the
  frame.
- **Power is part of the design.** On a weak supply the Pi reported
  under-voltage, dropped to 600 MHz and ran 2.3–2.5× slower (6.6 tokens/s
  reading, 77–92 s per look).

### Where it fits

- Periodic checks in stores and facilities: is the wet floor sign out, are
  boxes blocking an aisle?
- Finding equipment by name in warehouses and workshops, with no detector to
  train per item.
- Privacy-first cameras: frames never leave the device.
- Offline sites with no internet or cloud budget.

It doesn't fit real-time work such as tracking: it takes seconds per frame,
not frames per second.

### Compared with other small vision models on a Pi

From other people's published tests. Boards, image sizes and runtimes differ,
and a Pi 5 is much faster than a Pi 4, so treat this as a rough guide.

| Model | Board | Reported speed | Memory |
|---|---|---|---|
| **LFM2.5-VL-450M (this repo)** | **Pi 4, 2 GB** | **19–23 s per question** | **~0.5 GB** |
| Moondream 0.5B | Pi 4, 4 GB | 37 s image + 56 s answer | 1.9 GB |
| Moondream 2B | Pi 5 | 18.6 s image + 1.3 s answer | ~7 GB with desktop |
| Qwen3-VL-2B | Pi 5, 16 GB | 4.8 tokens/s writing (this repo: 10–13) | – |
| Qwen2.5-VL 3B | Pi 4, 8 GB | 302 s for one OCR task | needs more than 4 GB |

## Run

Needs a Raspberry Pi 4 or 5 (64-bit OS), Docker and a camera.

```sh
git clone https://github.com/ikaushalmartin/vlm-on-pi && cd vlm-on-pi
mkdir models && cd models
B=https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main
curl -LO $B/LFM2.5-VL-450M-Q4_0.gguf
curl -LO $B/mmproj-LFM2.5-VL-450m-Q8_0.gguf
cd ..
cp .env.example .env    # set CAMERA and other options
docker compose up -d --build
```

Then open `http://<pi-address>:8000`.

## Test

```sh
python3 -m unittest discover -s tests          # logic tests, no model needed
python3 tools/bench.py --image photo.jpg --items "cat, remote control"
```

`tools/video_test.py` runs the full pipeline on video frames or images and
scores the answers against hand labels.
