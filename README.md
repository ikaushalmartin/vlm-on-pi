# VLM on Pi

Runs the LFM2.5-VL-450M vision-language model on a Raspberry Pi with llama.cpp,
CPU only. Name the items to watch for in plain text; it checks the camera every
few minutes and shows the live view, what it found, and where.

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
python3 -m unittest discover -s tests
```
