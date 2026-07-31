# Airavata Quantization Service

Serve [AI4Bharat/Airavata](https://huggingface.co/ai4bharat/Airavata) under several
quantization schemes behind one HTTP API, and measure what each scheme actually
costs you in latency, throughput and memory on *your* hardware.

The point is the comparison. Published quantization numbers are measured on
A100s; this gives you the same table for the box you are deploying on.

## What it does

| Variant | Device | Weights | Notes |
| --- | --- | --- | --- |
| `original` | CPU or GPU | FP16 on GPU, FP32 on CPU | Baseline everything is compared against |
| `int8` | GPU only | 8-bit | bitsandbytes `LLM.int8()`, ~50% weight memory |
| `int4` | GPU only | 4-bit | bitsandbytes NF4 + double quantization, ~25% weight memory |
| `dynamic_quant` | CPU only | 8-bit `nn.Linear` | `torch.ao.quantization.quantize_dynamic` |

Variants that the detected device cannot run are reported as unsupported rather
than failing at request time. Anything not preloaded is loaded lazily on first
use.

`dynamic_quant` needs an architecture whose weights live in `nn.Linear` — LLaMA
family models, which includes Airavata. GPT-2 family models keep theirs in
`Conv1D`, which torch's dynamic quantization does not touch; loading the variant
there fails with a `503` naming the reason rather than serving an unquantized
copy under a quantized name.

Beyond generation, the service exposes per-variant benchmarking with warmup,
p50/p95 latency and aggregate throughput; a `/benchmark/all` sweep that returns a
speedup table relative to the FP baseline; live CPU/RAM/GPU telemetry; and an
export endpoint that writes a quantized checkpoint to disk so you do not pay the
quantization cost again.

## Install

Requires Python 3.9+. For GPU quantization you also need a CUDA-capable card
(compute capability 7.5+ for NF4), a matching CUDA runtime, and enough VRAM for
the variant you want (roughly 13 GB FP16, 7 GB int8, 4 GB int4 for a 7B model).

```bash
git clone https://github.com/SdSarthak/Quantization.git
cd Quantization
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate

# GPU: install a CUDA build of torch first
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
cp .env.example .env      # optional; every value has a default
```

CPU-only hosts can skip `accelerate` and `bitsandbytes`; the `int8`/`int4`
variants simply report as unsupported.

## Run

```bash
# HTTP API on http://127.0.0.1:8000 (interactive docs at /docs)
python -m airavata_quant serve

# Equivalent, and still the historical entrypoint
python airavata_quantization_service.py
```

`AIRAVATA_PRELOAD=none python -m airavata_quant serve` starts immediately and
loads weights on the first request instead - useful when you only want
telemetry, or when the FP baseline does not fit alongside a quantized variant.

### CLI

```bash
python -m airavata_quant info                      # device, config, variant support
python -m airavata_quant benchmark --iterations 5  # every supported variant
python -m airavata_quant benchmark int4 --iterations 10 --max-new-tokens 128
python -m airavata_quant export int4               # writes ./quantized_models/int4
python -m airavata_quant serve --port 9000 --no-preload
```

`benchmark` writes a timestamped JSON report to `AIRAVATA_BENCHMARK_DIR`
(`./benchmarks` by default) as well as printing it.

### Docker

```bash
docker compose up airavata-gpu    # CUDA image, port 8000
docker compose up airavata-cpu    # slim CPU image, port 8001
```

Both mount a shared `model-cache` volume so the checkpoint is downloaded once.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service, device, loaded and available variants |
| `GET` | `/health` | Liveness plus tokenizer/model readiness |
| `GET` | `/models` | Every variant with device support and load state |
| `POST` | `/generate/{variant}` | Generate text |
| `GET` | `/benchmark/{variant}` | Benchmark one variant |
| `GET` | `/benchmark/all` | Benchmark every supported variant and compare |
| `POST` | `/models/{variant}/save` | Export a loaded variant to disk |
| `DELETE` | `/models/{variant}` | Unload a variant and free its memory |
| `GET` | `/system/info` | CPU, RAM and GPU telemetry |

### Generate

```bash
curl -X POST http://localhost:8000/generate/int4 \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI in India", "max_length": 100, "temperature": 0.7, "top_p": 0.9}'
```

`max_length` is the number of **new** tokens (1-2048). `temperature` is 0-2,
where `0` selects greedy decoding and skips sampling entirely. Pass `seed` for
reproducible sampling. Response:

```json
{
  "generated_text": ["..."],
  "inference_time": 1.83,
  "tokens_per_second": 54.6,
  "model_type": "int4",
  "prompt_tokens": 7,
  "generated_tokens": 100,
  "device": "cuda"
}
```

Only the continuation is returned: the prompt is stripped by slicing the token
sequence, not the decoded string, so special tokens and normalisation cannot
bleed into the output.

### Benchmark

```bash
curl "http://localhost:8000/benchmark/int8?iterations=10&max_new_tokens=64"
curl "http://localhost:8000/benchmark/all?iterations=3"
```

Each run does a discarded warmup pass first, then times `iterations` sweeps over
three prompts of increasing length. Throughput is total generated tokens over
total wall time. `model_memory_mb` is the weight footprint of that variant
alone, measured from its tensors — including the packed weights of a
dynamically quantized model, which appear in neither `parameters()` nor
`buffers()`. `memory_usage` next to it is whole-process and therefore not
comparable between variants.

`/benchmark/all` adds a `comparison` block:

```json
{
  "comparison": {
    "original": {"latency_speedup": 1.0, "throughput_ratio": 1.0, "weight_memory_ratio": 1.0},
    "int4": {"latency_speedup": 2.7, "throughput_ratio": 2.7, "weight_memory_ratio": 0.27}
  }
}
```

`weight_memory_ratio` is measured, not derived from the bit width, so a variant
that quietly failed to quantize shows up as `1.0`.

The sweep releases each variant once it has been measured, so it runs on a
device that holds a single copy of the weights; pass `keep_loaded=true`
(`--keep-loaded` on the CLI) to keep them resident. Variants that were already
loaded before the sweep are never evicted.

### Status codes

| Code | Meaning |
| --- | --- |
| `404` | Variant does not exist |
| `409` | Variant exists but needs a different device than this host has |
| `503` | Variant is supported but failed to load (missing weights, OOM) |
| `400` | Request exceeds a configured server-side limit |
| `422` | Request failed schema validation |

## Configuration

Everything is environment driven with the `AIRAVATA_` prefix; see
[`.env.example`](.env.example) for the full list with defaults. A `.env` file in
the working directory is loaded automatically, and real environment variables
take precedence over it.

The ones worth knowing:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AIRAVATA_MODEL_NAME` | `ai4bharat/Airavata` | Any causal-LM repo id works |
| `AIRAVATA_DEVICE` | `auto` | `auto`, `cpu` or `cuda`; `cuda` fails fast if absent |
| `AIRAVATA_PRELOAD` | `auto` | `auto`, `none`, or e.g. `original,int4` |
| `AIRAVATA_HOST` / `AIRAVATA_PORT` | `127.0.0.1` / `8000` | Bind address |
| `AIRAVATA_HF_TOKEN` | unset | Token for gated repos; `HF_TOKEN` also works |
| `AIRAVATA_MAX_NEW_TOKENS` | `2048` | Server-side ceiling on generation length |
| `AIRAVATA_TRUST_REMOTE_CODE` | `true` | Airavata ships custom modeling code |

Never commit a real token. `.env` is gitignored.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The suite runs in seconds and downloads nothing. `tests/conftest.py` subclasses
`ModelManager` and replaces only the two methods that touch the Hugging Face
hub, so the real encode/generate/decode/benchmark/export paths are exercised
against a deterministic fake model.

## Layout

```
airavata_quant/
  config.py        Environment-driven settings and .env loading
  quantization.py  Variant registry (pure data, no torch)
  schemas.py       Pydantic request/response models
  benchmark.py     Percentiles, aggregation and comparison (pure Python)
  hardware.py      psutil/torch/GPUtil telemetry, degrades to None
  manager.py       Loading, generation, benchmarking, export
  api.py           FastAPI app factory
  cli.py           serve / benchmark / export / info
```

`config.py`, `quantization.py`, `benchmark.py` and `schemas.py` import no torch,
which is why the fast tests are fast.

## Production notes

Run under a process manager with a real ASGI worker setup:

```bash
gunicorn airavata_quantization_service:app \
  -k uvicorn.workers.UvicornWorker -w 1 --bind 0.0.0.0:8000 --timeout 300
```

Use **one worker per GPU**. Each worker loads its own copy of the weights, so
`-w 4` on a single card will run it out of memory. Scale concurrency with
`AIRAVATA_MAX_WORKERS` (the inference thread pool) instead, and put a rate
limiter and TLS termination in front.

`/health` is the container healthcheck target and stays responsive even when a
variant failed to load, so a bad checkpoint does not turn into a crash loop.

## Troubleshooting

**`bitsandbytes` fails to import on Windows.** The int8/int4 variants report as
unavailable; the FP16 baseline still works. WSL2 is the reliable path for
bitsandbytes on Windows.

**CUDA out of memory.** Unload the baseline before loading a quantized variant
(`DELETE /models/original`), or start with `AIRAVATA_PRELOAD=int4`.

**Model download is slow or fails.** The checkpoint is tens of gigabytes. Point
`AIRAVATA_CACHE_DIR` at a disk with room, and set `AIRAVATA_HF_TOKEN` if the
repository is gated.

**`device=cuda was requested but torch reports no CUDA device`.** Your torch is
a CPU build. Reinstall from the CUDA wheel index.

## License

MIT. See [LICENSE](LICENSE).

The Airavata model itself is licensed separately by AI4Bharat; check the model
card before using it in production.
