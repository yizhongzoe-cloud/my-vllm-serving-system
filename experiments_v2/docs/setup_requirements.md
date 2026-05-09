# Setup & Requirements

Steps to set up this fork on a new machine (e.g., L40S cluster).

## 1. Clone the repo

```bash
cd /home/yzhong76/code
git clone <repo-url> my-vllm-serving-system
cd my-vllm-serving-system
```

If you already have it, just `cd /home/yzhong76/code/my-vllm-serving-system`.

## 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Python 3.10–3.13 all work (`pyproject.toml` declares `>=3.10,<3.14`). On Ubuntu 24.04 the system `python3` is 3.12 — use it directly, no need to install 3.10.

## 3. Upgrade base build tools

```bash
pip install -U pip setuptools wheel
```

## 4. Install dependencies

```bash
pip install -r requirements/cuda.txt    # pulls in common.txt automatically
pip install -r requirements/ft.txt      # ortools + weasyprint (FT-specific extras)
```

## 5. Install this fork as editable

```bash
VLLM_USE_PRECOMPILED=1 pip install -e .
```

`VLLM_USE_PRECOMPILED=1` tells pip to download the official prebuilt CUDA `.so` files instead of compiling `csrc/` locally — saves 20–30 minutes and avoids needing `nvcc` / CUDA toolkit on the host.

Drop the flag (plain `pip install -e .`) only if you've modified `csrc/` and need your C++/CUDA changes compiled in. Then `nvcc` (CUDA toolkit) must be installed and on PATH.

Either way, this makes Python-side modifications under `vllm/` usable without reinstalling on every code change.

## 6. Verify

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "from ortools.sat.python import cp_model; print('ortools OK')"
python -c "import torch; assert torch.cuda.is_available(); print('GPUs:', torch.cuda.device_count())"
```

All three should print and exit cleanly. The `ortools` check matters: without it the Benders FT scheduler silently falls back to greedy admission and Our-System experiment results are invalid (see `requirements/ft.txt` for the incident note).

## Notes

- If the L40S host already has CUDA 12.x installed, the `cuda.txt` requirements should match. If not, you may need to manually pin `torch` / `flash-attn` to versions matching the host CUDA.
- `ft.txt` contains `ortools` (Benders solver) and `weasyprint` (for PDF report generation). If you don't plan to run the solver or generate reports, this file can be skipped.
- For multi-GPU setups (TP=2 on L40S), no extra install steps — vLLM handles it via `--tensor-parallel-size 2`.
