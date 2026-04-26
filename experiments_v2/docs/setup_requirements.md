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
python3.10 -m venv venv
source venv/bin/activate
```

Use Python 3.10 — newer versions may not match vLLM's CUDA build.

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
pip install -e .
```

This makes our local vLLM modifications usable without reinstalling on every code change.

## 6. Verify

```bash
python -c "import vllm; print(vllm.__version__)"
```

Should print a version string and exit cleanly. If it errors out, check CUDA driver / Python version match.

## Notes

- If the L40S host already has CUDA 12.x installed, the `cuda.txt` requirements should match. If not, you may need to manually pin `torch` / `flash-attn` to versions matching the host CUDA.
- `ft.txt` contains `ortools` (Benders solver) and `weasyprint` (for PDF report generation). If you don't plan to run the solver or generate reports, this file can be skipped.
- For multi-GPU setups (TP=2 on L40S), no extra install steps — vLLM handles it via `--tensor-parallel-size 2`.
