# RIFE / PyTorch CPU setup

FootballCinematicAI ULTRA is configured for CPU-only server deployments by default.
The project pins a matching PyTorch/torchvision pair and adds the official PyTorch
CPU wheel index to `requirements.txt`.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The resulting pair is:
- `torch==2.6.0+cpu`
- `torchvision==0.21.0+cpu`

The upstream ECCV2022-RIFE requirements accept `torch>=1.6.0` and
`torchvision>=0.7.0`, so this pair satisfies those minimums. RIFE also selects
CUDA only when `torch.cuda.is_available()` is true; otherwise it uses CPU.

Do not replace the pinned CPU packages with an unqualified `pip install torch
torchvision` in the production CPU image.

RIFE is optional at runtime. On CPU-only infrastructure, interpolation can be
substantially slower than YOLO/FFmpeg processing; set `RIFE_ENABLED=false` when
throughput is more important than interpolation.

## Manual RIFE installation (required if you want interpolation)

RIFE is **not** a Python package and must **not** be added to `requirements*.txt`.
`pip install git+https://github.com/megvii-research/ECCV2022-RIFE.git` fails with
*"does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found"*
because the upstream repository is a research codebase, not a distributable wheel.

Instead, clone it next to the app and point the `RIFE_REPO` environment variable at it:

```bash
git clone --depth 1 https://github.com/megvii-research/ECCV2022-RIFE.git /opt/rife
# Download the trained model (e.g. RIFE_trained_model_v3.6.zip) into /opt/rife/train_log
export RIFE_REPO=/opt/rife        # the CLI invokes /opt/rife/inference_video.py via subprocess
```

If `/opt/rife/inference_video.py` is missing, the renderer raises
`RIFE not installed` and skips interpolation gracefully — the rest of the pipeline
(YOLO tracking, events, FFmpeg render, commentary) still works.
