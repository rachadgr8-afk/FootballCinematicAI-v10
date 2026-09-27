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
