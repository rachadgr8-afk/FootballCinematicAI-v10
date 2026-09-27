#!/usr/bin/env python3
try:
    import torch
    import torchvision
except Exception as e:
    print(f'CPU Torch import FAILED: {e}')
    raise SystemExit(1)
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.__version__ != '2.6.0+cpu':
    raise SystemExit(f'Expected torch 2.6.0+cpu, found {torch.__version__}')
if torchvision.__version__ != '0.21.0+cpu':
    raise SystemExit(f'Expected torchvision 0.21.0+cpu, found {torchvision.__version__}')
if torch.cuda.is_available():
    raise SystemExit('CPU build unexpectedly reports CUDA available')
print('CPU PyTorch wheel verification: PASS')
