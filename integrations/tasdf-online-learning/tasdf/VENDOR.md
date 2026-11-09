# Vendored TASDF subset

Copied verbatim (no edits) so the backend builds without the research repo.
Re-copy from upstream instead of editing these files.

Source repo: `D:\Workspace\model\tasdf-audit`, commit `da777bc` (2026-11-09).

| File here | Upstream | sha256 |
|---|---|---|
| `models/__init__.py` | `tasdf/models/__init__.py` | 750d7134c900… |
| `models/unet_blocks.py` | `tasdf/models/unet_blocks.py` | 583dc5e90bf6… |
| `models/unet_backbone.py` | `tasdf/models/unet_backbone.py` | 5fe16dc77512… |
| `models/dual_heads.py` | `tasdf/models/dual_heads.py` | 67892b4e5197… |
| `models/itgm.py` | `tasdf/models/itgm.py` | 9470e83a59a4… |
| `models/final_model.py` | `tasdf/models/final_model.py` | beb25acd3471… |
| `losses/losses.py` | `variants/fix_m2b/tasdf/losses/losses.py` (M2-B: CPS loss; **untracked upstream at copy time**) | ebfe8ef52eb7… |
| `losses/__init__.py` | `tasdf/losses/__init__.py` | — |
| `utils/metrics_v2.py` | `tasdf/utils/metrics_v2.py` | 31969ef78346… |

The baseline checkpoint (M2-B, `experiments/fix_m2b/run/checkpoints/best.pt`) was trained with
`model.erosion = "cross"`, `base_channels = 64`, `k_iters = 10`, no denoise, image size 256.
