# BGFR-SAM2-UNet — implementation plan and starter code

This is an **experimental proposal**, not an independently trained or benchmarked model.
It inherits the [official SAM2-UNet](https://github.com/WZH0120/SAM2-UNet) implementation unchanged and adds only boundary-aware high-resolution refinement.

## Exact placement

1. Copy `SAM2UNet_Boundary.py` and `train_boundary_whu.py` to the **root** of the official SAM2-UNet repository.
2. The original `SAM2UNet.py`, encoder, adapters, RFB, and U-shape decoder do not change.
3. `BGFRSAM2UNet.forward` reproduces the original 3 decoder stages and adds the refinement branch **after `self.up3(...)`**.
4. Returns `(final_mask_logits, aux1_logits, aux2_logits, edge_logits_half)`.
5. The Hiera weights must be the original **SAM2 Hiera-Large**, not SAM2.1.

## Core tensor sizes for RGB 512x512

- Decoder output: `64 x 128 x 128`.
- RGB detail stem: `24 x 256 x 256`.
- Decoder projection and upsampling: `32 x 256 x 256`.
- Boundary features/logits: `32 x 256 x 256` / `1 x 256 x 256`.
- Boundary-gated fusion features: `32 x 256 x 256`.
- Residual logits: `1 x 256 x 256`.
- Main output: `1 x 512 x 512` from **coarse logits + upsampled residual logits**.

## Dataset layout

```text
WHU/
  train/images/001.tif
  train/masks/001.tif
  val/images/002.tif
  val/masks/002.tif
```

All image/mask filenames should have identical stems, and grayscale masks should be 0/255 (or 0/1). Images are converted to RGB; masks are binarized using `>0`, and resized with nearest-neighbor interpolation. If your masks use other conventions, change `BuildingDataset` carefully.

## Launch

```bash
python train_boundary_whu.py \
  --hiera_path /path/to/sam2_hiera_large.pt \
  --train_images /path/to/WHU/train/images \
  --train_masks /path/to/WHU/train/masks \
  --val_images /path/to/WHU/val/images \
  --val_masks /path/to/WHU/val/masks \
  --output ./outputs/bgfr \
  --size 512 --batch_size 1 --accum 2 --epochs 45
```

Checkpoint: `outputs/bgfr/bgfr_best.pth`.

## Inference

```python
import torch
from SAM2UNet_Boundary import BGFRSAM2UNet

ckpt = torch.load('outputs/bgfr/bgfr_best.pth', map_location='cpu', weights_only=True)
model = BGFRSAM2UNet(
    checkpoint_path=None,
    enable_edge_gate=not ckpt['args']['disable_edge_gate'],
)
model.load_state_dict(ckpt['model'], strict=True)
model.eval()
with torch.no_grad():
    logits, _, _, edge_half = model(image_batch)  # normalized RGB Bx3x512x512
    building = (logits.sigmoid() >= 0.5).to(torch.uint8)
    edge_prob = edge_half.sigmoid()
```

Do not per-image Min-Max normalize building probabilities. For general use, load trusted checkpoints only. If needed, adapt the `torch.load` usage to your installed version (checkpoint is a dict with metadata).

## Critical cautions

- A 0/1 **semantic** building mask cannot supervise boundaries between two adjacent buildings when their labels merge. Instance polygons are necessary to measure instance separation.
- In this starter implementation, the edge target is a morphological band at half resolution (`kernel_size=3`), derived from downsampled masks; this is *not* an extra human boundary annotation.
- Do not feed the ground-truth edge mask into the network's forward pass at training/inference. Only a **predicted** edge probability should guide the features.
- Keep all other training conditions identical when comparing to baseline, and run both baseline and proposed model with 512×512 inputs.
- Start with E0 official baseline (trained with same 512px conditions), E1 detail+residual only (`--disable_edge_gate --edge_weight 0`), E2 E1 + edge auxiliary supervision (`--disable_edge_gate --edge_weight 0.2`), E3 full BGFR (default settings), E4 full gating without supervised edge (`--edge_weight 0`). Distinguish changes due to higher resolution, extra parameters and boundary supervision.
- The training code is a demonstration template. Use a held-out **non-leaking** validation split and run multiple seeds for publication.

## Batch prediction (binary masks and optional edge maps)

Requires `tqdm` for a visible progress bar: `pip install tqdm`.

```bash
python predict_boundary.py \
  --checkpoint outputs/bgfr/bgfr_best.pth \
  --image_dir /path/to/WHU/test/images \
  --output_dir outputs/bgfr_predictions \
  --size 512 --threshold 0.5 --save_edge
```

Outputs are binary building PNGs (0 background, 255 building), plus `_edge.png` confidence maps when requested. No min-max scaling is applied to predictions. If original images are not 512x512, the output is resized to their original dimensions for saving; for accurate georeferencing, retain source GeoTIFF spatial metadata separately.

### Prediction + quantitative evaluation (IoU, F1, Precision, Recall, BF1)

If you have test-set ground-truth masks, give the additional `--gt_dir` argument:

```bash
python predict_boundary.py \
  --checkpoint outputs/bgfr/bgfr_best.pth \
  --image_dir /path/to/WHU/test/images \
  --gt_dir /path/to/WHU/test/masks \
  --output_dir outputs/bgfr_predictions \
  --size 512 --threshold 0.5 --save_edge \
  --boundary_tolerance 2
```

- `--mask_dir` is an alias for `--gt_dir`. If omitted, only prediction images are generated; no numeric metrics are computed.
- Image and GT files must share their **stem** (e.g. `001.tif` vs `001.png` are matched). Missing GT files, duplicate stems, or mismatched original image/GT dimensions raise a clear error.
- GT labels are converted to grayscale and binarized as `pixel > 0`, as in the training script. Binary predictions are thresholded at `--threshold` without per-image min-max normalization.
- Progress: a single `tqdm` bar (`Predicting TEST`) displays processed-image count, elapsed time and speed; no per-image metric lines flood the terminal.
- **After all TEST images**: print **overall foreground IoU, F1, Precision (Pre), Recall, and Boundary F1 (micro and macro)** as decimals and percentages. These are true dataset-level **micro** metrics from all pixel TP/FP/FN combined, **not** the average of image scores.
- `metrics_per_image.csv`: each image's foreground IoU, F1, Precision (`Pre`), Recall, and boundary F1, plus TP/FP/FN.
- `metrics_summary.json`: **micro** pixel-wise aggregate metrics and **macro** per-image averages. Boundary F1 is reported as both a **micro boundary-count aggregate** and a **macro per-image mean**, with `--boundary_tolerance` measured in original-image pixels.
- Both prediction and GT containing zero foreground pixels are treated as perfect agreement (score 1). If prediction is empty while GT has buildings, precision is defined as 0.
- The foreground IoU is the **building-class IoU**, not a two-class (building + background) mean IoU; define and report this distinction in any manuscript.
- Predictions and GT are compared at the **original image resolution** (after restoring logits to source image size). A full-scene giant image should be processed using overlap-tile inference, not squashed to 512 x 512.


### CUDA cache cleanup

The training script now calls `torch.cuda.empty_cache()` once after each epoch (after validation and checkpoint saving), only when CUDA is available. This releases unused cached blocks, not GPU memory occupied by live tensors; it is not a substitute for reducing batch size when CUDA OOM occurs.
