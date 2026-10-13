"""Predict BGFR-SAM2-UNet building masks, optionally evaluate them against GT.

Requires the original SAM2-UNet project and a BGFR training checkpoint.

Examples:
  # Predict only:
  python predict_boundary.py --checkpoint outputs/bgfr/bgfr_best.pth \
      --image_dir WHU/test/images --output_dir outputs/predictions --save_edge

  # Predict and calculate foreground IoU, F1, Precision, Recall and BF1:
  python predict_boundary.py --checkpoint outputs/bgfr/bgfr_best.pth \
      --image_dir WHU/test/images --gt_dir WHU/test/masks \
      --output_dir outputs/predictions --size 512 --save_edge

Outputs (when GT is supplied): test-wide IoU/F1/Precision/Recall and BF1 micro/macro printed AFTER
  the progress bar, plus metrics_per_image.csv and metrics_summary.json.
  Requires: pip install tqdm
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from SAM2UNet_Boundary import BGFRSAM2UNet

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def index_images(folder, label):
    """Index by stem, not position; reject ambiguous duplicate stems."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'{label} folder does not exist: {folder}')
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f'No supported image files found in {label} folder: {folder}')
    indexed = {}
    for path in paths:
        if path.stem in indexed:
            raise ValueError(f'Ambiguous {label} files with same stem: '
                             f'{indexed[path.stem]} and {path}')
        indexed[path.stem] = path
    return indexed


def count_pixels(prediction, ground_truth):
    """Compute the exact binary confusion counts at original image resolution.

    prediction, ground_truth: 2D bool numpy arrays; True means building.
    """
    if prediction.shape != ground_truth.shape:
        raise ValueError(f'Prediction/GT size mismatch: '
                         f'{prediction.shape} vs {ground_truth.shape}')
    tp = int(np.count_nonzero(prediction & ground_truth))
    fp = int(np.count_nonzero(prediction & ~ground_truth))
    fn = int(np.count_nonzero(~prediction & ground_truth))
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    """Foreground metrics, with empty/empty = 1 for every metric.

    If the model predicts no building but GT has buildings, precision is
    defined as 0, since no positive predictions can be correct.
    """
    if tp == fp == fn == 0:
        return {'iou': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'iou': iou, 'f1': f1, 'precision': precision, 'recall': recall}


def boundary_match_counts(prediction, ground_truth, tolerance=2):
    """Return matched/total boundary pixel counts, independently per image.

    3x3 morphological boundary at the ORIGINAL image resolution.
    Two boundary pixels are matched when their Chebyshev distance is <= tolerance.
    Returned tuple: (matched_pred, total_pred, matched_gt, total_gt).
    """
    if tolerance < 0:
        raise ValueError('boundary tolerance must be >= 0')
    if prediction.shape != ground_truth.shape:
        raise ValueError('Prediction and GT boundary shape mismatch')
    pred = torch.from_numpy(prediction.astype(np.float32))[None, None]
    gt = torch.from_numpy(ground_truth.astype(np.float32))[None, None]

    def border(x):
        dilated = F.max_pool2d(x, 3, stride=1, padding=1)
        eroded = -F.max_pool2d(-x, 3, stride=1, padding=1)
        return (dilated - eroded).clamp(0, 1)

    ep, eg = border(pred), border(gt)
    pred_total = int(ep.sum().item())
    gt_total = int(eg.sum().item())
    if pred_total == 0 or gt_total == 0:
        return 0, pred_total, 0, gt_total

    k = 2 * tolerance + 1
    near_gt = F.max_pool2d(eg, k, stride=1, padding=tolerance)
    near_pred = F.max_pool2d(ep, k, stride=1, padding=tolerance)
    matched_pred = int((ep * near_gt).sum().item())
    matched_gt = int((eg * near_pred).sum().item())
    return matched_pred, pred_total, matched_gt, gt_total


def boundary_f1_from_matches(matched_pred, total_pred, matched_gt, total_gt):
    """Boundary F1 from boundary counts; both empty -> 1, one empty -> 0."""
    if total_pred == 0 and total_gt == 0:
        return 1.0
    if total_pred == 0 or total_gt == 0:
        return 0.0
    precision = matched_pred / total_pred
    recall = matched_gt / total_gt
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def boundary_counts(prediction, ground_truth, tolerance=2):
    """Backward-compatible per-image boundary F1 helper."""
    return boundary_f1_from_matches(
        *boundary_match_counts(prediction, ground_truth, tolerance=tolerance)
    )


def main(args):
    if args.size <= 0 or args.size % 32 != 0:
        raise ValueError('--size must be a positive multiple of 32, e.g. 352 or 512.')
    if not (0 <= args.threshold <= 1):
        raise ValueError('--threshold must be in [0, 1].')
    if args.boundary_tolerance < 0:
        raise ValueError('--boundary_tolerance must be >= 0.')

    image_map = index_images(args.image_dir, 'image')
    gt_map = index_images(args.gt_dir, 'GT') if args.gt_dir else None
    if gt_map is not None:
        missing = sorted(set(image_map) - set(gt_map))
        if missing:
            raise RuntimeError(f'Missing GT masks for {len(missing)} images. '
                               f'First examples: {missing[:10]}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    use_gate = not ckpt.get('args', {}).get('disable_edge_gate', False)
    model = BGFRSAM2UNet(checkpoint_path=None, enable_edge_gate=use_gate)
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    total_tp = total_fp = total_fn = 0
    total_boundary_counts = [0, 0, 0, 0]  # matched_pred, pred_count, matched_gt, gt_count

    for stem, path in tqdm(image_map.items(), total=len(image_map),
                           desc='Predicting TEST', unit='img', dynamic_ncols=True):
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        original_hw = rgb.height, rgb.width
        inp = TF.resize(rgb, [args.size, args.size],
                        interpolation=InterpolationMode.BILINEAR)
        inp = TF.normalize(TF.to_tensor(inp),
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        with torch.inference_mode():
            mask_logits, _, _, edge_half = model(inp.unsqueeze(0).to(device))
            # Restore logits BEFORE sigmoid/threshold. No per-image min-max.
            mask_logits = F.interpolate(mask_logits.float(),
                                        size=original_hw, mode='bilinear',
                                        align_corners=False)
            building = (torch.sigmoid(mask_logits)[0, 0] >= args.threshold)
            building = building.to(torch.uint8).cpu().numpy().astype(bool)
            if args.save_edge:
                edge = torch.sigmoid(F.interpolate(
                    edge_half.float(), size=original_hw,
                    mode='bilinear', align_corners=False
                ))[0, 0].cpu().numpy()

        Image.fromarray(building.astype(np.uint8) * 255).save(output / f'{stem}.png')
        if args.save_edge:
            Image.fromarray((np.clip(edge, 0, 1) * 255).astype(np.uint8)).save(
                output / f'{stem}_edge.png'
            )

        if gt_map is None:
            continue

        with Image.open(gt_map[stem]) as im:
            gt = np.asarray(im.convert('L')) > 0  # Same GT convention as train.py
        if gt.shape != building.shape:
            raise ValueError(f'{path.name}: GT has shape {gt.shape} but prediction '
                             f'has shape {building.shape}. Check GT alignment.')

        tp, fp, fn = count_pixels(building, gt)
        scores = metrics_from_counts(tp, fp, fn)
        matched_counts = boundary_match_counts(
            building, gt, tolerance=args.boundary_tolerance
        )
        scores['boundary_f1'] = boundary_f1_from_matches(*matched_counts)
        for i, count in enumerate(matched_counts):
            total_boundary_counts[i] += count
        total_tp += tp
        total_fp += fp
        total_fn += fn
        rows.append({'image': path.name, 'gt': gt_map[stem].name,
                     'tp': tp, 'fp': fp, 'fn': fn, **scores})

    if not rows:
        print('Prediction finished. No GT folder was provided; metrics not calculated.')
        return

    csv_path = output / 'metrics_per_image.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = ['iou', 'f1', 'precision', 'recall', 'boundary_f1']
    macro = {key: float(np.mean([r[key] for r in rows])) for key in metrics}
    micro = metrics_from_counts(total_tp, total_fp, total_fn)
    # Micro BF1 aggregates boundary matches and counts across images, without
    # matching across different images. Macro BF1 averages per-image BF1.
    micro['boundary_f1'] = boundary_f1_from_matches(*total_boundary_counts)
    summary = {
        'image_count': len(rows),
        'foreground_label': 'building',
        'threshold': args.threshold,
        'boundary_tolerance_px': args.boundary_tolerance,
        'micro': micro,
        'macro': macro,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'boundary_counts': {
            'matched_pred_boundary': total_boundary_counts[0],
            'total_pred_boundary': total_boundary_counts[1],
            'matched_gt_boundary': total_boundary_counts[2],
            'total_gt_boundary': total_boundary_counts[3],
        },
        'boundary_definition': '3x3 morphological band; Chebyshev pixel tolerance',
        'empty_mask_convention': 'If both pred and GT have no building: all scores = 1.',
    }
    summary_path = output / 'metrics_summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Overall TEST result: count TP/FP/FN across ALL images first,
    # then compute metrics. Do not average per-image scores here.
    print('\n' + '=' * 60)
    print(f'TEST SET OVERALL RESULTS  |  {len(rows)} images')
    print('Building foreground class; aggregated TP/FP/FN (micro)')
    print('-' * 60)
    print(f'IoU       : {micro["iou"]:.4f}  ({micro["iou"] * 100:.2f}%)')
    print(f'F1        : {micro["f1"]:.4f}  ({micro["f1"] * 100:.2f}%)')
    print(f'Pre       : {micro["precision"]:.4f}  ({micro["precision"] * 100:.2f}%)')
    print(f'Recall    : {micro["recall"]:.4f}  ({micro["recall"] * 100:.2f}%)')
    print(f'BF1@{args.boundary_tolerance}px (micro) : '
          f'{micro["boundary_f1"]:.4f}  ({micro["boundary_f1"] * 100:.2f}%)')
    print(f'BF1@{args.boundary_tolerance}px (macro) : '
          f'{macro["boundary_f1"]:.4f}  ({macro["boundary_f1"] * 100:.2f}%)')
    print('=' * 60)
    print(f'Per-image scores saved to: {csv_path}')
    print(f'Aggregate scores saved to: {summary_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='BGFR building segmentation inference and optional GT evaluation'
    )
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--gt_dir', '--mask_dir', dest='gt_dir', default=None,
                    help='GT building binary mask folder; optional. Match by filename stem.')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--save_edge', action='store_true')
    ap.add_argument('--boundary_tolerance', type=int, default=2,
                    help='Boundary F1 tolerance in original-image pixels (default: 2).')
    main(ap.parse_args())
