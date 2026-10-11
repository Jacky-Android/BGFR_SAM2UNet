"""Predict building masks and optional boundary probabilities from BGFR weights.

The checkpoint stores the full model state (including SAM2-Hiera). It does not
require supplying the original Hiera checkpoint again at inference time.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from SAM2UNet_Boundary import BGFRSAM2UNet


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    use_gate = not ckpt.get('args', {}).get('disable_edge_gate', False)
    model = BGFRSAM2UNet(checkpoint_path=None, enable_edge_gate=use_gate)
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_files = sorted(p for p in Path(args.image_dir).iterdir()
                         if p.is_file() and p.suffix.lower() in
                         {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'})
    for path in image_files:
        with Image.open(path) as im:
            rgb = im.convert('RGB')
        original_hw = rgb.height, rgb.width
        inp = TF.resize(rgb, [args.size, args.size],
                        interpolation=InterpolationMode.BILINEAR)
        inp = TF.normalize(TF.to_tensor(inp),
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        with torch.no_grad():
            mask_logits, _, _, edge_half = model(inp.unsqueeze(0).to(device))
            # For original-size geometry restore logits, then take sigmoid.
            mask_logits = F.interpolate(mask_logits.float(),
                                        size=original_hw, mode='bilinear',
                                        align_corners=False)
            building = (torch.sigmoid(mask_logits)[0, 0] >= args.threshold)
            building = building.to(torch.uint8).cpu().numpy() * 255
            edge = torch.sigmoid(F.interpolate(
                edge_half.float(), size=original_hw,
                mode='bilinear', align_corners=False
            ))[0, 0].cpu().numpy()
        Image.fromarray(building).save(output / f'{path.stem}.png')
        if args.save_edge:
            Image.fromarray((np.clip(edge, 0, 1) * 255).astype(np.uint8)).save(
                output / f'{path.stem}_edge.png'
            )
        print(path.name, '->', output / f'{path.stem}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--save_edge', action='store_true')
    main(ap.parse_args())
