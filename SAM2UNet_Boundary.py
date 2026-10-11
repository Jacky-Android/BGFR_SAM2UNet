"""BGFR-SAM2-UNet: boundary-guided feature refinement for binary building extraction.

Place this file next to the original repository's SAM2UNet.py.
It reuses the frozen SAM2-Hiera, adapters, RFBs and U-shaped decoder.
No changes are required inside the vendored `sam2` package.
"""
import torch
from torch import nn
from torch.nn import functional as F
from SAM2UNet import SAM2UNet


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DSConvBNAct(nn.Sequential):
    """3x3 depthwise spatial filtering + 1x1 pointwise channel mixing."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BGFRSAM2UNet(SAM2UNet):
    """A minimal extension of official SAM2-UNet for sharper binary boundaries.

    Returns:
      mask_logits:   [B,1,H,W]       main building logits
      aux1_logits:   [B,1,H,W]       original coarse auxiliary head
      aux2_logits:   [B,1,H,W]       original middle auxiliary head
      edge_logits:   [B,1,H/2,W/2]  auxiliary boundary logits

    Requirement: image H,W divisible by 32, recommended 512x512.
    """
    def __init__(self, checkpoint_path=None, detail_channels=24,
                 refine_channels=32, enable_edge_gate=True):
        super().__init__(checkpoint_path=checkpoint_path)
        d, c = detail_channels, refine_channels
        self.enable_edge_gate = enable_edge_gate

        # HRDB: RGB -> 1/2 spatial resolution (24 channels by default).
        self.detail_stem = nn.Sequential(
            ConvBNAct(3, d, kernel_size=3, stride=2),
            DSConvBNAct(d, d),
        )
        # Original decoder features are [B,64,H/4,W/4].
        self.decoder_project = ConvBNAct(64, c, kernel_size=1)

        # Edge prediction sees both SAM2 semantics and RGB fine details.
        self.edge_fusion = DSConvBNAct(c + d, c)
        self.edge_head = nn.Conv2d(c, 1, kernel_size=1)

        # A spatial/channel gate; edge probability also multiplies the gate
        # explicitly, so boundary prediction is used IN THE MAIN PATH.
        self.detail_gate = nn.Sequential(
            nn.Conv2d(c, d, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine_fusion = nn.Sequential(
            DSConvBNAct(c + d, c),
            DSConvBNAct(c, c),
        )
        self.refine_head = nn.Conv2d(c, 1, kernel_size=1)
        # At initialization, main output equals original coarse head exactly.
        nn.init.zeros_(self.refine_head.weight)
        nn.init.zeros_(self.refine_head.bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected RGB image [B,3,H,W].')
        h, w = image.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError('Image H and W should be divisible by 32.')

        # The complete official SAM2-UNet encoder + RFB + decoder is unchanged.
        x1, x2, x3, x4 = self.encoder(image)
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)
        dec3 = self.up1(x4, x3)            # 64 x H/16 x W/16
        aux1 = self.side1(dec3)
        dec2 = self.up2(dec3, x2)          # 64 x H/8 x W/8
        aux2 = self.side2(dec2)
        dec1 = self.up3(dec2, x1)          # 64 x H/4 x W/4

        # Preserve the original building logits as a residual base.
        coarse = F.interpolate(
            self.head(dec1), size=(h, w), mode='bilinear', align_corners=False
        )
        aux1 = F.interpolate(aux1, size=(h, w), mode='bilinear', align_corners=False)
        aux2 = F.interpolate(aux2, size=(h, w), mode='bilinear', align_corners=False)

        # HRDB and projection both produce feature maps at H/2 x W/2.
        detail = self.detail_stem(image)    # d x H/2 x W/2
        semantic = self.decoder_project(dec1)
        semantic = F.interpolate(
            semantic, size=detail.shape[-2:],
            mode='bilinear', align_corners=False
        )                                    # c x H/2 x W/2

        edge_feature = self.edge_fusion(torch.cat([semantic, detail], dim=1))
        edge_logits = self.edge_head(edge_feature)  # 1 x H/2 x W/2
        edge_probability = torch.sigmoid(edge_logits)

        # BGM: emphasize detail ONLY where the model believes there is an edge.
        # The +1 path retains all original local detail, including missed edges.
        if self.enable_edge_gate:
            gate = self.detail_gate(edge_feature)
            guided_detail = detail * (1.0 + edge_probability * gate)
        else:
            guided_detail = detail  # useful for a controlled ablation

        # BRH: predict a spatial correction to original mask logits.
        refined = self.refine_fusion(torch.cat([semantic, guided_detail], dim=1))
        residual = self.refine_head(refined)   # 1 x H/2 x W/2
        residual = F.interpolate(
            residual, size=(h, w), mode='bilinear', align_corners=False
        )
        final_logits = coarse + residual
        return final_logits, aux1, aux2, edge_logits
