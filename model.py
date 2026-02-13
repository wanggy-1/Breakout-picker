"""
DeepLabV3+ with the ResNet-18 backbone and circular convolution.
"""

import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# -------------------------
# Circular convolution (only width wraps)
# -------------------------
class CircularConv2d(nn.Module):
    """
    Conv2d with circular padding ONLY along width (theta) dimension and zero padding along height (z) dimension.
    Input:  [B, C, Z(H), Theta(W)]
    kernel_size: (kz, ktheta)
    dilation:    (dz, dtheta)
    stride:      (sz, stheta)
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=(3, 3),
        stride=(1, 1),
        dilation=(1, 1),
        bias=False,
        groups=1,
    ):
        super().__init__()
        self.kz, self.kt = (
            kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        )
        self.sz, self.st = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        self.dz, self.dt = (
            dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
        )
        # SAME padding each dim:
        self.pad_z = (self.kz // 2) * self.dz
        self.pad_t = (self.kt // 2) * self.dt
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(self.kz, self.kt),
            stride=(self.sz, self.st),
            dilation=(self.dz, self.dt),
            padding=0,  # we pad manually
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        # x: [B,C,Z,Theta]
        if self.pad_t > 0:
            x = F.pad(x, (self.pad_t, self.pad_t, 0, 0), mode="circular")  # pad width only
        if self.pad_z > 0:
            x = F.pad(x, (0, 0, self.pad_z, self.pad_z), mode="constant", value=0.0)
        return self.conv(x)


# -------------------------
# Circular MaxPool2d (only width wraps)
# -------------------------
class CircularMaxPool2d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1, ceil_mode=False):
        super().__init__()
        # normalize to tuples
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        if not isinstance(stride, (tuple, list)):
            stride = (stride, stride)
        if not isinstance(padding, (tuple, list)):
            padding = (padding, padding)
        self.kz, self.kt = kernel_size
        self.sz, self.st = stride
        self.pz, self.pt = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        # Width-dim circular padding; height-dim zero padding
        if self.pt > 0:
            x = F.pad(x, (self.pt, self.pt, 0, 0), mode="circular")
        if self.pz > 0:
            x = F.pad(x, (0, 0, self.pz, self.pz), mode="constant", value=0.0)
        return F.max_pool2d(
            x,
            kernel_size=(self.kz, self.kt),
            stride=(self.sz, self.st),
            padding=0,
            ceil_mode=self.ceil_mode,
        )


# -------------------------
# Utilities: recursively convert Conv2d/MaxPool2d to circular versions
# -------------------------

def convert_conv_pool_to_circular(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        # replace Conv2d
        if isinstance(child, nn.Conv2d):
            k = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size, child.kernel_size)
            s = child.stride if isinstance(child.stride, tuple) else (child.stride, child.stride)
            d = child.dilation if isinstance(child.dilation, tuple) else (child.dilation, child.dilation)
            new_conv = CircularConv2d(
                in_ch=child.in_channels,
                out_ch=child.out_channels,
                kernel_size=k,
                stride=s,
                dilation=d,
                bias=(child.bias is not None),
                groups=child.groups,
            )
            # weight copy for compatibility with (optional) pretrained weights
            with torch.no_grad():
                new_conv.conv.weight.copy_(child.weight)
                if child.bias is not None and new_conv.conv.bias is not None:
                    new_conv.conv.bias.copy_(child.bias)
            setattr(module, name, new_conv)
        # replace MaxPool2d
        elif isinstance(child, nn.MaxPool2d):
            k = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size, child.kernel_size)
            s = child.stride if child.stride is not None else k
            if not isinstance(s, tuple):
                s = (s, s)
            p = child.padding if isinstance(child.padding, tuple) else (child.padding, child.padding)
            new_pool = CircularMaxPool2d(kernel_size=k, stride=s, padding=p, ceil_mode=child.ceil_mode)
            setattr(module, name, new_pool)
        else:
            convert_conv_pool_to_circular(child)
    return module


# -------------------------
# ASPP (isotropic: dilate along both z and theta)
# -------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates: List[int], dropout=0.2):
        super().__init__()
        blocks = []

        # 1x1
        blocks.append(
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        )

        # dilated branches (kz=3, kt=3, dilate along both z and theta)
        for r in rates:
            blocks.append(
                nn.Sequential(
                    CircularConv2d(
                        in_ch, out_ch, kernel_size=(3, 3), dilation=(r, r), bias=False
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # image pooling branch (global)
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(blocks)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(blocks) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        feats = [b(x) for b in self.blocks]
        gp = self.img_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(gp)
        x = torch.cat(feats, dim=1)
        return self.project(x)


# -------------------------
# Decoder head (DeepLabV3+)
# -------------------------
class DecoderHead(nn.Module):
    def __init__(self, low_ch, out_ch, aspp_ch=256):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.conv1 = CircularConv2d(aspp_ch + 48, out_ch, kernel_size=(3, 3))
        self.conv2 = CircularConv2d(out_ch, out_ch, kernel_size=(3, 3))
        self.out_bn = nn.BatchNorm2d(out_ch)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, aspp_feat, low_feat):
        # aspp_feat upsample to low_feat size
        x = F.interpolate(aspp_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low_feat)
        x = torch.cat([x, low], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_bn(x)
        x = self.out_act(x)
        return x


# -------------------------
# Backbone: ResNet-18 (OS=16) with low-level feature (circular conv/pool)
# -------------------------
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False, os16: bool = True):
        super().__init__()
        # build standard resnet18 first
        m = resnet18(weights=None if not pretrained else None)

        # control output stride: modify layer3/layer4 strides & dilations before conversion
        if os16:
            # keep /16 at layer3
            m.layer3[0].conv1.stride = (2, 2)
            if m.layer3[0].downsample is not None:
                m.layer3[0].downsample[0].stride = (2, 2)
            # dilate layer4 (stride 1, dilation 2)
            for b in m.layer4:
                b.conv1.dilation = (2, 2)
                b.conv1.padding = (2, 2)
                b.conv1.stride = (1, 1)
                b.conv2.dilation = (2, 2)
                b.conv2.padding = (2, 2)
                if b.downsample is not None:
                    b.downsample[0].stride = (1, 1)

        # convert all Conv2d & MaxPool2d to circular versions
        m = convert_conv_pool_to_circular(m)

        # expose layers
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        # x: [B,C,Z,Theta] ; treat Z as H, Theta as W
        x = self.conv1(x)  # /2 (theta-circular)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /4 (theta-circular maxpool)
        low = self.layer1(x)  # /4
        x = self.layer2(low)  # /8
        x = self.layer3(x)   # /16
        x = self.layer4(x)   # /16 (dilated)
        return x, low


# -------------------------
# The full DeepLabV3+
# -------------------------
class DeepLabV3PlusResNet18(nn.Module):
    def __init__(
        self,
        in_ch=4,
        num_classes=1,
        aspp_out=256,
        rates=(1, 3, 5, 7), 
    ):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=False, os16=True)
        # input stem for arbitrary in_ch (replace the first conv)
        self.backbone.conv1 = CircularConv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), dilation=(1, 1), bias=False)
        self.aspp = ASPP(in_ch=512, out_ch=aspp_out, rates=list(rates), dropout=0.2)
        self.decoder = DecoderHead(low_ch=64, out_ch=128, aspp_ch=aspp_out)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)
        # Init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,C,Z,Theta]
        feats, low = self.backbone(x)  # [B,512,Z/16,Theta/16]
        x = self.aspp(feats)           # [B,aspp,Z/16,Theta/16]
        x = self.decoder(x, low)       # up to /4
        logits = self.classifier(x)    # [B,1,Z/4,Theta/4]
        # Upscale to the input's resolution
        logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
        return logits  # logits (not sigmoid)


# -------------------------
# Dice loss
# -------------------------
class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,1,Z,Theta]
        targets: [B,1,Z,Theta] in {0,1}
        """
        eps = 1e-6
        probs = torch.sigmoid(logits)
        # flatten per-sample
        B = probs.shape[0]
        p = probs.view(B, -1)
        t = targets.view(B, -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        dice = (2 * inter + eps) / (denom + eps)
        loss = 1 - dice
        return loss.mean()


class DiceBCELoss(nn.Module):
    """
    Dice + BCE loss for binary segmentation.
    """
    def __init__(
        self, 
        dice_weight: float = 1.0, 
        bce_weight: float = 1.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        targets = targets.float()
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return loss, {"dice": float(dice_loss.detach()), "bce": float(bce_loss.detach())}


# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C_in, Z, T = 2, 4, 256, 256  # example sizes
    x = torch.randn(B, C_in, Z, T, device=device)
    y = (torch.rand(B, 1, Z, T, device=device) > 0.95).float()

    model = DeepLabV3PlusResNet18(
        in_ch=C_in, 
        num_classes=1, 
        aspp_out = 256, 
        rates=(1, 3, 5, 7)
    ).to(device)
    
    criterion = DiceLoss()

    logits = model(x)
    loss = criterion(logits, y)
    print("loss:", loss.item())
