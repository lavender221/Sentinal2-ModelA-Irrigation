"""
Model.py

模型架構與 Loss 函式。
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(in_channels: int = 17, encoder_name: str = "resnet34") -> nn.Module:
    """
    建立 U-Net 分割模型。

    Args:
        in_channels:  輸入通道數（預設 17 = 4 連續波段 + 13 SCL one-hot）
        encoder_name: Encoder 名稱（預設 resnet34，載入 ImageNet 預訓練權重）

    Returns:
        smp.Unet 模型實例
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
    )


class MaskedDiceBCELoss(nn.Module):
    """
    結合 BCE 與 Dice 的複合 Loss，排除下列位置的像素：
        - ignore_index（nodata，預設 999）
        - 0.5（「疑似但低度確信」的弱相關灌水標籤，不計入 loss）

    Args:
        ignore_index: 不計入 loss 的像素值（預設 999）
        bce_weight:   BCE 部分的加權（預設 0.5）
        dice_weight:  Dice 部分的加權（預設 0.5）
        pos_weight:   BCEWithLogitsLoss 的正樣本權重（None 表示不加權）
        use_scl_mask: True 時同時遮蔽 SCL∈{3,8,9}（雲影/中機率雲/高機率雲）的像素
        weak_label_value: 排除於 loss 之外的「弱相關」標籤值（預設 0.5）
    """

    def __init__(self, ignore_index: int = 999, bce_weight: float = 0.5,
                 dice_weight: float = 0.5, pos_weight: float = None,
                 use_scl_mask: bool = False, weak_label_value: float = 0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce_weight   = bce_weight
        self.dice_weight  = dice_weight
        self.use_scl_mask = use_scl_mask
        self.weak_label_value = weak_label_value
        pw = torch.tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                scl_mask: torch.Tensor = None) -> torch.Tensor:
        if y_true.dim() == y_pred.dim() - 1:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.float()

        mask = ((y_true != self.ignore_index) & (y_true != self.weak_label_value)).float()  # (B,1,P,P)

        if self.use_scl_mask and scl_mask is not None:
            mask = mask * scl_mask.unsqueeze(1)                # (B,P,P) → (B,1,P,P)

        masked_bce = (self.bce(y_pred, y_true) * mask).sum() / (mask.sum() + 1e-8)

        probs   = torch.sigmoid(y_pred)
        y_clean = y_true.clone()
        y_clean[mask == 0] = 0
        inter     = (probs * y_clean * mask).sum()
        dice_loss = 1 - (2 * inter) / ((probs * mask).sum() + (y_clean * mask).sum() + 1e-8)

        return self.bce_weight * masked_bce + self.dice_weight * dice_loss
