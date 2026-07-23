"""
TrainingFuncs.py

訓練、驗證與評估函式。
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data_cut import reconstruct_y_patches
from myplot import plot_training_history

try:
    from tensorboardX import SummaryWriter as _SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    except (ImportError, SystemError):
        _SummaryWriter = None


# ── 混淆矩陣與指標 ────────────────────────────────────────────────────────────

def _update_cm(cm: np.ndarray, y_pred: torch.Tensor, y_true: torch.Tensor,
               threshold: float = 0.5, ignore_index: int = 999,
               weak_label_value: float = 0.5) -> np.ndarray:
    """將一個 batch 的預測結果累加進混淆矩陣。
    排除 ignore_index（nodata）與 weak_label_value（弱相關灌水標籤）像素，
    與 MaskedDiceBCELoss 的排除規則一致。
    """
    with torch.no_grad():
        probs = torch.sigmoid(y_pred.detach())
        if probs.dim() == y_true.dim() + 1:
            probs = probs.squeeze(1)
        preds = (probs >= threshold).long()
        valid = (y_true != ignore_index) & (y_true != weak_label_value)
        p = preds[valid].cpu().numpy().ravel()
        t = y_true[valid].long().cpu().numpy().ravel()
    if t.size > 0:
        cm += confusion_matrix(t, p, labels=[0, 1])
    return cm


def _metrics(cm: np.ndarray) -> Dict:
    """由混淆矩陣 [[TN, FP], [FN, TP]] 計算 accuracy / precision / recall / f1 / iou。"""
    tn, fp, fn, tp = cm.ravel()
    total     = tn + fp + fn + tp
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou}


# ── 單 Epoch 訓練 / 驗證 ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    device, ignore_index: int = 999) -> Tuple[float, Dict]:
    """跑完一個 epoch 的訓練，回傳 (平均 loss, 指標 dict)。"""
    model.train()
    total_loss, n = 0.0, 0
    cm = np.zeros((2, 2), dtype=np.int64)
    for x, y, scl_mask in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        scl_mask = scl_mask.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y, scl_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n          += x.size(0)
        cm = _update_cm(cm, pred, y, ignore_index=ignore_index)
    return total_loss / max(n, 1), _metrics(cm)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion,
                       device, ignore_index: int = 999) -> Tuple[float, Dict]:
    """跑完一個 epoch 的驗證，回傳 (平均 loss, 指標 dict)。"""
    model.eval()
    total_loss, n = 0.0, 0
    cm = np.zeros((2, 2), dtype=np.int64)
    for x, y, _ in tqdm(loader, desc="Valid", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        n          += x.size(0)
        cm = _update_cm(cm, pred, y, ignore_index=ignore_index)
    return total_loss / max(n, 1), _metrics(cm)


# ── 主訓練迴圈 ────────────────────────────────────────────────────────────────

def training(model, train_loader, val_loader, criterion,
             optimizer, config: Dict, device: str,
             writer=None) -> Dict:
    """
    主訓練迴圈，含 early stopping 與 checkpoint。

    Args:
        model, train_loader, val_loader, criterion, optimizer: 模型與訓練元件
        config: CONFIG dict，需含 output_dir / epochs / patience / nodata_value
        device: 運算裝置字串
        writer: SummaryWriter 實例（可選）；提供時每 epoch 寫入 TensorBoard 曲線

    Returns:
        history dict，含各 epoch 指標與最佳模型路徑
    """
    output_dir    = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    best_path     = os.path.join(output_dir, "best_model.pt")
    ignore_index  = config.get("nodata_value", 999)
    best_val_loss = float("inf")
    no_improve    = 0
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1"]}

    for epoch in range(1, config["epochs"] + 1):
        tl, tm = train_one_epoch(model, train_loader, optimizer, criterion, device, ignore_index)
        vl, vm = validate_one_epoch(model, val_loader, criterion, device, ignore_index)

        for key, val in [("train_loss", tl), ("val_loss", vl),
                         ("train_acc", tm["accuracy"]), ("val_acc", vm["accuracy"]),
                         ("train_f1",  tm["f1"]),       ("val_f1",  vm["f1"])]:
            history[key].append(val)

        if writer is not None:
            writer.add_scalars("Loss",     {"train": tl,             "valid": vl},             epoch)
            writer.add_scalars("Accuracy", {"train": tm["accuracy"], "valid": vm["accuracy"]}, epoch)
            writer.add_scalars("F1",       {"train": tm["f1"],       "valid": vm["f1"]},       epoch)

        print(f"[Epoch {epoch:03d}/{config['epochs']}] "
              f"loss {tl:.4f}/{vl:.4f}  f1 {tm['f1']:.4f}/{vm['f1']:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            no_improve    = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vl, "config": config}, best_path)
            print(f"  ✓ 已儲存最佳模型（val_loss={vl:.4f}）")
        else:
            no_improve += 1
            print(f"  ⚠ 未改善 ({no_improve}/{config['patience']})")
            if no_improve >= config["patience"]:
                print("  Early stopping")
                break

    history["best_val_loss"]   = best_val_loss
    history["best_model_path"] = best_path
    plot_training_history(history, output_dir=output_dir)
    return history


# ── 評估 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluating_all(
    model,
    loader,
    device,
    threshold: float = 0.5,
    ignore_index: int = 999,
    dataset=None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    評估方式一：統計所有 pixel 的預測結果（合併計算）。

    若同時傳入 dataset 與 save_dir，則額外：
      1. 將每個 patch 的 0/1 預測（nodata 位置還原為 ignore_index）
         存至 save_dir/test_pred/{date}/patch_*.npy
      2. 複製 patch_info.json 後呼叫 reconstruct_y_patches，
         在各日期資料夾下生成 result.npy（完整圖重建）

    注意：loader 必須 shuffle=False，num_workers 建議 0，
          以確保 batch 順序與 dataset.samples 一致。

    Returns:
        dict，含 accuracy / precision / recall / f1 / iou / confusion_matrix
    """
    model.eval()
    cm = np.zeros((2, 2), dtype=np.int64)

    save_preds = (dataset is not None) and (save_dir is not None)
    pred_root  = Path(save_dir) / "test_pred" if save_preds else None
    sample_idx = 0

    for x, y, _ in tqdm(loader, desc="Eval (all)", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        cm = _update_cm(cm, logits, y, threshold, ignore_index)

        if save_preds:
            probs = torch.sigmoid(logits.detach())
            if probs.dim() == y.dim() + 1:
                probs = probs.squeeze(1)
            preds = (probs >= threshold).long()          # (B, H, W)  值 0 / 1

            for i in range(x.size(0)):
                s       = dataset.samples[sample_idx]
                pred_np = preds[i].cpu().numpy().astype(np.int16)   # (H, W)
                y_np    = y[i].cpu().numpy()                         # (H, W)

                # nodata 位置（y==999）還原為 ignore_index
                pred_np[y_np == ignore_index] = ignore_index

                out_dir = pred_root / s["date"] / s["region"]
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / s["patch_name"], pred_np)
                sample_idx += 1

    if cm.sum() == 0:
        print("[警告] 沒有有效像素可評估。")
        return {}

    m = _metrics(cm)
    m["confusion_matrix"] = cm.tolist()

    # 各 (日期, 區域)：複製 patch_info.json → 逐區域重建完整圖
    if save_preds:
        pairs_done = sorted({(s["date"], s["region"]) for s in dataset.samples})
        print("\n── 重建完整預測圖（逐區域）──")
        for date, region in pairs_done:
            src_info = dataset.data_dir / "y" / date / region / "patch_info.json"
            dst_info = pred_root / date / region / "patch_info.json"
            if src_info.exists():
                dst_info.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_info, dst_info)
                reconstruct_y_patches(
                    date_dir=str(pred_root / date / region),
                    nodata_value=float(ignore_index),
                    output_name="result.npy",
                )
            else:
                print(f"  [警告] 找不到 {src_info.name}，跳過 {date}/{region} 合併")

    return m
