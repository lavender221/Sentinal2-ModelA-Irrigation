"""
myplot.py

patch 視覺化工具。
"""

import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.font_manager import FontProperties



def _find_chinese_font() -> FontProperties:
    """嘗試找到 Windows 中文字型檔案，找不到則回傳 None。"""
    candidates = [
        r"C:\Windows\Fonts\msjh.ttc",     # 微軟正黑體（繁體）
        r"C:\Windows\Fonts\msjhbd.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",   # 新細明體
        r"C:\Windows\Fonts\kaiu.ttf",      # 標楷體
        r"C:\Windows\Fonts\msyh.ttc",      # 微軟雅黑（簡體）
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return FontProperties(fname=path)
    return None


_CHINESE_FP = _find_chinese_font()


def plot_patch(npy_path: str, nodata_value: int = 999) -> None:
    """
    顯示一張 y patch（shape P×P，值 0/1/nodata）。

    顏色對應：
        白色  → 0（無灌水，有效區域）
        藍色  → 1（灌水）
        灰色  → nodata（無效區域）

    colorbar 同時顯示各類別的像素數量。

    Args:
        npy_path:     .npy 檔案路徑
        nodata_value: 無效像素值（預設 999）
    """
    arr = np.load(npy_path).astype(np.float32)

    n_nodata = int(np.sum(arr == nodata_value))
    n_water  = int(np.sum(arr == 1))
    n_bg     = int(np.sum(arr == 0))

    # 將三種值映射到 0/1/2 供 imshow 使用
    display = np.where(arr == nodata_value, 2.0, arr)

    cmap = mcolors.ListedColormap(["white", "steelblue", "lightgray"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)

    if _CHINESE_FP:
        cbar.ax.set_yticklabels(
            [f"0 無灌水 ({n_bg})", f"1 灌水 ({n_water})", f"{nodata_value} nodata ({n_nodata})"],
            fontproperties=_CHINESE_FP,
        )
        ax.set_title(Path(npy_path).name, fontproperties=_CHINESE_FP)
    else:
        cbar.ax.set_yticklabels(
            [f"0 no-water ({n_bg})", f"1 water ({n_water})", f"{nodata_value} nodata ({n_nodata})"]
        )
        ax.set_title(Path(npy_path).name)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, output_dir: str = None) -> None:
    """
    繪製訓練過程的 Loss / Accuracy / F1-score 折線圖（Train vs Valid）。

    Args:
        history:    training() 的回傳值，含 train_loss / val_loss / train_acc /
                    val_acc / train_f1 / val_f1 各 epoch 列表
        output_dir: 若提供，圖片另存為 {output_dir}/training_curves.png
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc",  "val_acc",  "Accuracy"),
        ("train_f1",   "val_f1",   "F1-score"),
    ]
    for ax, (tk, vk, title) in zip(axes, specs):
        ax.plot(epochs_range, history[tk], label="Train", marker="o", color="tab:blue")
        ax.plot(epochs_range, history[vk], label="Valid", marker="o", color="tab:orange")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if output_dir:
        import os
        path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  訓練曲線已存至 {path}")
    plt.show()

def plot_confusion_map(
    pred_path: str,
    gt_path: str,
    ignore_index: int = 999,
    save_path: str = None,
    title: str = None,
) -> dict:
    """
    比對預測結果與 ground truth，畫出每個 pixel 的 TP/TN/FP/FN 狀態圖。

    pred_path / gt_path 可傳入 .npy 檔路徑或 np.ndarray（即時推論結果直接傳陣列）。

    顏色規則：
        TP  (pred=1, gt=1)  → 綠色  #2ca02c
        TN  (pred=0, gt=0)  → 灰色  #aec7e8
        FP  (pred=1, gt=0)  → 藍色  #1f77b4
        FN  (pred=0, gt=1)  → 紅色  #d62728
        忽略 (pred or gt == ignore_index) → 黑色

    Returns:
        dict 含 tp / tn / fp / fn / total_valid pixel 數
    """
    pred = pred_path if isinstance(pred_path, np.ndarray) else np.load(pred_path)
    gt   = gt_path   if isinstance(gt_path,   np.ndarray) else np.load(gt_path)

    assert pred.shape == gt.shape, (
        f"shape mismatch: pred={pred.shape}, gt={gt.shape}"
    )

    # ── 建立 RGB 畫布 ──
    H, W = pred.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    mask_ignore = (pred == ignore_index) | (gt == ignore_index)
    mask_tp = (~mask_ignore) & (pred == 1) & (gt == 1)
    mask_tn = (~mask_ignore) & (pred == 0) & (gt == 0)
    mask_fp = (~mask_ignore) & (pred == 1) & (gt == 0)
    mask_fn = (~mask_ignore) & (pred == 0) & (gt == 1)

    # 黑色 (ignore)  → 預設已是 0
    rgb[mask_tp] = [0.173, 0.627, 0.173]   # 綠
    rgb[mask_tn] = [0.682, 0.780, 0.910]   # 灰藍
    rgb[mask_fp] = [0.122, 0.467, 0.706]   # 藍
    rgb[mask_fn] = [0.839, 0.153, 0.157]   # 紅

    tp = int(mask_tp.sum())
    tn = int(mask_tn.sum())
    fp = int(mask_fp.sum())
    fn = int(mask_fn.sum())
    total_valid = tp + tn + fp + fn

    accuracy  = (tp + tn) / total_valid if total_valid > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    counts = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "ignore": int(mask_ignore.sum()),
        "total_valid": total_valid,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "iou": iou,
    }

    # ── 畫圖 ──
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(
        title,
        fontsize=11,
    )
    ax.axis('off')

    # 右下角：指標文字框
    metrics_text = (
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1       : {f1:.4f}\n"
        f"IoU      : {iou:.4f}"
    )
    ax.text(
        0.99, 0.01, metrics_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
    )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'圖片已儲存：{save_path}')
    plt.show()
    return counts

if __name__ == "__main__":
    plot_patch(
        npy_path=r"C:\Users\cathyhsu\Desktop\Geospatial predictive modeling\2023_1&2\valid_data\y\20230306\E1\result.npy"
    )
