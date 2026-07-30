"""
eda.py

針對 data_cut.py 產生的 cutV1/{train,valid,test}_info.json
計算每張 patch 的灌水率與 SCL 雲量遮蔽率，並繪製分布直方圖。

json 格式（與 data_cut.py 輸出一致）：
    {"mode": ..., "patch_size": ..., "items": [{date, region, x_date,
     original_h/w, pad_h/w, padded_h/w, patches: [{idx, row, col, ...}]}]}
    train 的每個 patch 條目另含 water_ratio / scl_cloud_ratio / nodata_ratio（0~1）。
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_irrigation_cloud_cover_rate(
    info_json: str,
    dates: List[str] = None,
    kept_indices: Optional[dict] = None,
) -> pd.DataFrame:
    """
    計算每張 patch 的灌水率（%）與 SCL 遮蔽率（%）。

    直接讀取 {mode}_info.json 中切割時已算好的 water_ratio / scl_cloud_ratio
    欄位（僅 train 的 json 有比例欄位；valid/test 的 json 只有座標，
    對應 patch 會被略過）。

    Args:
        info_json:    cut_xy_patches() 輸出的 json 檔路徑（train_info.json 等）
        dates:        日期列表；None 表示全部
        kept_indices: OneSliceDataset.kept_indices（{date: {region: [idx, ...]}}）；
                      提供時只統計 Dataset 篩選／下採樣後保留的 patch

    Returns:
        pd.DataFrame，欄位：date, region, patch_name, water_ratio_%, scl_cloud_ratio_%
    """
    with open(info_json, encoding="utf-8") as f:
        info = json.load(f)

    dates_set = set(dates) if dates is not None else None
    records = []
    for item in info["items"]:
        date, region = item["date"], item["region"]
        if dates_set is not None and date not in dates_set:
            continue
        kept = None
        if kept_indices is not None:
            kept = set(kept_indices.get(date, {}).get(region, []))
        for meta in item["patches"]:
            if kept is not None and meta["idx"] not in kept:
                continue
            if "water_ratio" not in meta:
                continue  # valid/test 的 json 只有座標，無比例欄位
            wr = meta["water_ratio"]
            sr = meta.get("scl_cloud_ratio")
            records.append({
                "date": date,
                "region": region,
                "patch_name": f"patch_{meta['idx']:04d}.npy",
                "water_ratio_%": float(wr) * 100 if wr is not None else float("nan"),
                "scl_cloud_ratio_%": float(sr) * 100 if sr is not None else float("nan"),
            })

    return pd.DataFrame(records).round(2)


def compute_source_irrigation_cloud_rate(
    xy_dir: str,
    dates: List[str] = None,
    nodata_value: int = 999,
) -> pd.DataFrame:
    """
    以未切割 patch 的來源資料（xy對應/，每個日期底下 11 個區域）
    計算每個日期的灌水率（%）與 SCL 雲量遮蔽率（%），跨區域彙總。

    Args:
        xy_dir:       來源資料夾（xy對應/），含 y/ 和 X/ 子目錄
        dates:        要處理的日期列表（如 ["20230130", "20230219"]）；
                      None 表示自動掃描 y/ 下所有日期資料夾
        nodata_value: 無效像素值（預設 999）

    Returns:
        pd.DataFrame，每日期一行，欄位：date, water_ratio_%, scl_cloud_ratio_%
    """
    base_path = Path(xy_dir)
    y_dir = base_path / "y"
    x_dir = base_path / "X"

    date_dirs = sorted(p for p in y_dir.iterdir() if p.is_dir())
    if dates is not None:
        dates_set = set(dates)
        date_dirs = [p for p in date_dirs if p.name in dates_set]

    records = []
    for date_dir in date_dirs:
        date = date_dir.name

        water_hits, water_total = 0, 0
        cloud_hits, cloud_total = 0, 0

        for y_npy in sorted(date_dir.glob("*.npy")):
            region = y_npy.stem
            try:
                y = np.load(y_npy).astype(np.float32)  # (1, H, W)
                if y.ndim == 3:
                    y = y[0]  # -> (H, W)
            except Exception as e:
                print(f"[警告] 無法讀取 y/{date}/{region}.npy：{e}")
                continue

            valid_mask = y != nodata_value
            valid_y = y[valid_mask]
            water_hits  += int((valid_y == 1).sum())
            water_total += int(valid_y.size)

            x_path = x_dir / date / f"{region}.npy"
            if not x_path.exists():
                print(f"[警告] 找不到 X/{date}/{region}.npy，SCL 略過")
                continue
            try:
                x = np.load(x_path).astype(np.float32)  # (5, H, W)
                scl = x[4]                               # band index 4 = SCL
                valid_scl = scl[valid_mask]
                cloud_hits  += int(np.isin(valid_scl, [3, 8, 9]).sum())
                cloud_total += int(valid_scl.size)
            except Exception as e:
                print(f"[警告] 讀取 SCL 失敗（{date}/{region}）：{e}")

        water_ratio     = (water_hits / water_total * 100) if water_total > 0 else float("nan")
        scl_cloud_ratio = (cloud_hits / cloud_total * 100) if cloud_total > 0 else float("nan")

        records.append({
            "date": date,
            "water_ratio_%": round(water_ratio, 2),
            "scl_cloud_ratio_%": round(scl_cloud_ratio, 2),
        })

    return pd.DataFrame(records)


def count_zero_nodata_patches(
    info_json: str,
    dates: List[str] = None,
    kept_indices: Optional[dict] = None,
) -> pd.DataFrame:
    """
    統計「所有訓練資料整體」（不分日期／區塊）的 patch 概況。

    直接使用 {mode}_info.json 中切割時已算好的比例欄位判斷，不重新讀取影像：
        total：      patch 總數
        all_nodata： water_ratio 為 null（整張 patch 皆為 nodata）
        all_zero：   water_ratio == 0（有效像素中無灌水）
        have_one：   total - all_nodata - all_zero（含灌水像素）
        water_ratio_%_mean：    平均灌水率（%，排除全 nodata 的 null）
        scl_cloud_ratio_%_mean：平均遮蔽率（%，排除 null）
    僅 train 的 json 有比例欄位；沒有比例欄位的 patch（valid/test）會被略過。

    Args:
        info_json:    cut_xy_patches() 輸出的 json 檔路徑（train_info.json 等）
        dates:        要檢查的日期列表；None 表示全部
        kept_indices: OneSliceDataset.kept_indices（{date: {region: [idx, ...]}}）；
                      提供時只統計 Dataset 篩選／下採樣後保留的 patch，
                      None=統計 json 中全部 patch

    Returns:
        pd.DataFrame（單列），欄位：total, all_nodata, all_zero, have_one,
        water_ratio_%_mean, scl_cloud_ratio_%_mean
    """
    with open(info_json, encoding="utf-8") as f:
        info = json.load(f)
    dates_set = set(dates) if dates is not None else None

    total = all_nodata = all_zero = 0
    water_vals: List[float] = []
    scl_vals: List[float] = []
    for item in info["items"]:
        date, region = item["date"], item["region"]
        if dates_set is not None and date not in dates_set:
            continue

        # kept_indices 提供時，只統計 Dataset 篩選／下採樣後保留的 idx
        kept = None
        if kept_indices is not None:
            kept = set(kept_indices.get(date, {}).get(region, []))

        for meta in item["patches"]:
            if kept is not None and meta["idx"] not in kept:
                continue
            if "water_ratio" not in meta:
                continue  # valid/test 的 json 無比例欄位
            total += 1
            wr = meta["water_ratio"]
            sr = meta.get("scl_cloud_ratio")
            if wr is None:
                all_nodata += 1
            elif wr == 0:
                all_zero += 1
            if wr is not None:
                water_vals.append(wr)
            if sr is not None:
                scl_vals.append(sr)

    df = pd.DataFrame([{
        "total": total,
        "all_nodata": all_nodata,
        "all_zero": all_zero,
        "have_one": total - all_nodata - all_zero,
        "water_ratio_%_mean": round(float(np.mean(water_vals)) * 100, 2) if water_vals else float("nan"),
        "scl_cloud_ratio_%_mean": round(float(np.mean(scl_vals)) * 100, 2) if scl_vals else float("nan"),
    }])
    print(df.to_string(index=False))
    return df


def plot_irrigation_cloud_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    繪製「所有訓練資料整體」（不分日期／區塊）的灌水率與 SCL 遮蔽率
    頻率分布直方圖（共兩張，左：灌水率、右：遮蔽率）。

    Args:
        df:        compute_irrigation_cloud_cover_rate() 的回傳值
        save_path: 圖片儲存完整路徑（含副檔名，例如 "output/cloud_dist.png"）；
                   None 時僅顯示，不儲存
    """
    if df.empty:
        print("DataFrame 為空。")
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    specs = [
        ("water_ratio_%",     "blue",   "Water Ratio Distribution",     np.arange(0, 101, 5)),
        ("scl_cloud_ratio_%", "orange", "SCL Cloud Ratio Distribution", np.arange(0, 101, 10)),
    ]
    for ax, (col, color, title, ticks) in zip(axes, specs):
        clean = df[col].dropna()
        if not clean.empty:
            ax.hist(clean, bins=np.arange(0, 101, 5), color=color, alpha=0.7, edgecolor="black")
            ax.set_title(f"{title} (all train patches, n={len(clean)})")
            ax.set_ylabel("Count")
            ax.text(
                0.98, 0.97,
                f"mean={clean.mean():.2f}\nmin={clean.min():.2f}\nmax={clean.max():.2f}",
                transform=ax.transAxes,
                ha="right", va="top",
                multialignment="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )
        else:
            ax.set_title(f"{title} (No Data)")
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_xticks(ticks)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[已儲存] {save_path}")
    plt.show()


if __name__ == "__main__":
    from src.visualization.myplot import plot_patch
    XY_DIR = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"

    # ── 全圖（原始來源）每日統計 ──────────────────────────────────────────────
    source_df = compute_source_irrigation_cloud_rate(
        xy_dir=XY_DIR,
    )
    print("── 全圖每日統計 ──")
    print(source_df.to_string(index=False))

    # ── 統計全 0 / 全 nodata 的 patch 張數 ───────────────────────────────────
    TRAIN_INFO = str(Path(XY_DIR).parent / "cutV1" / "train_info.json")
    print("\n── 訓練資料整體統計（不分日期／區塊）──")
    count_zero_nodata_patches(info_json=TRAIN_INFO)

    # ── Patch 層級分布圖（train_info.json，整體） ────────────────────────────
    TRAIN_DATES = ["20230130", "20230219", "20230301"]

    df = compute_irrigation_cloud_cover_rate(
        info_json=TRAIN_INFO,
        dates=TRAIN_DATES,
    )
    plot_irrigation_cloud_distribution(df)
