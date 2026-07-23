"""
eda.py

針對 data_cut.py 產生的 training_data / testing_data / valid_data 資料夾
計算每張 patch 的灌水率與 SCL 雲量遮蔽率，並繪製分布直方圖。

目錄格式（與 data_cut.py 輸出一致，每個日期底下有 11 個區域）：
    data_dir/
        y/{date}/{region}/patch_*.npy          shape (P, P)，值 0/0.5/1/999
        X/{date}/{region}/patch_*.npy          shape (5, P, P)，band 順序：
                                                [0] Green, [1] Red, [2] NIR, [3] SWIR, [4] SCL
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_irrigation_cloud_cover_rate(
    data_dir: str,
    dates: List[str],
    nodata_value: int = 999,
) -> pd.DataFrame:
    """
    計算每張 patch 的灌水率（%）與 SCL 遮蔽率（%）。

    Args:
        data_dir:     資料夾路徑（training_data/、testing_data/、valid_data/ 其中之一）
        dates:        日期列表，需與資料夾名稱一致（如 ["20230130", "20230219"]）
        nodata_value: 無效像素值（預設 999）

    Returns:
        pd.DataFrame，欄位：date, patch_name, water_ratio_%, scl_cloud_ratio_%
    """
    base_path = Path(data_dir)
    y_dir = base_path / "y"
    x_dir = base_path / "X"

    records = []
    for date in dates:
        y_date_dir = y_dir / date
        if not y_date_dir.exists():
            print(f"[警告] 找不到 y/{date}，跳過")
            continue

        region_dirs = sorted(p for p in y_date_dir.iterdir() if p.is_dir())
        for region_dir in region_dirs:
            region = region_dir.name
            for y_path in sorted(region_dir.glob("patch_*.npy")):
                patch_name = y_path.name
                x_path = x_dir / date / region / patch_name
                if not x_path.exists():
                    continue

                # ── Y（ground truth）──────────────────────────────────────────────
                try:
                    y_data = np.load(y_path).astype(np.float32)
                    if y_data.ndim == 3:
                        y_data = y_data[0]
                    valid_mask = y_data != nodata_value
                    valid_y = y_data[valid_mask]
                    if valid_y.size > 0:
                        water_ratio = float((valid_y == 1).mean() * 100)
                    else:
                        water_ratio = float("nan")
                except Exception:
                    continue

                # ── SCL（X patch 的第 5 個 band，index 4）────────────────────────
                try:
                    x_data = np.load(x_path).astype(np.float32)  # (5, P, P)
                    scl = x_data[4]                               # SCL band
                    valid_scl = scl[valid_mask]
                    if valid_scl.size > 0:
                        is_cloud = np.isin(valid_scl, [3, 8, 9])
                        scl_cloud_ratio = float(is_cloud.mean() * 100)
                    else:
                        scl_cloud_ratio = float("nan")
                except Exception:
                    scl_cloud_ratio = float("nan")

                records.append({
                    "date": date,
                    "region": region,
                    "patch_name": patch_name,
                    "water_ratio_%": water_ratio,
                    "scl_cloud_ratio_%": scl_cloud_ratio,
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
    data_dir: str,
    dates: List[str] = None,
    nodata_value: int = 999,
) -> pd.DataFrame:
    """
    統計各日期中，y patch 整張全為 nodata_value、或整張只含 0 與 nodata_value 的張數。

    Args:
        data_dir:     資料夾路徑（training_data/ 等）
        dates:        要檢查的日期列表；None 表示自動掃描 y/ 下所有日期子資料夾
        nodata_value: 無效像素值（預設 999）

    Returns:
        pd.DataFrame，欄位：date, total, all_nodata, all_zero, have_one
        （all_nodata：整張 patch 都是 nodata_value；
          all_zero：非 nodata 的有效像素全為 0（純背景）；
          have_one = total - all_nodata - all_zero）
    """
    y_dir = Path(data_dir) / "y"

    date_dirs = sorted(y_dir.iterdir()) if dates is None else [y_dir / d for d in dates]

    records = []
    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        patches = [
            p
            for region_dir in sorted(date_dir.iterdir())
            if region_dir.is_dir()
            for p in sorted(region_dir.glob("patch_*.npy"))
        ]
        total = len(patches)
        all_nodata = 0
        all_zero = 0
        for p in patches:
            arr = np.load(p)
            valid = arr[arr != nodata_value]
            if valid.size == 0:
                all_nodata += 1
            elif np.all(valid == 0):
                all_zero += 1
        records.append({
            "date": date,
            "total": total,
            "all_nodata": all_nodata,
            "all_zero": all_zero,
            "have_one": total - all_nodata - all_zero,
        })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    return df


def plot_irrigation_cloud_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    繪製每個日期的灌水率與 SCL 遮蔽率頻率分布直方圖。

    Args:
        df:        compute_irrigation_cloud_cover_rate() 的回傳值
        save_path: 圖片儲存完整路徑（含副檔名，例如 "output/cloud_dist.png"）；
                   None 時僅顯示，不儲存
    """
    if df.empty:
        print("DataFrame 為空。")
        return

    dates = sorted(df["date"].unique())
    n = len(dates)

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    water_ticks = np.arange(0, 101, 5)
    scl_ticks   = np.arange(0, 101, 10)

    for i, date in enumerate(dates):
        date_df = df[df["date"] == date]
        plots = [
            (date_df["water_ratio_%"],     "blue",   "Water Ratio Distribution",     water_ticks),
            (date_df["scl_cloud_ratio_%"], "orange", "SCL Cloud Ratio Distribution", scl_ticks),
        ]
        for j, (data, color, title, ticks) in enumerate(plots):
            ax = axes[i][j] if n > 1 else axes[j]
            clean = data.dropna()
            if not clean.empty:
                ax.hist(clean, bins=np.arange(0, 101, 5), color=color, alpha=0.7, edgecolor="black")
                ax.set_title(f"{date} - {title}")
                ax.set_ylabel("Count")
                ax.text(
                    0.98, 0.97,
                    f"min={clean.min():.2f}\nmax={clean.max():.2f}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    multialignment="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )
            else:
                ax.set_title(f"{date} - {title} (No Data)")
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
    from myplot import plot_patch
    XY_DIR = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"

    # ── 全圖（原始來源）每日統計 ──────────────────────────────────────────────
    source_df = compute_source_irrigation_cloud_rate(
        xy_dir=XY_DIR,
    )
    print("── 全圖每日統計 ──")
    print(source_df.to_string(index=False))

    # ── 統計全 0 / 全 nodata 的 patch 張數 ───────────────────────────────────
    print("\n── 全 0 / 全 nodata patch 統計 ──")
    count_zero_nodata_patches(data_dir="training_data")

    # ── Patch 層級統計（training_data） ──────────────────────────────────────
    TRAIN_DATES = ["20230130", "20230219", "20230301"]

    df = compute_irrigation_cloud_cover_rate(
        data_dir="training_data",
        dates=TRAIN_DATES,
    )

    print("\n── Patch 每日平均 ──")
    print(df.groupby("date")[["water_ratio_%", "scl_cloud_ratio_%"]].mean().round(2))

    plot_irrigation_cloud_distribution(df)
