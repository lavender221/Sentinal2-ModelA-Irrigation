"""
Preprocess.py

自動掃描 xy_dir/y/ 下的所有日期，依 train_ratio / val_ratio 按時間順序切分，
再執行 data_cut.py 的切割流程；valid 與 test 完成後額外執行 reconstruct_y_patches()
將 patch 還原成完整圖像（result.npy）。

資料來源固定為切割成 11 個區域後的 "xy對應" 資料夾：
    xy_dir/y/{date}/{region}.npy        shape (h, w)
    xy_dir/X/{date}/{region}.npy   shape (5, h, w)
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# band 名稱 → X array 的 channel index（shape: (5, H, W)）
_BAND_INDEX = {"Green": 0, "Red": 1, "NIR": 2, "SWIR": 3}


# ── 日期切分 ──────────────────────────────────────────────────────────────────

def train_valid_test_split(
    base_dir: str,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    從 base_dir/y/ 掃描所有日期資料夾，依時間順序切分為 train / valid / test。

    Args:
        base_dir:    來源資料夾（含 y/ 子目錄，y/ 下每個子資料夾名稱即為日期）
        train_ratio: 訓練集比例（0 < train_ratio < 1）
        val_ratio:   驗證集比例（0 <= val_ratio < 1，且 train_ratio + val_ratio < 1）

    Returns:
        (train_dates, val_dates, test_dates)
    """
    y_dir = Path(base_dir) / "y"
    if not y_dir.exists():
        raise FileNotFoundError(f"找不到 y/ 資料夾：{y_dir}")

    dates = sorted(p.name for p in y_dir.iterdir() if p.is_dir())
    if not dates:
        raise FileNotFoundError(f"在 {y_dir} 中找不到任何日期資料夾。")

    total = len(dates)

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必須落在有效範圍內，且 train_ratio + val_ratio < 1")
    if total < 3:
        raise ValueError(f"日期數量 ({total}) < 3，無法切分為 train/valid/test")

    n_train = max(1, round(total * train_ratio))
    n_val   = max(1, round(total * val_ratio))

    if n_train + n_val >= total:
        n_val = max(1, total - n_train - 1)
    if n_train + n_val >= total:
        n_train = max(1, total - n_val - 1)

    n_test = total - n_train - n_val
    if n_test < 1:
        raise ValueError("依照給定比例無法產生測試集，請調整 train_ratio 或 val_ratio")

    train_dates = dates[:n_train]
    val_dates   = dates[n_train:n_train + n_val]
    test_dates  = dates[n_train + n_val:]
    return train_dates, val_dates, test_dates


def train_valid_test_split_by_year(
    base_dir: str,
    train_ratio: float,
    seed: int = 42,
    train_valid_years: List[str] = ("2023",),
    test_years: List[str] = ("2026",),
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    從 base_dir/y/ 掃描所有日期資料夾，依日期字首年份分組：
        train_valid_years（預設 ("2023",)）開頭的日期 → 用固定 seed 隨機打亂後依 train_ratio 切成 train / valid
        test_years（預設 ("2026",)）開頭的日期 → 全部作為 test

    並展開每組日期資料夾下所有區域的 .npy 檔名（格式 "{date}/{region}.npy"）。

    Args:
        base_dir:           來源資料夾（含 y/ 子目錄，y/ 下每個子資料夾名稱即為日期）
        train_ratio:        train 佔 train_valid_years 日期數的比例（0 < train_ratio < 1），
                             其餘日期歸為 valid
        seed:                打亂 train_valid_years 日期順序用的隨機種子
        train_valid_years:  train/valid 日期的年份字首列表
        test_years:         test 日期的年份字首列表

    Returns:
        (train_dates, val_dates, test_dates, train_files, val_files, test_files)
    """
    y_dir = Path(base_dir) / "y"
    if not y_dir.exists():
        raise FileNotFoundError(f"找不到 y/ 資料夾：{y_dir}")

    all_dates = sorted(p.name for p in y_dir.iterdir() if p.is_dir())

    train_valid_dates = [d for d in all_dates if d.startswith(tuple(train_valid_years))]
    test_dates = [d for d in all_dates if d.startswith(tuple(test_years))]

    if not train_valid_dates:
        raise ValueError(f"找不到任何 {train_valid_years} 開頭的日期資料夾")
    if not test_dates:
        raise ValueError(f"找不到任何 {test_years} 開頭的日期資料夾")

    shuffled = train_valid_dates.copy()
    random.Random(seed).shuffle(shuffled)

    n_train = round(len(shuffled) * train_ratio)
    train_dates = sorted(shuffled[:n_train])
    val_dates   = sorted(shuffled[n_train:])
    test_dates  = sorted(test_dates)

    def _expand(dates: List[str]) -> List[str]:
        files = []
        for date in dates:
            for npy_path in sorted((y_dir / date).glob("*.npy")):
                files.append(f"{date}/{npy_path.name}")
        return files

    train_files = _expand(train_dates)
    val_files   = _expand(val_dates)
    test_files  = _expand(test_dates)

    return train_dates, val_dates, test_dates, train_files, val_files, test_files


REGION_NAMES = ["N1", "N2", "EN", "E1", "E2", "E3", "W1", "W2", "W3", "W4", "W5"]


def split_regions_train_valid_test(
    base_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    regions: List[str] = REGION_NAMES,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    隨機打亂 11 個區塊（N1、N2、EN、E1、E2、E3、W1、W2、W3、W4、W5），
    依 train_ratio / val_ratio 切成 train / valid / test 三組區塊（預設 6:2:2）。

    區塊分組在所有 date 間共用（同一個區塊的所有日期都歸在同一組），
    避免同一區塊同時出現在不同 split 造成空間上的資料洩漏。

    再展開成 base_dir/y/ 下「所有日期」底下對應區塊的檔名（格式 "{date}/{region}.npy"）。
    例如 N1 被分到 train，則 train_files 會包含 y/ 下所有日期的 "{日期}/N1.npy"。

    Args:
        base_dir:     來源資料夾（含 y/ 子目錄，y/ 下每個子資料夾名稱即為日期）
        train_ratio:  train 佔區塊數的比例（預設 0.6）
        val_ratio:    valid 佔區塊數的比例（預設 0.2），其餘歸為 test
        seed:         打亂區塊順序用的隨機種子
        regions:      區塊名稱列表（預設為 11 個固定區塊）

    Returns:
        (train_regions, val_regions, test_regions, train_files, val_files, test_files)
    """
    y_dir = Path(base_dir) / "y"
    if not y_dir.exists():
        raise FileNotFoundError(f"找不到 y/ 資料夾：{y_dir}")

    dates = sorted(p.name for p in y_dir.iterdir() if p.is_dir())
    if not dates:
        raise FileNotFoundError(f"在 {y_dir} 中找不到任何日期資料夾。")

    total = len(regions)

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必須落在有效範圍內，且 train_ratio + val_ratio < 1")

    shuffled = list(regions)
    random.Random(seed).shuffle(shuffled)

    n_train = max(1, round(total * train_ratio))
    n_val   = max(1, round(total * val_ratio))
    if n_train + n_val >= total:
        n_val = max(1, total - n_train - 1)
    if n_train + n_val >= total:
        n_train = max(1, total - n_val - 1)

    train_regions = shuffled[:n_train]
    val_regions   = shuffled[n_train:n_train + n_val]
    test_regions  = shuffled[n_train + n_val:]

    def _expand(region_group: List[str]) -> List[str]:
        files = []
        for date in dates:
            for region in region_group:
                if (y_dir / date / f"{region}.npy").exists():
                    files.append(f"{date}/{region}.npy")
        return files

    train_files = _expand(train_regions)
    val_files   = _expand(val_regions)
    test_files  = _expand(test_regions)

    return train_regions, val_regions, test_regions, train_files, val_files, test_files


# ── 標準化參數計算 ────────────────────────────────────────────────────────────

def compute_band_percentiles(
    base_dir: str,
    dates: List[str],
    numeric_bands: List[str],
    regions: Optional[List[str]] = None,
    lower: float = 1.0,
    upper: float = 99.0,
    nodata_value: int = 999,
    hist_max: int = 65535,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    只用 train_dates（與 Case2 空間切分時的 train_regions）計算各 numeric band
    的 lower~upper 百分位數，作為離群值裁切邊界（模型規劃.md.md Step 2：以訓練集
    統計量裁切，避免驗證/測試集的統計資訊洩漏到訓練流程中）。

    以直方圖（bin 寬度 = 1，範圍 0..hist_max）累加各波段有效像素
    （排除 nodata）的次數分布，再由累積分布反推百分位數，避免把所有
    像素值一次串接進記憶體。

    資料來源：
        base_dir/y/{date}/{region}.npy      shape (1, H, W)，用於排除 nodata 像素
        base_dir/X/{date}/{region}.npy      shape (5, H, W)

    重要：只能傳入 train_dates（時間切分）或 train_regions（空間切分）計算，
    絕不可使用 val/test 日期或區塊，否則會有資訊洩漏。

    Args:
        base_dir:      切割後的來源資料夾（xy對應/）
        dates:         訓練日期列表（train_dates）
        numeric_bands: 要計算的波段名稱列表（如 ["Green", "Red", "NIR", "SWIR"]）
        regions:       訓練區塊列表（空間切分 Case2 用；None 表示不篩選，使用該日期下
                       全部區塊 —— 時間切分 Case1 應保持 None）
        lower:         下百分位數（預設 1）
        upper:         上百分位數（預設 99）
        nodata_value:  無效像素值，排除於統計之外（預設 999）
        hist_max:      直方圖涵蓋的最大像素值（預設 65535，對應 uint16 上限）

    Returns:
        (clip_min, clip_max)，各為 shape (len(numeric_bands),) 的 float32 ndarray
    """
    base_path = Path(base_dir)
    n_bands = len(numeric_bands)
    hist = np.zeros((n_bands, hist_max + 1), dtype=np.int64)
    regions_set = set(regions) if regions is not None else None

    for date in dates:
        x_dir = base_path / "X" / date
        y_dir = base_path / "y" / date

        if not x_dir.exists() or not y_dir.exists():
            print(f"[警告] 找不到 {date} 的 X 或 y 資料夾，跳過")
            continue

        for x_path in sorted(x_dir.glob("*.npy")):
            region = x_path.stem
            if regions_set is not None and region not in regions_set:
                continue
            y_path = y_dir / f"{region}.npy"
            if not y_path.exists():
                print(f"[警告] 找不到 {date}/{region} 對應的 y 檔案，跳過")
                continue

            y = np.load(y_path).astype(np.float32)
            if y.ndim == 3:
                y = y[0]  # (1, H, W) -> (H, W)
            x = np.load(x_path)  # (5, H, W)
            valid_mask = y != nodata_value

            for i, band in enumerate(numeric_bands):
                ch = _BAND_INDEX.get(band, i)
                pixels = x[ch][valid_mask]
                pixels = pixels[(pixels >= 0) & (pixels <= hist_max)]
                if pixels.size == 0:
                    continue
                hist[i] += np.bincount(pixels.astype(np.int64), minlength=hist_max + 1)[: hist_max + 1]

    values   = np.arange(hist_max + 1)
    clip_min = np.zeros(n_bands, dtype=np.float32)
    clip_max = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        total = hist[i].sum()
        if total == 0:
            print(f"[警告] {numeric_bands[i]} 沒有任何有效像素，clip 邊界退回 (0, 10000)")
            clip_min[i], clip_max[i] = 0.0, 10000.0
            continue
        cdf = np.cumsum(hist[i]) / total * 100.0
        clip_min[i] = float(values[np.searchsorted(cdf, lower)])
        clip_max[i] = float(values[np.searchsorted(cdf, upper)])

    return clip_min, clip_max


def compute_band_mean_std(
    base_dir: str,
    dates: List[str],
    numeric_bands: List[str],
    regions: Optional[List[str]] = None,
    nodata_value: int = 999,
    clip_min: Optional[np.ndarray] = None,
    clip_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算各 numeric band 的 mean / std，供 Z-score 標準化使用。
    來源為切割成 11 個區域後的 "xy對應" 資料夾（每個日期底下有多個區域 .npy，
    如 N1.npy, N2.npy, ...），逐一讀取該日期下所有區域檔案累計統計量。

    重要：只能傳入 train_dates（時間切分）或 train_regions（空間切分）計算，
    絕不可使用 val/test 日期或區塊，否則驗證/測試集的統計資訊會洩漏到訓練流程中。

    資料來源：
        base_dir/y/{date}/{region}.npy      shape (1, H, W)，用於排除 nodata 像素
        base_dir/X/{date}/{region}.npy      shape (5, H, W)
            channel 順序：[0]=Green, [1]=Red, [2]=NIR, [3]=SWIR, [4]=SCL（未列入 numeric_bands 則不計算）

    Args:
        base_dir:      切割後的來源資料夾（xy對應/）
        dates:         訓練日期列表（train_dates）
        numeric_bands: 要計算的波段名稱列表（如 ["Green", "Red", "NIR", "SWIR"]）
        regions:       訓練區塊列表（空間切分 Case2 用；None 表示不篩選，時間切分 Case1
                       應保持 None）
        nodata_value:  無效像素值，排除於統計之外（預設 999）
        clip_min:      各波段裁切下界（shape (len(numeric_bands),)）；None 時退回 0
                       （建議傳入 compute_band_percentiles() 的回傳值，做 1st~99th 百分位裁切）
        clip_max:      各波段裁切上界；None 時退回 10000

    Returns:
        (mean, std)，各為 shape (len(numeric_bands),) 的 float32 ndarray
    """
    base_path = Path(base_dir)
    n_bands = len(numeric_bands)
    sums    = np.zeros(n_bands, dtype=np.float64)
    sums_sq = np.zeros(n_bands, dtype=np.float64)
    counts  = np.zeros(n_bands, dtype=np.float64)

    lo = clip_min if clip_min is not None else np.zeros(n_bands, dtype=np.float32)
    hi = clip_max if clip_max is not None else np.full(n_bands, 10000.0, dtype=np.float32)
    regions_set = set(regions) if regions is not None else None

    for date in dates:
        x_dir = base_path / "X" / date
        y_dir = base_path / "y" / date

        if not x_dir.exists() or not y_dir.exists():
            print(f"[警告] 找不到 {date} 的 X 或 y 資料夾，跳過")
            continue

        for x_path in sorted(x_dir.glob("*.npy")):
            region = x_path.stem
            if regions_set is not None and region not in regions_set:
                continue
            y_path = y_dir / f"{region}.npy"
            if not y_path.exists():
                print(f"[警告] 找不到 {date}/{region} 對應的 y 檔案，跳過")
                continue

            y = np.load(y_path).astype(np.float32)
            if y.ndim == 3:
                y = y[0]  # (1, H, W) -> (H, W)
            x = np.load(x_path).astype(np.float32)   # (5, H, W)
            valid_mask = y != nodata_value            # (H, W)

            for i, band in enumerate(numeric_bands):
                ch = _BAND_INDEX.get(band, i)
                pixels = np.clip(x[ch], lo[i], hi[i])[valid_mask]
                sums[i]    += pixels.sum()
                sums_sq[i] += np.square(pixels, dtype=np.float64).sum()
                counts[i]  += pixels.size

    safe_counts = np.maximum(counts, 1)
    mean = sums / safe_counts
    var  = sums_sq / safe_counts - np.square(mean)
    std  = np.sqrt(np.maximum(var, 1e-8))

    return mean.astype(np.float32), std.astype(np.float32)


# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    XY_DIR         = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"
    TRAIN_RATIO    = 0.6
    VAL_RATIO      = 0.2
    PATCH_SIZE     = 256
    NUMERIC_BANDS  = ["Green", "Red", "NIR", "SWIR"]

    # ── 日期切分 ──────────────────────────────────────────────────────────────

    TRAIN_DATES, VALID_DATES, TEST_DATES = train_valid_test_split(
        base_dir=XY_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )
    print(f"Train：{TRAIN_DATES}")
    print(f"Valid ：{VALID_DATES}")
    print(f"Test  ：{TEST_DATES}\n")

    # ── 標準化參數（僅用 train_dates）────────────────────────────────────────

    mean, std = compute_band_mean_std(
        base_dir=XY_DIR,
        dates=TRAIN_DATES,
        numeric_bands=NUMERIC_BANDS,
    )
    print(f"mean={mean}")
    print(f"std ={std}\n")
