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

# y 中的遮蔽值：999 代表非農地範圍/無效區域。X 中的 999 因資料處理問題
# 混雜了原本的有效值，不能直接剔除；但各日期同一區塊的 999 位置相同，
# 因此統計 X 時改用 y（固定基準日期）中 999 的位置做遮蔽
_Y_MASK_VALUE = 999

# X 中需剔除的 nodata 值（y 遮蔽後農地範圍內仍可能出現）
_X_NODATA = 65535


def _get_y_mask(base_path: Path, mask_date: str, region: str, cache: dict) -> Optional[np.ndarray]:
    """
    載入 y/{mask_date}/{region}.npy 並回傳「有效像素」布林遮罩（y != 999）。
    y 檔案可能為 (1, H, W)，會先 squeeze 降成 (H, W) 再比對。
    各日期同一區塊的 999 位置相同，因此固定以 mask_date 的 y 為基準；
    結果存入 cache，同一區塊只讀檔一次。檔案不存在時回傳 None 並警告。
    """
    if region not in cache:
        y_path = base_path / "y" / mask_date / f"{region}.npy"
        if not y_path.exists():
            print(f"[警告] 找不到遮罩基準檔 {y_path}，該區塊將被跳過")
            cache[region] = None
        else:
            y = np.squeeze(np.load(y_path))  # (1, H, W) → (H, W)
            cache[region] = y != _Y_MASK_VALUE
    return cache[region]


# ── 日期切分 ──────────────────────────────────────────────────────────────────

def match_x_dates(
    base_dir: str,
    y_dates: List[str],
    lag: int = 1,
) -> List[List[str]]:
    """
    為每個 y 日期配對 X 日期：從 base_dir/X/ 掃描所有 X 觀測日期，
    對每個 y 日期取「小於等於該 y 日期」當中最大的 lag 筆 X 日期。

    Args:
        base_dir: 來源資料夾（含 X/ 子目錄，X/ 下每個子資料夾名稱即為 X 觀測日期）
        y_dates:  y 日期列表（如 ["20230130", "20230219"]）
        lag:      每個 y 日期要配對的 X 日期數量（預設 1）

    Returns:
        與 y_dates 一一對應的列表，每個元素為該 y 日期配對到的 lag 筆 X 日期
        （由近到遠排序，即 [最接近的, 次接近的, ...]）

    Raises:
        ValueError: 當某個 y 日期找不到足夠 lag 筆（≤ 該日期）的 X 日期時
    """
    if lag < 1:
        raise ValueError(f"lag 必須 >= 1，收到：{lag}")

    x_dir = Path(base_dir) / "X"
    if not x_dir.exists():
        raise FileNotFoundError(f"找不到 X/ 資料夾：{x_dir}")

    x_dates = sorted(p.name for p in x_dir.iterdir() if p.is_dir())
    if not x_dates:
        raise FileNotFoundError(f"在 {x_dir} 中找不到任何日期資料夾。")

    matched: List[List[str]] = []
    for y_date in y_dates:
        candidates = [d for d in x_dates if d <= y_date]
        if len(candidates) < lag:
            raise ValueError(
                f"y 日期 {y_date} 只找到 {len(candidates)} 筆 <= 該日期的 X 日期，"
                f"不足 lag={lag} 筆"
            )
        matched.append(candidates[-lag:][::-1])  # 由近到遠
    return matched

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


def train_valid_test_split_by_date(
    base_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    lag: int = 1,
) -> Tuple[
    List[List[str]], List[str],
    List[List[str]], List[str],
    List[List[str]], List[str],
    List[str], List[str], List[str],
]:
    """
    從 base_dir/y/ 掃描「所有」日期資料夾（不分年份），用固定 seed 隨機打亂後
    依 train_ratio / val_ratio 切成 train / valid / test（其餘歸為 test，預設 6:2:2）。

    輸出的日期分成 X 與 y 兩組：*_dates_y 為 y 的日期，*_dates_X 為以
    match_x_dates() 配對到的 X 日期（每個 y 日期取「小於等於該 y 日期」中
    最大的 lag 筆 X 日期，與 *_dates_y 一一對應）。

    並展開每組 y 日期資料夾下所有區域的 .npy 檔名（格式 "{date}/{region}.npy"）。

    Args:
        base_dir:     來源資料夾（含 y/ 與 X/ 子目錄，子資料夾名稱即為日期）
        train_ratio:  train 佔全部日期數的比例（預設 0.6）
        val_ratio:    valid 佔全部日期數的比例（預設 0.2），其餘歸為 test
        seed:         打亂日期順序用的隨機種子
        lag:          每個 y 日期要配對的 X 日期數量（預設 1）

    Returns:
        (train_dates_X, train_dates_y,
         val_dates_X,   val_dates_y,
         test_dates_X,  test_dates_y,
         train_files, val_files, test_files)
        其中 *_dates_X[i] 為 *_dates_y[i] 配對到的 lag 筆 X 日期（由近到遠）
    """
    y_dir = Path(base_dir) / "y"
    if not y_dir.exists():
        raise FileNotFoundError(f"找不到 y/ 資料夾：{y_dir}")

    all_dates = sorted(p.name for p in y_dir.iterdir() if p.is_dir())
    if not all_dates:
        raise FileNotFoundError(f"在 {y_dir} 中找不到任何日期資料夾。")

    total = len(all_dates)

    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必須落在有效範圍內，且 train_ratio + val_ratio < 1")
    if total < 3:
        raise ValueError(f"日期數量 ({total}) < 3，無法切分為 train/valid/test")

    shuffled = all_dates.copy()
    random.Random(seed).shuffle(shuffled)

    n_train = max(1, round(total * train_ratio))
    n_val   = max(1, round(total * val_ratio))
    if n_train + n_val >= total:
        n_val = max(1, total - n_train - 1)
    if n_train + n_val >= total:
        n_train = max(1, total - n_val - 1)

    train_dates_y = sorted(shuffled[:n_train])
    val_dates_y   = sorted(shuffled[n_train:n_train + n_val])
    test_dates_y  = sorted(shuffled[n_train + n_val:])

    train_dates_X = match_x_dates(base_dir, train_dates_y, lag=lag)
    val_dates_X   = match_x_dates(base_dir, val_dates_y, lag=lag)
    test_dates_X  = match_x_dates(base_dir, test_dates_y, lag=lag)

    def _expand(dates: List[str]) -> List[str]:
        files = []
        for date in dates:
            for npy_path in sorted((y_dir / date).glob("*.npy")):
                files.append(f"{date}/{npy_path.name}")
        return files

    train_files = _expand(train_dates_y)
    val_files   = _expand(val_dates_y)
    test_files  = _expand(test_dates_y)

    return (train_dates_X, train_dates_y,
            val_dates_X,   val_dates_y,
            test_dates_X,  test_dates_y,
            train_files, val_files, test_files)


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
    hist_max: int = 65535,
    mask_date: str = "20230130",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    只用 train_dates（與 Case2 空間切分時的 train_regions）計算各 numeric band
    的 lower~upper 百分位數，作為離群值裁切邊界（以訓練集統計量裁切，
    避免驗證/測試集的統計資訊洩漏到訓練流程中）。

    以直方圖（bin 寬度 = 1，範圍 0..hist_max）累加各波段像素的次數分布，
    再由累積分布反推百分位數，避免把所有像素值一次串接進記憶體。

    統計前先以 y/{mask_date} 各區塊中 999 的位置遮蔽 X（各日期同一區塊的
    999 位置相同，故固定用單一基準日期），並剔除 X 中值為 65535 的 nodata
    像素；X 本身的 999 可能為有效值，不剔除。

    資料來源：
        base_dir/X/{date}/{region}.npy        shape (5, H, W)
        base_dir/y/{mask_date}/{region}.npy   shape (H, W)，999 位置作為遮罩

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
        hist_max:      直方圖涵蓋的最大像素值（預設 65535，對應 uint16 上限）
        mask_date:     遮罩基準的 y 日期（預設 "20230130"），以
                       y/{mask_date}/{region}.npy 中值為 999 的位置遮蔽 X

    Returns:
        (clip_min, clip_max)，各為 shape (len(numeric_bands),) 的 float32 ndarray
    """
    base_path = Path(base_dir)
    n_bands = len(numeric_bands)
    hist = np.zeros((n_bands, hist_max + 1), dtype=np.int64)
    regions_set = set(regions) if regions is not None else None
    mask_cache: dict = {}

    for date in dates:
        x_dir = base_path / "X" / date

        if not x_dir.exists():
            print(f"[警告] 找不到 {date} 的 X 資料夾，跳過")
            continue

        for x_path in sorted(x_dir.glob("*.npy")):
            region = x_path.stem
            if regions_set is not None and region not in regions_set:
                continue

            mask = _get_y_mask(base_path, mask_date, region, mask_cache)
            if mask is None:
                continue

            x = np.load(x_path)  # (5, H, W)
            if mask.shape != x.shape[1:]:
                print(f"[警告] {date}/{region} 的 X 尺寸 {x.shape[1:]} 與 y 遮罩尺寸 {mask.shape} 不符，跳過")
                continue

            for i, band in enumerate(numeric_bands):
                ch = _BAND_INDEX.get(band, i)
                pixels = x[ch][mask]
                pixels = pixels[(pixels >= 0) & (pixels <= hist_max) & (pixels != _X_NODATA)]
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
    clip_min: Optional[np.ndarray] = None,
    clip_max: Optional[np.ndarray] = None,
    mask_date: str = "20230130",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算各 numeric band 的 mean / std，供 Z-score 標準化使用。
    來源為切割成 11 個區域後的 "xy對應" 資料夾（每個日期底下有多個區域 .npy，
    如 N1.npy, N2.npy, ...），逐一讀取該日期下所有區域檔案累計統計量。

    統計前先以 y/{mask_date} 各區塊中 999 的位置遮蔽 X（各日期同一區塊的
    999 位置相同，故固定用單一基準日期），並剔除 X 中值為 65535 的 nodata
    像素；X 本身的 999 可能為有效值，不剔除。被遮蔽/剔除的像素完全不計入統計。

    重要：只能傳入 train_dates（時間切分）或 train_regions（空間切分）計算，
    絕不可使用 val/test 日期或區塊，否則驗證/測試集的統計資訊會洩漏到訓練流程中。

    資料來源：
        base_dir/X/{date}/{region}.npy        shape (5, H, W)
            channel 順序：[0]=Green, [1]=Red, [2]=NIR, [3]=SWIR, [4]=SCL（未列入 numeric_bands 則不計算）
        base_dir/y/{mask_date}/{region}.npy   shape (H, W)，999 位置作為遮罩

    Args:
        base_dir:      切割後的來源資料夾（xy對應/）
        dates:         訓練日期列表（train_dates）
        numeric_bands: 要計算的波段名稱列表（如 ["Green", "Red", "NIR", "SWIR"]）
        regions:       訓練區塊列表（空間切分 Case2 用；None 表示不篩選，時間切分 Case1
                       應保持 None）
        clip_min:      各波段裁切下界（shape (len(numeric_bands),)）；None 時退回 0
                       （建議傳入 compute_band_percentiles() 的回傳值，做 1st~99th 百分位裁切）
        clip_max:      各波段裁切上界；None 時退回 10000
        mask_date:     遮罩基準的 y 日期（預設 "20230130"），以
                       y/{mask_date}/{region}.npy 中值為 999 的位置遮蔽 X

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
    mask_cache: dict = {}

    for date in dates:
        x_dir = base_path / "X" / date

        if not x_dir.exists():
            print(f"[警告] 找不到 {date} 的 X 資料夾，跳過")
            continue

        for x_path in sorted(x_dir.glob("*.npy")):
            region = x_path.stem
            if regions_set is not None and region not in regions_set:
                continue

            mask = _get_y_mask(base_path, mask_date, region, mask_cache)
            if mask is None:
                continue

            x = np.load(x_path).astype(np.float32)   # (5, H, W)
            if mask.shape != x.shape[1:]:
                print(f"[警告] {date}/{region} 的 X 尺寸 {x.shape[1:]} 與 y 遮罩尺寸 {mask.shape} 不符，跳過")
                continue

            for i, band in enumerate(numeric_bands):
                ch = _BAND_INDEX.get(band, i)
                pixels = x[ch][mask]
                pixels = pixels[pixels != _X_NODATA]
                if pixels.size == 0:
                    continue
                pixels = np.clip(pixels, lo[i], hi[i])
                sums[i]    += pixels.sum(dtype=np.float64)
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
