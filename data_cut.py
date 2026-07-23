"""
data_cut.py

將 xy對應/ 的 y/{date}/{region}.npy 和 X/{date}/{region}.npy 切割成 patches。
每個日期底下有 11 個區域（N1,N2,EN,E1,E2,E3,W1-W5），各自視為一張獨立的圖分別切割。

輸出目錄（依 mode 固定）：
    train  → training_data/
    test   → testing_data/
    valid  → valid_data/
輸出結構：out_dir/y/{date}/{region}/patch_*.npy、out_dir/X/{date}/{region}/patch_*.npy
"""

import json
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


# ── Helper functions ──────────────────────────────────────────────────────────

def pad_to_multiple(arr: np.ndarray, patch_size: int, pad_value: float = 0.0) -> np.ndarray:
    """將陣列 padding 到 patch_size 的整倍數（右側與下側補值）。"""
    H, W = arr.shape[-2], arr.shape[-1]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 2:
        return np.pad(arr, ((0, pad_h), (0, pad_w)),
                      mode="constant", constant_values=pad_value)
    else:  # (C, H, W)
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)),
                      mode="constant", constant_values=pad_value)


def compute_irrigation_ratio(window: np.ndarray, ignore_value: float = 999.0) -> float:
    """計算視窗內的灌水比例，排除 ignore_value 後再算 1 的比例。"""
    valid = window[window != ignore_value]
    if valid.size == 0:
        return 0.0
    return float((valid == 1.0).mean())


def ratio_to_stride_x(ratio: float, stride_x_max: int, stride_x_min: int) -> int:
    """灌水比例 → 向右 stride 的線性插值映射。
    ratio 越大（灌水越多）→ stride 越小（切越密）。
    """
    ratio = min(max(ratio, 0.0), 1.0)
    stride = stride_x_max - ratio * (stride_x_max - stride_x_min)
    return max(stride_x_min, min(int(round(stride)), stride_x_max))


def get_patch_positions_dynamic(
    mask_arr: np.ndarray,
    patch_size: int,
    stride_y: int = 256,
    stride_x_max: int = 256,
    stride_x_min: int = 64,
    ignore_value: float = 999.0,
    skip_all_nodata: bool = True,
) -> List[Tuple[int, int]]:
    """依當前視窗的灌水比例動態決定下一次向右的 stride，向下 stride 固定。
    skip_all_nodata=True：跳過全部為 ignore_value 的視窗（靜默移除）。
    skip_all_nodata=False：保留全 nodata 視窗，供後續篩選 block 計數。
    """
    H, W = mask_arr.shape
    positions: List[Tuple[int, int]] = []
    y = 0
    while y <= H - patch_size:
        x = 0
        while x <= W - patch_size:
            window = mask_arr[y:y + patch_size, x:x + patch_size]
            if not skip_all_nodata or not np.all(window == ignore_value):
                positions.append((y, x))
            ratio = compute_irrigation_ratio(window, ignore_value=ignore_value)
            x += ratio_to_stride_x(ratio, stride_x_max, stride_x_min)
        y += stride_y
    return positions


def get_patch_positions_fixed(
    mask_arr: np.ndarray,
    patch_size: int,
    stride_x: int,
    stride_y: int,
    ignore_value: float = 999.0,
    skip_all_nodata: bool = True,
) -> List[Tuple[int, int]]:
    """固定 stride 版本。
    skip_all_nodata=True：跳過全部為 ignore_value 的視窗（靜默移除）。
    skip_all_nodata=False：保留所有視窗，包含全 nodata（供重建或後續篩選 block 使用）。
    """
    H, W = mask_arr.shape
    positions: List[Tuple[int, int]] = []
    y = 0
    while y <= H - patch_size:
        x = 0
        while x <= W - patch_size:
            if not skip_all_nodata or not np.all(
                mask_arr[y:y + patch_size, x:x + patch_size] == ignore_value
            ):
                positions.append((y, x))
            x += stride_x
        y += stride_y
    return positions


# ── Main function ─────────────────────────────────────────────────────────────

def cut_xy_patches(
    xy_dir: str,
    dates: List[str] = None,
    regions: Optional[List[str]] = None,
    mode: str = "train",
    out_dir_name: Optional[str] = None,
    patch_size: int = 256,
    stride_x: int = 256,
    stride_y: int = 256,
    keep_remainder: bool = True,
    use_dynamic_stride_x: bool = False,
    stride_x_min: int = 64,
    filter_all_nodata: bool = False,
    filter_no_water: bool = False,
    nodata_value: int = 999,
    water_value: int = 1,
    cloud_rate_threshold: Optional[float] = None,
    cloud_scl_classes: Optional[List[int]] = None,
) -> str:
    """
    將 xy對應/ 底下每個日期的 11 個區域 .npy 切割成 patches
    （每個 (date, region) 都視為一張獨立的圖，套用相同的切割邏輯）。

    Args:
        xy_dir:               來源資料夾（含 y/ 和 X/ 子目錄）
        dates:                要處理的日期列表（如 ["20230130", "20230219"]）；
                              None 表示處理 y/ 下所有日期資料夾
        regions:              要處理的區塊列表（如 ["N1", "N2"]）；None 表示不篩選
                              （用於空間切分 Case2：同一批日期依區塊分別切給
                              train/valid/test，避免同一區塊出現在多個 split）
        mode:                 "train" | "test" | "valid"
        out_dir_name:         輸出資料夾名稱；None 時依 mode 決定
                              （"training_data" / "valid_data" / "testing_data"）。
                              時間切分與空間切分兩個 case 應各自傳入不同名稱
                              （例如 "training_data_spatial"），避免共用固定資料夾名稱
                              導致兩個 case 的切割結果互相覆蓋
        patch_size:           patch 邊長（正方形）
        stride_x:             train 模式向右 stride；動態模式下作為 stride_x_max
        stride_y:             train 模式向下 stride
        keep_remainder:       True=padding 保留邊界；僅 train 模式有效
                              （test/valid 模式強制 True）
        use_dynamic_stride_x: True=依視窗灌水比例動態調整向右 stride；僅 train 有效
        stride_x_min:         動態模式下最小向右 stride
        filter_all_nodata:    True=過濾全 nodata 的 patch；僅 train 有效
        filter_no_water:      True=過濾無水體像素的 patch；僅 train 有效
        nodata_value:         y 遮罩中代表無效像素的值
        water_value:          y 遮罩中代表水體的值
        cloud_rate_threshold: 雲遮蔽率閾值（0–1）；超過則剔除 patch；僅 train 有效；None=不篩選
        cloud_scl_classes:    雲相關 SCL 類別；None=預設 [3, 8, 9]（雲影/中機率雲/高機率雲）

    Returns:
        輸出資料夾的完整路徑字串
    """
    xy_dir = Path(xy_dir)
    parent = xy_dir.parent

    mode = mode.lower()
    if out_dir_name is not None:
        out_dir = parent / out_dir_name
    elif mode == "train":
        out_dir = parent / "training_data"
    elif mode == "test":
        out_dir = parent / "testing_data"
    elif mode == "valid":
        out_dir = parent / "valid_data"
    else:
        raise ValueError(f"mode 必須是 'train'、'test' 或 'valid'，收到：{mode!r}")

    if mode not in ("train", "test", "valid"):
        raise ValueError(f"mode 必須是 'train'、'test' 或 'valid'，收到：{mode!r}")

    is_test_like = mode in ("test", "valid")
    P = patch_size
    y_src = xy_dir / "y"
    x_src = xy_dir / "X"

    date_dirs = sorted(p for p in y_src.iterdir() if p.is_dir())
    if dates is not None:
        dates_set = set(dates)
        date_dirs = [p for p in date_dirs if p.name in dates_set]

    regions_set = set(regions) if regions is not None else None
    region_jobs = [
        (date_dir.name, y_npy.stem, y_npy)
        for date_dir in date_dirs
        for y_npy in sorted(date_dir.glob("*.npy"))
        if regions_set is None or y_npy.stem in regions_set
    ]

    for target_date, region, y_npy in region_jobs:
        print(f"── 處理 {target_date}/{region} ──")

        mask = np.load(y_npy).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[0]  # (1, H, W) -> (H, W)
        orig_h, orig_w = mask.shape

        if is_test_like:
            mask_proc = pad_to_multiple(mask, P, pad_value=nodata_value)
            eff_stride_x = P
            eff_stride_y = P
            eff_keep = True
        else:
            mask_proc = pad_to_multiple(mask, P, pad_value=nodata_value) if keep_remainder else mask
            eff_stride_x = stride_x
            eff_stride_y = stride_y
            eff_keep = keep_remainder

        padded_h, padded_w = mask_proc.shape
        pad_h = padded_h - orig_h
        pad_w = padded_w - orig_w

        # 計算切割位置
        if is_test_like:
            positions = get_patch_positions_fixed(
                mask_proc, P,
                stride_x=P, stride_y=P,
                ignore_value=nodata_value,
                skip_all_nodata=False,
            )
        elif use_dynamic_stride_x:
            positions = get_patch_positions_dynamic(
                mask_proc, P,
                stride_y=eff_stride_y,
                stride_x_max=eff_stride_x,
                stride_x_min=stride_x_min,
                ignore_value=nodata_value,
                skip_all_nodata=False,
            )
        else:
            positions = get_patch_positions_fixed(
                mask_proc, P,
                stride_x=eff_stride_x,
                stride_y=eff_stride_y,
                ignore_value=nodata_value,
                skip_all_nodata=False,
            )

        # train mode 篩選
        if mode == "train" and (filter_all_nodata or filter_no_water):
            total_before = len(positions)
            nodata_cnt = 0
            no_water_cnt = 0
            kept = []
            for r, c in positions:
                patch = mask_proc[r:r + P, c:c + P]
                if filter_all_nodata and np.all(patch == nodata_value):
                    nodata_cnt += 1
                    continue
                if filter_no_water:
                    valid = patch[patch != nodata_value]
                    if valid.size > 0 and not np.any(valid == water_value):
                        no_water_cnt += 1
                        continue
                kept.append((r, c))
            positions = kept
            print(f"  篩選：總 {total_before} | nodata 剔除 {nodata_cnt} | "
                  f"無水體剔除 {no_water_cnt} | 保留 {len(positions)}")

        # 早期載入 X 資料（供雲量篩選使用）
        x_proc = None
        x_npy = x_src / target_date / f"{region}.npy"
        if not x_npy.exists():
            print(f"  [警告] 找不到 X/{target_date}/{region}.npy，跳過 X 切割")
        else:
            x_arr = np.load(x_npy).astype(np.float32)  # (5, H, W)
            x_proc = pad_to_multiple(x_arr, P, pad_value=0.0) if eff_keep else x_arr

        # 雲量篩選（train only）
        if mode == "train" and cloud_rate_threshold is not None and x_proc is not None:
            _cloud_cls = cloud_scl_classes if cloud_scl_classes is not None else [3, 8, 9]
            total_before_cloud = len(positions)
            cloud_cnt = 0
            kept_cloud = []
            for r, c in positions:
                y_patch = mask_proc[r:r + P, c:c + P]
                scl_patch = x_proc[4, r:r + P, c:c + P]
                valid_mask = y_patch != nodata_value
                total_valid = int(valid_mask.sum())
                if total_valid == 0:
                    kept_cloud.append((r, c))
                    continue
                cloud_count = int(np.isin(scl_patch[valid_mask], _cloud_cls).sum())
                if cloud_count / total_valid > cloud_rate_threshold:
                    cloud_cnt += 1
                    continue
                kept_cloud.append((r, c))
            positions = kept_cloud
            print(f"  雲量篩選（閾值={cloud_rate_threshold}）：保留 {len(positions)}，"
                  f"雲量過高剔除 {cloud_cnt}")

        print(f"  共切出 {len(positions)} 張 patches")

        # 儲存 y patches（先清空舊資料）
        out_y = out_dir / "y" / target_date / region
        if out_y.exists():
            shutil.rmtree(out_y)
        out_y.mkdir(parents=True)
        for i, (r, c) in enumerate(positions):
            np.save(out_y / f"patch_{i:04d}.npy", mask_proc[r:r + P, c:c + P])

        # test/valid: 儲存 patch_info.json（座標 + padding 資訊，供重建用）
        if is_test_like:
            patch_info = {
                "original_h": orig_h,
                "original_w": orig_w,
                "pad_h": pad_h,
                "pad_w": pad_w,
                "padded_h": padded_h,
                "padded_w": padded_w,
                "patches": [{"idx": i, "row": r, "col": c}
                            for i, (r, c) in enumerate(positions)],
            }
            with open(out_y / "patch_info.json", "w", encoding="utf-8") as f:
                json.dump(patch_info, f, ensure_ascii=False, indent=2)

        # 儲存 X patches
        if x_proc is not None:
            out_x = out_dir / "X" / target_date / region
            if out_x.exists():
                shutil.rmtree(out_x)
            out_x.mkdir(parents=True)
            for i, (r, c) in enumerate(positions):
                np.save(out_x / f"patch_{i:04d}.npy", x_proc[:, r:r + P, c:c + P])

        print(f"  ✓ 已儲存至 {out_dir}")

    return str(out_dir)


def reconstruct_y_patches(
    date_dir: str,
    nodata_value: float = 999.0,
    output_name: str = "result.npy",
) -> str:
    """
    依 patch_info.json 將日期資料夾下的 patch_*.npy 還原成完整圖像，
    存為 result.npy（預設）。

    Args:
        date_dir:     日期資料夾路徑（含 patch_*.npy 與 patch_info.json）
        nodata_value: 畫布初始填充值（無 patch 覆蓋的區域）
        output_name:  輸出檔名（預設 "result.npy"）

    Returns:
        輸出檔案的完整路徑字串
    """
    date_dir = Path(date_dir)
    info_path = date_dir / "patch_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"找不到 patch_info.json：{info_path}")

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    padded_h    = info["padded_h"]
    padded_w    = info["padded_w"]
    original_h  = info["original_h"]
    original_w  = info["original_w"]
    patches_meta = info["patches"]

    canvas = np.full((padded_h, padded_w), nodata_value, dtype=np.float32)

    P = None  # patch_size，從第一張 patch 推斷
    for meta in patches_meta:
        idx = meta["idx"]
        r   = meta["row"]
        c   = meta["col"]
        patch_path = date_dir / f"patch_{idx:04d}.npy"
        if not patch_path.exists():
            print(f"[警告] 找不到 {patch_path.name}，跳過")
            continue
        patch = np.load(patch_path).astype(np.float32)
        if P is None:
            P = patch.shape[0]
        canvas[r:r + P, c:c + P] = patch

    result   = canvas[:original_h, :original_w]
    out_path = date_dir / output_name
    np.save(out_path, result)
    print(f"  ✓ 已儲存：{out_path}  shape={result.shape}  dtype={result.dtype}")
    return str(out_path)


if __name__ == "__main__":
    XY_DIR = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"

    TRAIN_DATES = ["20230130", "20230219", "20230301"]
    VALID_DATES = ["20230306"]
    TEST_DATES  = ["20230316"]

    # train mode，固定 stride，過濾全 nodata patch
    out_train = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TRAIN_DATES,
        mode="train",
        patch_size=256,
        stride_x=256,
        stride_y=256,
        keep_remainder=False,
        use_dynamic_stride_x=False,
        filter_all_nodata=True,
        filter_no_water=False,
    )
    print(f"\nTrain 輸出：{out_train}\n")

    # test mode（強制 non-overlapping + padding）
    out_valid = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=VALID_DATES,
        mode="valid",
        patch_size=256,
    )
    print(f"\nValid 輸出：{out_valid}")

    out_test = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TEST_DATES,
        mode="test",
        patch_size=256,
    )
    print(f"\nTest 輸出：{out_test}\n")

    # reconstruct_y_patches 測試：將 test 切割結果逐區域還原成完整圖像
    print("── reconstruct_y_patches 測試 ──")
    for date in TEST_DATES:
        y_date_dir = os.path.join(out_test, "y", date)
        for region in sorted(os.listdir(y_date_dir)):
            region_dir = os.path.join(y_date_dir, region)
            if os.path.isdir(region_dir):
                reconstruct_y_patches(date_dir=region_dir)
