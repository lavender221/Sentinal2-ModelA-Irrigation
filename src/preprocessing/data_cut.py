"""
data_cut.py

將 xy對應/ 的 y/{date}/{region}.npy 和 X/{date}/{region}.npy 切割成 patches。
每個日期底下有 11 個區域（N1,N2,EN,E1,E2,E3,W1-W5），各自視為一張獨立的圖分別切割。

輸出（三個 split 集中在同一個資料夾，由 out_dir 指定，預設為 xy_dir 上一層的
cutV1/；檔名依 mode 固定）：
    train  → out_dir/train_info.json
    valid  → out_dir/valid_info.json
    test   → out_dir/test_info.json
json 內容：每個 (date, region) 一筆 item，記錄重建/即時切割用座標與 padding 資訊，
不儲存任何 patch .npy（由 Dataset 依座標從原始大圖即時切割）。
train 的每個 patch 條目額外記錄灌水比例（water_ratio）、SCL(3, 8, 9) 比例
（scl_cloud_ratio，分母皆為 y != nodata 的有效像素）與 999 比例
（nodata_ratio，分母為整張 patch 的像素數），皆為 0~1 小數。
切割階段不做任何 patch 篩選；篩選改由宣告 Dataset 時依這些比例欄位進行。
"""

import json
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


def get_patch_positions_fixed(
    mask_arr: np.ndarray,
    patch_size: int,
    stride: int,
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
            x += stride
        y += stride
    return positions


# ── Main function ─────────────────────────────────────────────────────────────

def cut_xy_patches(
    xy_dir: str,
    dates: List[str] = None,
    dates_X: Optional[List[List[str]]] = None,
    regions: Optional[List[str]] = None,
    mode: str = "train",
    out_dir: Optional[str] = None,
    patch_size: int = 256,
    stride: int = 256,
    keep_remainder: bool = True,
    nodata_value: int = 999,
    water_value: int = 1,
    cloud_scl_classes: Optional[List[int]] = None,
) -> str:
    """
    將 xy對應/ 底下每個日期的 11 個區域 .npy 切割成 patches
    （每個 (date, region) 都視為一張獨立的圖，套用相同的切割邏輯）。

    儲存行為（train/valid/test 一律相同）：
        整個 split 只寫一份 out_dir/{mode}_info.json（不儲存任何 patch .npy），
        每個 (date, region) 一筆 item，含 x_date、原圖尺寸、padding 資訊與
        各 patch 的切割座標，由 Dataset 依座標從原始大圖即時切割。
        train 的每個 patch 條目額外記錄 water_ratio（灌水比例）、
        scl_cloud_ratio（SCL(3,8,9) 比例，分母皆為 y != nodata 的有效像素）與
        nodata_ratio（999 比例，分母為整張 patch 的像素數），皆為 0~1 小數。
        切割階段不做任何 patch 篩選；篩選改由宣告 Dataset 時依這些比例欄位進行。

    Args:
        xy_dir:            來源資料夾（含 y/ 和 X/ 子目錄）
        dates:             要處理的日期列表（如 ["20230130", "20230219"]）；
                           None 表示處理 y/ 下所有日期資料夾
        dates_X:           與 dates 一一對應的 X 日期列表（match_x_dates() 的回傳格式，
                           每個 y 日期對應 lag 筆 X 日期，取最近的第 1 筆作為 X 來源）；
                           None 表示 X 直接使用與 y 相同的日期
        regions:           要處理的區塊列表（如 ["N1", "N2"]）；None 表示不篩選
                           （用於空間切分 Case2：同一批日期依區塊分別切給
                           train/valid/test，避免同一區塊出現在多個 split）
        mode:              "train" | "test" | "valid"，決定輸出檔名 {mode}_info.json
        out_dir:           輸出資料夾完整路徑（如模型輸出資料夾
                           model_output/case1_time_split）；None 時預設為
                           xy_dir 上一層的 cutV1/。時間切分與空間切分兩個 case
                           應各自傳入不同資料夾，避免切割結果互相覆蓋
        patch_size:        patch 邊長（正方形）
        stride:            train 模式滑動視窗 stride（向右與向下相同）
        keep_remainder:    True=padding 保留邊界；僅 train 模式有效
                           （test/valid 模式強制 True）
        nodata_value:      y 遮罩中代表無效像素的值
        water_value:       y 遮罩中代表水體的值
        cloud_scl_classes: 雲相關 SCL 類別；None=預設 [3, 8, 9]（雲影/中機率雲/高機率雲）

    Returns:
        輸出 json 檔（out_dir/{mode}_info.json）的完整路徑字串
    """
    xy_dir = Path(xy_dir)

    mode = mode.lower()
    if mode not in ("train", "test", "valid"):
        raise ValueError(f"mode 必須是 'train'、'test' 或 'valid'，收到：{mode!r}")

    out_dir = Path(out_dir) if out_dir is not None else xy_dir.parent / "cutV1"

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

    if dates_X is not None:
        if dates is None or len(dates_X) != len(dates):
            raise ValueError("dates_X 必須與 dates 一一對應（長度相同）")
        # lag > 1 時取最近（第 1 筆）的 X 日期
        x_date_map = {d: xs[0] for d, xs in zip(dates, dates_X)}
    else:
        x_date_map = None

    items = []
    for target_date, region, y_npy in region_jobs:
        print(f"── 處理 {target_date}/{region} ──")

        mask = np.load(y_npy).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[0]  # (1, H, W) -> (H, W)
        orig_h, orig_w = mask.shape

        if is_test_like:
            mask_proc = pad_to_multiple(mask, P, pad_value=nodata_value)
            eff_stride = P
            eff_keep = True
        else:
            mask_proc = pad_to_multiple(mask, P, pad_value=nodata_value) if keep_remainder else mask
            eff_stride = stride
            eff_keep = keep_remainder

        padded_h, padded_w = mask_proc.shape
        pad_h = padded_h - orig_h
        pad_w = padded_w - orig_w

        # 計算切割位置
        if is_test_like:
            positions = get_patch_positions_fixed(
                mask_proc, P,
                stride=P,
                ignore_value=nodata_value,
                skip_all_nodata=False,
            )
        else:
            positions = get_patch_positions_fixed(
                mask_proc, P,
                stride=eff_stride,
                ignore_value=nodata_value,
                skip_all_nodata=False,
            )

        # X 來源日期：優先使用 dates_X 配對到的日期，否則用與 y 相同的日期
        x_date = x_date_map.get(target_date, target_date) if x_date_map is not None else target_date

        # 載入 X（train 才需要：SCL 比例統計；
        # valid/test 只寫座標、由 Dataset 即時切割，不在此載入）
        x_proc = None
        if mode == "train":
            x_npy = x_src / x_date / f"{region}.npy"
            if not x_npy.exists():
                print(f"  [警告] 找不到 X/{x_date}/{region}.npy，SCL 比例將記為 null")
            else:
                x_arr = np.load(x_npy).astype(np.float32)  # (5, H, W)
                x_proc = pad_to_multiple(x_arr, P, pad_value=0.0) if eff_keep else x_arr

        print(f"  共切出 {len(positions)} 張 patches")

        # 建立每張 patch 的條目（不做任何篩選）：
        # 座標供重建/即時切割用；train 額外記錄 water_ratio / scl_cloud_ratio
        # （分母為 y != nodata 的有效像素，整張皆為 nodata 時為 null）與
        # nodata_ratio（999 佔整張 patch 的比例），供宣告 Dataset 時篩選
        _cloud_cls = cloud_scl_classes if cloud_scl_classes is not None else [3, 8, 9]
        patches_meta = []
        for i, (r, c) in enumerate(positions):
            meta = {"idx": i, "row": r, "col": c}
            if mode == "train":
                y_patch = mask_proc[r:r + P, c:c + P]
                valid_mask = y_patch != nodata_value
                meta["water_ratio"] = (
                    float((y_patch[valid_mask] == water_value).mean())
                    if valid_mask.any() else None
                )
                scl_cloud_ratio = None
                if x_proc is not None and valid_mask.any():
                    scl_patch = x_proc[4, r:r + P, c:c + P]
                    scl_cloud_ratio = float(np.isin(scl_patch[valid_mask], _cloud_cls).mean())
                meta["scl_cloud_ratio"] = scl_cloud_ratio
                meta["nodata_ratio"] = float((y_patch == nodata_value).mean())
            patches_meta.append(meta)

        items.append({
            "date": target_date,
            "region": region,
            "x_date": x_date,
            "original_h": orig_h,
            "original_w": orig_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "padded_h": padded_h,
            "padded_w": padded_w,
            "patches": patches_meta,
        })

    # 整個 split 集中寫成一份 json
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_info.json"
    info = {"mode": mode, "patch_size": P, "items": items}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"✓ 已儲存 {out_path}（{len(items)} 個 (date, region)，"
          f"共 {sum(len(it['patches']) for it in items)} 張 patches）")

    return str(out_path)


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

    # train mode（不做任何篩選，json 內含 water_ratio / scl_cloud_ratio / nodata_ratio）
    train_info = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TRAIN_DATES,
        mode="train",
        patch_size=256,
        stride=256,
        keep_remainder=False,
    )
    print(f"\nTrain 輸出：{train_info}\n")

    # valid/test mode（強制 non-overlapping + padding）
    valid_info = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=VALID_DATES,
        mode="valid",
        patch_size=256,
    )
    print(f"\nValid 輸出：{valid_info}")

    test_info = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TEST_DATES,
        mode="test",
        patch_size=256,
    )
    print(f"\nTest 輸出：{test_info}")
