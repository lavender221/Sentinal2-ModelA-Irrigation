import csv
import json
import mmap
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio

BANDS = ["B03_Green", "B04_Red", "B8A_NIR", "B11_SWIR", "SCL"]

# 11 個切割區域: name -> (x1, y1, x2, y2),(x1,y1) 左上角、(x2,y2) 右下角,單位為像素
REGIONS = {
    "N1": (10000, 0, 17500, 5000),
    "N2": (7500, 5000, 15000, 10000),
    "EN": (15000, 5000, 20000, 10000),
    "E1": (12500, 15000, 17500, 20000),
    "E2": (12500, 20000, 17500, 25000),
    "E3": (10000, 25000, 15000, 30000),
    "W1": (2500, 10000, 10000, 15000),
    "W2": (0, 15000, 10000, 20000),
    "W3": (0, 20000, 7500, 25000),
    "W4": (0, 25000, 7500, 30000),
    "W5": (2500, 30000, 7500, 35000),
}


def nc_to_npy(nc_file: str, npy_root: str) -> Tuple[List[str], List[str]]:
    r"""將 nc 檔案中每個 image_date 的 5 個波段疊合成 (5, H, W) 陣列,逐筆存成 npy 檔。

    image_date 若有重複值,只取第一個出現的 match 索引。
    每存完一筆就釋放記憶體,不會把所有日期的資料同時留在記憶體中。

    npy_root: 含 {image_date} 佔位符的路徑樣板,例如:
        r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\S2訓練資料\全年度Xy對應\{image_date}"
    實際檔名交給 np.save 自動補上 .npy 副檔名。
    """
    saved_dates = []
    saved_paths = []

    # netCDF4 底層的 C 函式庫在 Windows 上開啟非 ASCII 路徑時會失敗,
    # 因此改用 Python 自行以 mmap 映射檔案,再以記憶體模式交給 netCDF4 解析。
    with open(nc_file, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        with nc.Dataset(nc_file, "r", memory=mm) as ds:
            image_date = np.array(ds.variables["image_date"][:])

            # image_date 可能有重複值,重複時只取第一個出現的 match 索引
            date_to_index = {}
            for idx, d in enumerate(image_date):
                date_to_index.setdefault(d, idx)

            total = len(date_to_index)
            for i, (date_str, idx) in enumerate(date_to_index.items(), start=1):
                print(f"[{i}/{total}] 處理中 image_date={date_str} ...")

                bands = []
                for band_name in BANDS:
                    var = ds.variables[band_name]
                    var.set_auto_mask(False)
                    bands.append(np.asarray(var[idx, :, :]))

                stacked = np.stack(bands, axis=0)  # shape (5, 39001, 21001), dtype uint16

                npy_path = npy_root.format(image_date=date_str)
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, stacked)

                final_path = npy_path if npy_path.endswith(".npy") else npy_path + ".npy"
                saved_dates.append(date_str)
                saved_paths.append(final_path)
                print(f"[{i}/{total}] 已存檔 image_date={date_str} -> {final_path}  shape={stacked.shape}")

                del bands, stacked

        mm.close()

    return saved_dates, saved_paths

def tif_to_npy(tif_file: str, tif_txt: str, npy_root: str) -> Tuple[List[str], List[str]]:
    r"""將 tif 檔案中每個 S2_Image_Date 的 5 個波段疊合成 (5, H, W) 陣列,逐筆存成 npy 檔。

    tif_txt 為該 tif 的波段說明檔 (CSV 格式),欄位為:
        Band_Index, Band_Name, Target_Irrigation_Date, S2_Image_Date, Days_Relation, S2_Tiles_Used
    Band_Name 以 BANDS 中的名稱結尾 (如 ..._B03_Green、..._SCL) 用來判斷該列對應哪個波段。
    S2_Image_Date 若重複(不同 Target_Irrigation_Date 對應到同一張影像),只取第一次出現的分組。

    存檔方式與 nc_to_npy 相同: npy_root 為含 {image_date} 佔位符的路徑樣板,
    交給 np.save 自動補上 .npy 副檔名。
    """
    # 解析波段說明檔,依 S2_Image_Date 分組出每個日期對應的 5 個 Band_Index
    date_to_band_indices = {}
    with open(tif_txt, "r", encoding="utf-8") as f:
        next(f)  # 跳過表頭
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            band_index = int(parts[0].replace("Band_", ""))
            band_name = parts[1]
            image_date = parts[3]

            band_group = date_to_band_indices.setdefault(image_date, {})
            for band_key in BANDS:
                if band_name.endswith(band_key) and band_key not in band_group:
                    band_group[band_key] = band_index

    saved_dates = []
    saved_paths = []

    with rasterio.open(tif_file) as ds:
        total = len(date_to_band_indices)
        for i, (date_str, band_group) in enumerate(date_to_band_indices.items(), start=1):
            missing = [k for k in BANDS if k not in band_group]
            if missing:
                raise ValueError(f"image_date={date_str} 缺少波段 {missing},請檢查 {tif_txt}")

            print(f"[{i}/{total}] 處理中 image_date={date_str} ...")

            bands = [ds.read(band_group[band_key]) for band_key in BANDS]
            stacked = np.stack(bands, axis=0)  # shape (5, H, W), dtype 依 tif 而定

            npy_path = npy_root.format(image_date=date_str)
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, stacked)

            final_path = npy_path if npy_path.endswith(".npy") else npy_path + ".npy"
            saved_dates.append(date_str)
            saved_paths.append(final_path)
            print(f"[{i}/{total}] 已存檔 image_date={date_str} -> {final_path}  shape={stacked.shape}")

            del bands, stacked

    return saved_dates, saved_paths

def tif_to_npy_target(
    tif_file: str, tif_txt: str, npy_root: str, year: str = "2023"
) -> Tuple[List[str], List[str]]:
    r"""將灌溉網格 (y/label) tif 中每個 band 的單一波段存成 npy 檔。

    tif_txt 格式為每行 "BandN: MM/DD",例如 "Band1: 01/30"(無年份)。
    這裡補上 year 參數組成 YYYYMMDD 格式的 image_date,與 nc_to_npy / tif_to_npy
    存的 X 資料日期格式一致,方便日後依日期配對 X、y。

    每個 band 只有單一通道,存檔前補一個維度成 (1, H, W)。
    存檔方式與 nc_to_npy 相同: npy_root 為含 {image_date} 佔位符的路徑樣板,
    交給 np.save 自動補上 .npy 副檔名。
    """
    # 解析波段說明檔,取得每個 Band_Index 對應的日期 (補上年份)
    band_index_to_date = {}
    with open(tif_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            band_part, date_part = line.split(":")
            band_index = int(band_part.strip().replace("Band", ""))
            mmdd = date_part.strip().replace("/", "")
            band_index_to_date[band_index] = f"{year}{mmdd}"

    saved_dates = []
    saved_paths = []

    with rasterio.open(tif_file) as ds:
        total = len(band_index_to_date)
        for i, (band_index, date_str) in enumerate(band_index_to_date.items(), start=1):
            print(f"[{i}/{total}] 處理中 image_date={date_str} (Band_{band_index}) ...")

            band_data = ds.read(band_index)
            stacked = band_data[np.newaxis, ...]  # shape (1, H, W)

            npy_path = npy_root.format(image_date=date_str)
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, stacked)

            final_path = npy_path if npy_path.endswith(".npy") else npy_path + ".npy"
            saved_dates.append(date_str)
            saved_paths.append(final_path)
            print(f"[{i}/{total}] 已存檔 image_date={date_str} -> {final_path}  shape={stacked.shape}")

            del band_data, stacked

    return saved_dates, saved_paths

def plot_band_distributions(npy_path: str, bin_width: int = 500) -> str:
    r"""讀取 X 資料夾中的一個 npy 檔 (shape (C, H, W)),
    畫出前 4 個 band (B03_Green、B04_Red、B8A_NIR、B11_SWIR) 的數值分布直方圖 (2x2 排列),
    每 bin_width 一個分隔 (0~500, 500~1000, 1000~1500, ...),
    存檔至該 npy 檔所在資料夾下的 output.jpg。

    回傳存檔路徑。
    """
    arr = np.load(npy_path)
    band_names = BANDS[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        data = arr[i].ravel()
        values, counts = np.unique(data, return_counts=True)

        nodata_idx = np.argmax(counts)
        nodata_value = values[nodata_idx]
        print(f"{band_names[i]}: 疑似 nodata 值 = {nodata_value} (出現 {counts[nodata_idx]} 次)")

        ranges = [
            ("<0", data < 0),
            ("1~500", (data >= 1) & (data <= 500)),
            ("500~1000", (data > 500) & (data <= 1000)),
            ("1000~10000", (data > 1000) & (data <= 10000)),
            (">10000", data > 10000),
        ]
        counts_str = ", ".join(f"{label}: {mask.sum()}" for label, mask in ranges)
        print(f"{band_names[i]} 數值個數: {{{counts_str}}}")

        plot_data = data[data != nodata_value]
        bin_edges = np.arange(0, plot_data.max() + bin_width, bin_width)
        ax.hist(plot_data, bins=bin_edges)
        ax.set_title(band_names[i])
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.set_xlim(0, plot_data.max() + 100)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(npy_path), "output.jpg")
    fig.savefig(out_path)
    plt.close(fig)

    print(f"已存檔分布圖 -> {out_path}")
    return out_path


def plot_band_heatmaps(npy_path: str) -> str:
    r"""讀取一個 npy 檔 (shape (C, H, W)),畫出前 4 個 band (B03_Green、B04_Red、
    B8A_NIR、B11_SWIR) 的熱力圖 (2x2 排列)。

    每個 band 的 nodata 值(出現次數最多的數值)以黑色顯示,
    存檔至該 npy 檔所在資料夾下的 heatmap.jpg。

    回傳存檔路徑。
    """
    arr = np.load(npy_path)
    band_names = BANDS[:4]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        band = arr[i]
        values, counts = np.unique(band, return_counts=True)
        nodata_value = values[np.argmax(counts)]

        masked = np.ma.masked_where(band == nodata_value, band)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="black")

        im = ax.imshow(masked, cmap=cmap)
        ax.set_title(band_names[i])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(npy_path), "heatmap.jpg")
    fig.savefig(out_path)
    plt.close(fig)

    print(f"已存檔熱力圖 -> {out_path}")
    return out_path


CLOUD_SCL_VALUES = [3, 8, 9]


def print_band_minmax(xy_root: str) -> str:
    r"""掃描 xy_root/X 底下所有 {date}/{region}.npy,
    計算每個檔案前 4 個 band (Green、Red、NIR、SWIR) 的最大最小值、
    SCL band (index 4) 中雲汙染 (值為 3、8、9) 佔有效像素 (排除 nodata) 的比例,
    以及整體 nodata (999) 比例。

    結果輸出成 CSV 檔 xy_root/X/maxmin.csv,欄位為:
        日期、區塊、Green_min、Green_max、Red_min、Red_max、NIR_min、NIR_max、
        SWIR_min、SWIR_max、(SCL in 3, 8, 9)%、nodata%
    並回傳該檔案路徑。
    """
    x_dir = os.path.join(xy_root, "X")
    short_names = ["Green", "Red", "NIR", "SWIR"]
    out_path = os.path.join(x_dir, "maxmin.csv")

    date_dirs = sorted(
        name for name in os.listdir(x_dir) if os.path.isdir(os.path.join(x_dir, name))
    )

    header = ["日期", "區塊"]
    for name in short_names:
        header += [f"{name}_min", f"{name}_max"]
    header += ["(SCL in 3, 8, 9)%", "nodata%"]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for date_str in date_dirs:
            date_dir = os.path.join(x_dir, date_str)
            region_files = sorted(
                name for name in os.listdir(date_dir) if name.endswith(".npy")
            )

            for region_file in region_files:
                region = region_file[: -len(".npy")]
                arr = np.load(os.path.join(date_dir, region_file))  # shape (C, h, w)

                valid_mask = arr[0] != 999  # 以 Green band 判斷有效範圍 (排除 nodata)

                row = [date_str, region]
                for i in range(4):
                    band = arr[i]
                    valid = band[band != 999]
                    if valid.size:
                        band_min, band_max = int(valid.min()), int(valid.max())
                    else:
                        band_min, band_max = None, None
                    row += [band_min, band_max]

                valid_count = int(valid_mask.sum())
                if valid_count:
                    cloud_count = int(np.isin(arr[4][valid_mask], CLOUD_SCL_VALUES).sum())
                    cloud_pct = round(cloud_count / valid_count * 100)
                else:
                    cloud_pct = None
                nodata_pct = round((~valid_mask).sum() / valid_mask.size * 100)
                row += [cloud_pct, nodata_pct]

                print(f"X/{date_str}/{region} : {row}")
                writer.writerow(row)

                del arr

    print(f"已存檔 -> {out_path}")
    return out_path


def sync_nodata_to_reference(
    xy_root: str,
    reference_date: str = "20230130",
    nodata_value: int = 999,
    updated_value: int = 65535,
    dry_run: bool = True,
) -> str:
    r"""以 y/{reference_date} 底下每個區塊的 nodata_value (預設 999) 像素位置作為基準,
    檢查其餘每個 y/{日期} 與每個 X/{日期} 底下相同區塊的 999 像素位置是否與基準不一致,
    並將 X 中與基準「交集」的 999 (即符合基準邊界的正常 nodata) 改成 updated_value
    (預設 65535),藉此把「正常邊界 nodata」與「與基準不符的異常 999」區分開來。

    比對邏輯 (以基準為準):
        多了 = 此檔為 999,但基準不是 999 的像素數
        少了 = 基準為 999,但此檔不是 999 的像素數
    不論有沒有差集都會印出訊息。

    y 只做檢查、印出差集訊息,不修改檔案 (排除 reference_date 自己,自己比自己無意義)。
    X 逐 band 分別與基準比較並印出差集訊息;該 band 與基準的「交集」像素
    (band == nodata_value 且 基準 == nodata_value) 改成 updated_value,
    修改後覆寫回原檔案 (含 reference_date 自己的 X,用來確認當天 X、y 是否一致)。

    Args:
        xy_root:         xy 資料夾根目錄 (含 X/、y/ 子目錄)
        reference_date:  基準日期,取該日期下 y 每個區塊的 999 位置作為標準邊界
        nodata_value:    無效像素值 (預設 999)
        updated_value:   X 中與基準一致的 999 要改成的值 (預設 65535)
        dry_run:         True (預設) 只印出訊息、不寫檔,用於預覽;
                          確認無誤後改成 False 才會真的覆寫 X 的 npy 檔案 (不可逆,請自行備份)

    Returns:
        差集報告 CSV 的路徑 (xy_root/nodata_sync_report.csv)
    """
    x_dir = os.path.join(xy_root, "X")
    y_dir = os.path.join(xy_root, "y")
    ref_dir = os.path.join(y_dir, reference_date)
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"找不到基準日期資料夾: {ref_dir}")

    if dry_run:
        print("── dry_run=True,僅預覽差集與預計修改的像素數,不會寫檔 ──")

    ref_regions = sorted(n[: -len(".npy")] for n in os.listdir(ref_dir) if n.endswith(".npy"))
    ref_masks = {}
    for region in ref_regions:
        y_arr = np.load(os.path.join(ref_dir, f"{region}.npy"))
        ref_masks[region] = (y_arr[0] if y_arr.ndim == 3 else y_arr) == nodata_value

    out_path = os.path.join(xy_root, "nodata_sync_report.csv")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["來源", "日期", "區塊", "Band", "基準_999", "此檔_999", "多了(此檔有基準無)", "少了(基準有此檔無)", "改成65535個數"]
        )

        # ── y: 只檢查,不修改 ──
        y_dates = sorted(
            name for name in os.listdir(y_dir)
            if os.path.isdir(os.path.join(y_dir, name)) and name != reference_date
        )
        for date_str in y_dates:
            date_dir = os.path.join(y_dir, date_str)
            regions = set(n[: -len(".npy")] for n in os.listdir(date_dir) if n.endswith(".npy"))

            for missing_region in sorted(set(ref_regions) - regions):
                print(f"[警告] y/{date_str} 缺少區塊 {missing_region},跳過")

            for region in ref_regions:
                if region not in regions:
                    continue
                path = os.path.join(date_dir, f"{region}.npy")
                y_arr = np.load(path)
                mask = (y_arr[0] if y_arr.ndim == 3 else y_arr) == nodata_value
                ref_mask = ref_masks[region]

                if mask.shape != ref_mask.shape:
                    print(f"[警告] 形狀不一致,跳過: {path}  基準={ref_mask.shape}  此檔={mask.shape}")
                    continue

                extra = int(np.sum(mask & ~ref_mask))
                missing = int(np.sum(ref_mask & ~mask))
                tag = "差集" if (extra or missing) else "一致"
                print(f"[{tag}] y/{date_str}/{region} : 多了={extra}  少了={missing}")

                writer.writerow(["y", date_str, region, "-", int(ref_mask.sum()), int(mask.sum()), extra, missing, "-"])

                del y_arr

        # ── X: 檢查 + 依交集改成 updated_value ──
        x_dates = sorted(
            name for name in os.listdir(x_dir) if os.path.isdir(os.path.join(x_dir, name))
        )
        for date_str in x_dates:
            date_dir = os.path.join(x_dir, date_str)
            regions = set(n[: -len(".npy")] for n in os.listdir(date_dir) if n.endswith(".npy"))

            for missing_region in sorted(set(ref_regions) - regions):
                print(f"[警告] X/{date_str} 缺少區塊 {missing_region},跳過")

            for region in ref_regions:
                if region not in regions:
                    continue
                path = os.path.join(date_dir, f"{region}.npy")
                x_arr = np.load(path)
                ref_mask = ref_masks[region]

                if x_arr.shape[-2:] != ref_mask.shape:
                    print(f"[警告] 形狀不一致,跳過: {path}  基準={ref_mask.shape}  此檔={x_arr.shape[-2:]}")
                    continue

                changed = False
                for i in range(x_arr.shape[0]):
                    band_name = BANDS[i] if i < len(BANDS) else f"band{i}"
                    band_mask = x_arr[i] == nodata_value

                    extra = int(np.sum(band_mask & ~ref_mask))
                    missing = int(np.sum(ref_mask & ~band_mask))
                    tag = "差集" if (extra or missing) else "一致"
                    print(f"[{tag}] X/{date_str}/{region}  {band_name} : 多了={extra}  少了={missing}")

                    intersection_mask = band_mask & ref_mask
                    n_updated = int(intersection_mask.sum())
                    if n_updated and not dry_run:
                        x_arr[i][intersection_mask] = updated_value
                        changed = True

                    writer.writerow(
                        ["X", date_str, region, band_name,
                         int(ref_mask.sum()), int(band_mask.sum()), extra, missing, n_updated]
                    )

                if changed:
                    np.save(path, x_arr)
                    print(f"  ✓ 已更新並覆寫: {path}")

                del x_arr

    print(f"已存檔差集報告 -> {out_path}")
    if dry_run:
        print("── dry_run=True,未寫檔;確認報告無誤後將 dry_run 設為 False 才會真的覆寫 X ──")
    return out_path


def scl_water_confusion_matrix(xy_root: str, water_scl_value: int = 6, nodata_value: int = 999) -> dict:
    r"""取出 X、y 資料夾中日期相同的所有資料,以「SCL=water_scl_value (預設 6,水體)
    且屬於農地」(X 第 5 個 band、index 4) 作為預測,對照 y 的灌水標籤 (1=有灌溉),
    只統計農地範圍內 (以 y != nodata_value 表示農地) 的像素,做混淆矩陣。

    Returns:
        dict,包含累計像素數 tp、fp、fn、tn。
    """
    x_dir = os.path.join(xy_root, "X")
    y_dir = os.path.join(xy_root, "y")

    x_dates = set(n for n in os.listdir(x_dir) if os.path.isdir(os.path.join(x_dir, n)))
    y_dates = set(n for n in os.listdir(y_dir) if os.path.isdir(os.path.join(y_dir, n)))
    dates = sorted(x_dates & y_dates)

    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for date_str in dates:
        x_date_dir = os.path.join(x_dir, date_str)
        y_date_dir = os.path.join(y_dir, date_str)

        x_regions = set(n[: -len(".npy")] for n in os.listdir(x_date_dir) if n.endswith(".npy"))
        y_regions = set(n[: -len(".npy")] for n in os.listdir(y_date_dir) if n.endswith(".npy"))

        for region in sorted(x_regions & y_regions):
            scl = np.load(os.path.join(x_date_dir, f"{region}.npy"))[4]
            y = np.load(os.path.join(y_date_dir, f"{region}.npy"))[0]

            farmland = y != nodata_value
            pred = scl[farmland] == water_scl_value
            actual = y[farmland] == 1

            cm["tp"] += int(np.sum(pred & actual))
            cm["fp"] += int(np.sum(pred & ~actual))
            cm["fn"] += int(np.sum(~pred & actual))
            cm["tn"] += int(np.sum(~pred & ~actual))

    total = sum(cm.values())
    precision = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    accuracy = (cm["tp"] + cm["tn"]) / total if total else 0.0

    print(f"混淆矩陣 (預測=SCL{water_scl_value}且屬於農地, 實際=y 灌水標籤, 僅農地範圍)")
    print(f"              實際=1        實際=0")
    print(f"  預測=1    TP={cm['tp']:>10}  FP={cm['fp']:>10}")
    print(f"  預測=0    FN={cm['fn']:>10}  TN={cm['tn']:>10}")
    print(f"  precision={precision:.4f}  recall={recall:.4f}  accuracy={accuracy:.4f}")

    return cm


def _crop(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """裁切 (..., H, W) 陣列的最後兩軸。範圍超出邊界時裁到邊界內;範圍完全落在邊界外則回傳 None。"""
    h, w = arr.shape[-2], arr.shape[-1]
    xx1, yy1, xx2, yy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if xx2 <= xx1 or yy2 <= yy1:
        return None
    return arr[..., yy1:yy2, xx1:xx2]


def cut_regions(xy_root: str) -> dict:
    r"""將 xy_root 下 X、y 資料夾中所有 {image_date}.npy 依 REGIONS 切成 11 個區塊 (N1~W5)。

    每個區塊分別存到:
        X\{image_date}\{區塊名}.npy   (原始 shape (C, H, W) 裁切後 (C, h, w))
        y\{image_date}\{區塊名}.npy   (原始 shape (1, H, W) 裁切後 (1, h, w))

    並在 xy_root 下存一份 index.json,記錄每個區塊名對應的原始座標範圍、
    是否有效 (valid,裁切範圍是否落在資料邊界內) 與裁切後大小 (size)。
    區塊的座標範圍與資料網格是固定的,因此 index.json 只記錄一次 (不分日期)。
    """
    index = {}

    for subdir in ("X", "y"):
        folder = os.path.join(xy_root, subdir)
        if not os.path.isdir(folder):
            print(f"略過不存在的資料夾: {folder}")
            continue

        date_files = sorted(
            m.group(1)
            for name in os.listdir(folder)
            if (m := re.fullmatch(r"(\d{8})\.npy", name))
        )

        total = len(date_files)
        for i, date_str in enumerate(date_files, start=1):
            print(f"[{subdir}][{i}/{total}] 處理中 image_date={date_str} ...")

            arr = np.load(os.path.join(folder, f"{date_str}.npy"))  # X: (C,H,W) / y: (1,H,W)

            out_dir = os.path.join(folder, date_str)
            os.makedirs(out_dir, exist_ok=True)

            for region, (x1, y1, x2, y2) in REGIONS.items():
                crop = _crop(arr, x1, y1, x2, y2)
                valid = crop is not None

                if region not in index:
                    index[region] = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "valid": valid,
                        "size": list(crop.shape[-2:]) if valid else None,
                    }

                if not valid:
                    print(f"  [警告] {subdir}/{date_str}/{region} 裁切範圍為空,略過")
                    continue

                np.save(os.path.join(out_dir, f"{region}.npy"), crop)

            del arr

    index_path = os.path.join(xy_root, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"已存 index.json -> {index_path}  (共 {len(index)} 個區塊)")

    return index


def delete_flat_npy(xy_root: str) -> List[str]:
    r"""刪除 xy_root 下 X、y 資料夾中「直接」存在的 {image_date}.npy 檔案
    (即 nc_to_npy / tif_to_npy / tif_to_npy_target 存的原始扁平檔)。

    只用 os.listdir 掃描 X、y 的第一層,並檢查 os.path.isfile,
    不會進到 cut_regions 產生的 X\{date}\ 或 y\{date}\ 子資料夾裡刪任何區塊 .npy。

    回傳實際刪除的檔案路徑列表。
    """
    deleted = []

    for subdir in ("X", "y"):
        folder = os.path.join(xy_root, subdir)
        if not os.path.isdir(folder):
            print(f"略過不存在的資料夾: {folder}")
            continue

        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and re.fullmatch(r"\d{8}\.npy", name):
                os.remove(path)
                deleted.append(path)
                print(f"已刪除: {path}")

    print(f"共刪除 {len(deleted)} 個檔案")
    return deleted

def train_cut_info(train_dates, patch_size=256, stride=256):
    r"""對 y 資料夾中「資料夾名稱在 train_dates 內」的每個日期,
    讀取該日期下 cut_regions 已切好的每個區塊 .npy(y\{date}\{region}.npy),
    再依 patch_size、stride 用滑動視窗進一步切成訓練用的小 patch。

    這裡不另外存 patch 的 npy 檔,只記錄每個 patch 的:
        date (日期)、region (區塊名)、y0/x0 (patch 在該區塊裡的起始像素位置)、
        irrigation_ratio (該 patch 內排除 999 之後,剩餘像素中數值為 1 的比例;
        若整個 patch 都是 999,則為 None)

    只保留右下邊界內的完整 patch (捨棄除不盡 stride 的邊緣殘餘)。

    輸出 json 檔到:
        D:\研究一所\嘉璯\S2全台資料_Xy對應\train\{patch_size}_{stride}_info.json
    """
    xy_root = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"
    y_dir = os.path.join(xy_root, "y")

    info = []
    for date_str in train_dates:
        date_dir = os.path.join(y_dir, date_str)
        if not os.path.isdir(date_dir):
            print(f"略過不存在的資料夾: {date_dir}")
            continue

        region_files = sorted(
            name for name in os.listdir(date_dir) if name.endswith(".npy")
        )

        for region_file in region_files:
            region = region_file[:-len(".npy")]
            arr = np.load(os.path.join(date_dir, region_file))  # shape (1, h, w)
            _, h, w = arr.shape

            n_patches = 0
            for y0 in range(0, h - patch_size + 1, stride):
                for x0 in range(0, w - patch_size + 1, stride):
                    patch = arr[:, y0 : y0 + patch_size, x0 : x0 + patch_size]

                    valid_mask = patch != 999
                    valid_count = int(valid_mask.sum())
                    if valid_count == 0:
                        irrigation_ratio = None
                    else:
                        irrigation_ratio = float(np.sum(patch[valid_mask] == 1) / valid_count)

                    info.append(
                        {
                            "date": date_str,
                            "region": region,
                            "y0": y0,
                            "x0": x0,
                            "irrigation_ratio": irrigation_ratio,
                        }
                    )
                    n_patches += 1

            print(f"{date_str}/{region}: shape={arr.shape} -> {n_patches} 個 patch")
            del arr

    out_path = os.path.join(xy_root, "train", f"{patch_size}_{stride}_info.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"共記錄 {len(info)} 個 patch,已存 -> {out_path}")

    return info


def test_val_cut_info(test_dates, val_dates, patch_size=256):
    r"""對 y 資料夾中「資料夾名稱在 test_dates 或 val_dates 內」的每個日期,
    讀取該日期下 cut_regions 已切好的每個區塊 .npy(y\{date}\{區塊名}.npy),
    以 patch_size 做不重疊 (stride = patch_size) 的滑動視窗切割。

    因為後續要用這份 info 重組回原圖,邊緣除不盡 patch_size 的地方**不能捨棄**,
    而是保留該 patch、並記錄其中實際有資料的範圍 (valid_h, valid_w);
    實際載入該 patch 時只要讀到邊界為止,不足 patch_size 的部分再另外補 padding。

    每筆記錄: date、region、region 原始大小 (shape = [h, w])、
    patch 起始像素位置 (y0, x0)、patch 內有效(未 padding)的大小 (valid_h, valid_w)。

    與 train_cut_info 不同:這裡不計算 irrigation_ratio,
    且每個 (date, region) 各自存一份 json(而非全部合併成一份),輸出到:
        D:\研究一所\嘉璯\S2全台資料_Xy對應\test\{date}_{區塊名}_{patch_size}_info.json
        D:\研究一所\嘉璯\S2全台資料_Xy對應\val\{date}_{區塊名}_{patch_size}_info.json
    """
    xy_root = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"
    y_dir = os.path.join(xy_root, "y")

    result = {}
    for subdir_name, dates in (("test", test_dates), ("val", val_dates)):
        out_dir = os.path.join(xy_root, subdir_name)
        os.makedirs(out_dir, exist_ok=True)

        for date_str in dates:
            date_dir = os.path.join(y_dir, date_str)
            if not os.path.isdir(date_dir):
                print(f"略過不存在的資料夾: {date_dir}")
                continue

            region_files = sorted(
                name for name in os.listdir(date_dir) if name.endswith(".npy")
            )

            for region_file in region_files:
                region = region_file[: -len(".npy")]
                arr = np.load(os.path.join(date_dir, region_file))  # shape (1, h, w)
                _, h, w = arr.shape

                patches = []
                for y0 in range(0, h, patch_size):
                    for x0 in range(0, w, patch_size):
                        valid_h = min(patch_size, h - y0)
                        valid_w = min(patch_size, w - x0)
                        patches.append(
                            {
                                "date": date_str,
                                "region": region,
                                "shape": [h, w],
                                "y0": y0,
                                "x0": x0,
                                "valid_h": valid_h,
                                "valid_w": valid_w,
                            }
                        )

                out_path = os.path.join(
                    out_dir, f"{date_str}_{region}_{patch_size}_info.json"
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(patches, f, ensure_ascii=False, indent=2)

                print(
                    f"[{subdir_name}] {date_str}/{region}: shape={arr.shape} "
                    f"-> {len(patches)} 個 patch -> {out_path}"
                )
                result[out_path] = patches

                del arr

    return result


if __name__ == "__main__":

    # nc_file = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\2026年1期作全台\S2特徵合併_原波長版_nday20.nc"
    # npy_root = r"D:\研究一所\嘉璯\S2全台資料_Xy對應\X\{image_date}"
    # saved_dates, saved_paths = nc_to_npy(nc_file, npy_root)
    # print(f"共處理 {len(saved_dates)} 個不重複的 image_date")

    # tif_file = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\S2訓練資料\2023年一期作\2023全台\S2特徵合併_原波長版_nday20.tif"
    # tif_txt = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\S2訓練資料\2023年一期作\2023全台\S2_原波長_Band說明.txt"
    # npy_root_tif = r"D:\研究一所\嘉璯\S2全台資料_Xy對應\X\{image_date}"
    # saved_dates_tif, saved_paths_tif = tif_to_npy(tif_file, tif_txt, npy_root_tif)
    # print(f"共處理 {len(saved_dates_tif)} 個不重複的 image_date")

    # tif_file_y = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\S2訓練資料\2023年一期作\2023全台\灌水網格合併_含弱相關.tif"
    # tif_txt_y = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\S2訓練資料\2023年一期作\2023全台\灌水網格合併Band說明.txt"
    # npy_root_y = r"D:\研究一所\嘉璯\S2全台資料_Xy對應\y\{image_date}"
    # saved_dates_y, saved_paths_y = tif_to_npy_target(tif_file_y, tif_txt_y, npy_root_y)
    # print(f"共處理 {len(saved_dates_y)} 個不重複的 image_date")

    # tif_file_y = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\2026年1期作全台\灌水網格合併_含弱相關.tif"
    # tif_txt_y = r"Z:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\2026年1期作全台\灌水網格合併Band說明_含弱相關.txt"
    # npy_root_y = r"D:\研究一所\嘉璯\S2全台資料_Xy對應\y\{image_date}"
    # saved_dates_y, saved_paths_y = tif_to_npy_target(tif_file_y, tif_txt_y, npy_root_y)
    # print(f"共處理 {len(saved_dates_y)} 個不重複的 image_date")

    # xy_root=r"D:\研究一所\嘉璯\S2全台資料_Xy對應"
    # cut_regions(xy_root)
    # delete_flat_npy(xy_root)

    
    # plot_band_distributions(npy_path=r"D:\研究一所\嘉璯\S2全台資料_Xy對應\X\20230110\E2.npy")
    # plot_band_heatmaps(npy_path=r"D:\研究一所\嘉璯\S2全台資料_Xy對應\X\20230110\E2.npy")
    # print_band_minmax(xy_root=r"D:\研究一所\嘉璯\S2全台資料_Xy對應")
    # sync_nodata_to_reference(xy_root=r"D:\研究一所\嘉璯\S2全台資料_Xy對應", reference_date="20230130", dry_run=True)
    scl_water_confusion_matrix(xy_root=r"D:\研究一所\嘉璯\S2全台資料_Xy對應")
