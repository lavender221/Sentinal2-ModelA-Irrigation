"""
Dataset.py

改寫自 model_v1_base.ipynb 的 OneSliceDataset。

切割步驟需由外部先呼叫 data_cut.py 的 cut_xy_patches() 完成
（產生 cutV1/train_info.json、valid_info.json、testing 的 test_info.json 其中之一），
宣告時直接傳入該 json 檔路徑（info_json），
__init__ 只負責建立 patch 索引供訓練使用，不會觸發任何切割 I/O。

讀取方式（train/valid/test 一律相同）：
    依 json 內各 (date, region) item 的座標，用 np.load(..., mmap_mode="r")
    從 xy_dir 原始大圖即時切出 patch（超出原圖邊界的 padding 區域：
    y 補 nodata_value、X 補 0），不讀取任何 patch .npy，必須傳入 xy_dir。

train 專屬篩選（依 train_info.json 中切割時算好的比例欄位，於建索引時剔除）：
    filter_all_nodata:    剔除整張皆為 nodata 的 patch（nodata_ratio == 1）
    filter_no_water:      剔除有效像素中無水體的 patch（water_ratio == 0）
    cloud_rate_threshold: 剔除 scl_cloud_ratio 超過閾值的 patch
    no_water_keep_ratio:  無灌水 patch 下採樣——含灌水像素（water_ratio > 0）的
                          patch（n 張）全數保留，water_ratio == 0 的 patch 只隨機
                          保留 n × ratio 張（如 0.25 → n/4；固定 seed 可重現）
    valid/test 的 json 不含比例欄位，這些參數不會有作用。

X patch 格式（來自 xy_dir 的 X 大圖）：(5, P, P)
    [0] Green, [1] Red, [2] NIR, [3] SWIR, [4] SCL

可選變數（features 參數，共 8 種）：
    連續型：Green、Red、NIR、SWIR（各 1 通道，可 Z-score 標準化）
    SCL：one-hot 編碼（scl_num_classes + 1 通道，預設 12+1=13；
         最後一類放 nodata 與超出範圍的異常值如 65535）
    衍生指數：NDVI、NDWI、LSWI（各 1 通道，由原始未標準化值計算）
        NDVI = (NIR - Red) / (NIR + Red + ε)
        NDWI = (Green - NIR) / (Green + NIR + ε)
        LSWI = (NIR - SWIR) / (NIR + SWIR + ε)

固定附加通道（不可省略，永遠是最後一個通道）：
    時間差通道：1 通道，(y 日期 − X 日期) 天數差 / nday 的常數平面
    （反映資料新鮮度；nday 為 X 資料視窗上限，預設 20）

連續型波段的離群值裁切（clip）邊界：
    預設為 [0, 10000]；建議改傳入 Preprocess.compute_band_percentiles()
    在 train_dates 上算出的 1st~99th 百分位數（clip_min / clip_max），
    避免使用寫死邊界，同時確保 val/test 沿用 train 的統計量、不洩漏資訊。

使用方式：
    from src.preprocessing.data_cut import cut_xy_patches

    train_info = cut_xy_patches(xy_dir=r"...\\xy對應", mode="train", dates=[...], ...)
    train_dataset = OneSliceDataset(
        info_json=train_info,
        dates=["20230130", "20230219", "20230301"],
        features=["Green", "Red", "NIR", "SWIR", "SCL", "NDVI", "NDWI", "LSWI"],
        xy_dir=r"...\\xy對應",
        filter_all_nodata=True,   # train 篩選（依 json 比例欄位）
    )
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# 所有可選變數名稱（順序固定，供外部查閱通道數用）
VALID_FEATURES: List[str] = ["Green", "Red", "NIR", "SWIR", "SCL", "NDVI", "NDWI", "LSWI"]

# SCL 展開為 scl_num_classes 個通道，其餘各 1 通道
_FEATURE_CHANNELS = {f: 1 for f in VALID_FEATURES}
_FEATURE_CHANNELS["SCL"] = None  # 由 scl_num_classes 決定，執行期填入


class OneSliceDataset(Dataset):
    """
    讀取 data_cut.py 產生的 {mode}_info.json，依座標從 xy_dir 原始大圖即時切割 patch。

    切割步驟需由外部先呼叫 cut_xy_patches() 完成；宣告時直接傳入該 json 檔路徑
    （info_json），只負責建立 patch 索引，不會觸發任何切割 I/O。

    篩選後保留的 patch 對應回 {mode}_info.json 的 idx：
        - 每個 self.samples 條目含 "idx" 欄位（該 patch 在 json 中的 idx）
        - self.kept_indices：{date: {region: [idx, ...]}}，彙整各 (date, region)
          篩選後保留下來的 patch idx
    """

    def __init__(
        self,
        info_json: str,
        dates: List[str] = None,
        features: List[str] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        clip_min: Optional[np.ndarray] = None,
        clip_max: Optional[np.ndarray] = None,
        scl_num_classes: int = 12,
        nodata_value: int = 999,
        xy_dir: str = None,
        filter_all_nodata: bool = False,
        filter_no_water: bool = False,
        cloud_rate_threshold: Optional[float] = None,
        no_water_keep_ratio: Optional[float] = None,
        seed: int = 42,
        nday: int = 20,
    ):
        """
        Args:
            info_json:             cut_xy_patches() 輸出的 json 檔路徑
                                   （cutV1/train_info.json 等，即其回傳值）
            dates:                 要處理的日期列表；None 表示全部
            features:              要回傳的變數清單，可包含：
                                   "Green", "Red", "NIR", "SWIR", "SCL",
                                   "NDVI", "NDWI", "LSWI"
                                   None 表示全選（預設）
            mean:                  連續型波段的均值（shape (4,)），用於 Z-score 標準化
                                   順序固定為 [Green, Red, NIR, SWIR]
            std:                   連續型波段的標準差（shape (4,)）
            clip_min:              連續型波段裁切下界（shape (4,)），None 時退回 0
                                   （建議傳入 Preprocess.compute_band_percentiles() 的
                                   回傳值，僅用 train_dates 計算的 1st 百分位數）
            clip_max:              連續型波段裁切上界（shape (4,)），None 時退回 10000
                                   （對應 99th 百分位數）
            scl_num_classes:       SCL 真實類別數（預設 12，Sentinel-2 SCL 0~11）；
                                   one-hot 會額外增加一類放 nodata 與超出範圍的
                                   異常值（如 65535），輸出 scl_num_classes + 1 通道
            nodata_value:          無效像素值（預設 999）
            xy_dir:                原始大圖來源資料夾（xy對應/，含 y/ 與 X/ 子目錄），
                                   必填，供即時切割使用
            filter_all_nodata:     True=剔除整張皆為 nodata 的 patch
                                   （nodata_ratio == 1；僅 train json 有比例欄位）
            filter_no_water:       True=剔除有效像素中無水體的 patch（water_ratio == 0）
            cloud_rate_threshold:  雲遮蔽率閾值（0–1）；scl_cloud_ratio 超過則剔除；
                                   None=不篩選
            no_water_keep_ratio:   無灌水 patch 下採樣比例；含灌水像素的 patch
                                   （n 張）全數保留，water_ratio == 0 的 patch 只
                                   隨機保留 n × ratio 張（如 0.25 → n/4）；
                                   None=不下採樣（全部保留）。於其他篩選
                                   （filter_* / cloud）之後套用
            seed:                  無灌水 patch 抽樣的隨機種子（固定可重現）
            nday:                  時間差通道的正規化分母（X 資料視窗上限，
                                   nday20 → 20），天數差 / nday 後約落在 0~1
        """
        self.info_path = Path(info_json)
        self.scl_num_classes = scl_num_classes
        self.nodata_value = nodata_value
        self.dates = dates
        if xy_dir is None:
            raise ValueError("必須傳入 xy_dir（原始大圖來源資料夾）供即時切割使用")
        self.xy_dir = Path(xy_dir)
        self.filter_all_nodata = filter_all_nodata
        self.filter_no_water = filter_no_water
        self.cloud_rate_threshold = cloud_rate_threshold
        self.no_water_keep_ratio = no_water_keep_ratio
        self.seed = seed
        self.nday = nday
        self._mmap_cache: Dict[str, np.ndarray] = {}

        self.clip_min = np.asarray(clip_min, dtype=np.float32) if clip_min is not None else np.zeros(4, dtype=np.float32)
        self.clip_max = np.asarray(clip_max, dtype=np.float32) if clip_max is not None else np.full(4, 10000.0, dtype=np.float32)

        # ── 讀取切割資訊 json ─────────────────────────────────────────────────
        with open(self.info_path, encoding="utf-8") as f:
            info = json.load(f)
        self.mode = info.get("mode")
        self.patch_size = info["patch_size"]
        self.items: List[Dict] = info["items"]

        # ── 變數選擇 ─────────────────────────────────────────────────────────
        if features is None:
            self.features = list(VALID_FEATURES)
        else:
            invalid = [f for f in features if f not in VALID_FEATURES]
            if invalid:
                raise ValueError(f"不支援的變數：{invalid}，可選：{VALID_FEATURES}")
            self.features = list(features)

        # 計算輸出通道數，供外部查閱（時間差通道固定 +1，不可省略；
        # SCL 為 scl_num_classes + 1 通道，最後一類放 nodata/異常值）
        self.num_channels = sum(
            self.scl_num_classes + 1 if f == "SCL" else 1
            for f in self.features
        ) + 1  # 時間差通道永遠是最後一個通道

        if mean is not None and std is not None:
            self.mean = torch.from_numpy(mean.astype(np.float32)).view(-1, 1, 1)
            self.std = torch.from_numpy(std.astype(np.float32)).view(-1, 1, 1)
            self.normalize = True
        else:
            self.mean = self.std = None
            self.normalize = False

        # ── 建立 patch 索引 ──────────────────────────────────────────────────
        self.samples = self._build_samples_list()
        print(
            f"[OneSliceDataset] 完成，共 {len(self.samples)} 個 patch"
            f"（info_json={self.info_path}）"
        )

    # ── 內部方法 ──────────────────────────────────────────────────────────────

    def _keep_patch(self, meta: Dict) -> bool:
        """依 json 中的比例欄位判斷 patch 是否保留（欄位不存在時一律保留）。"""
        if self.filter_all_nodata and meta.get("nodata_ratio") == 1.0:
            return False
        if self.filter_no_water and meta.get("water_ratio") == 0.0:
            return False
        if self.cloud_rate_threshold is not None:
            scl_ratio = meta.get("scl_cloud_ratio")
            if scl_ratio is not None and scl_ratio > self.cloud_rate_threshold:
                return False
        return True

    def _build_samples_list(self) -> List[Dict]:
        dates_set = set(self.dates) if self.dates is not None else None
        n_filtered = 0
        # 篩選後保留的 patch 在 {mode}_info.json 中的 idx：{date: {region: [idx, ...]}}
        self.kept_indices: Dict[str, Dict[str, List[int]]] = {}

        # ── 第一階段：套用 filter_* / cloud 篩選，收集候選 patch ─────────────
        candidates: List[Tuple[Optional[float], Dict]] = []  # (water_ratio, sample)
        for item in self.items:
            date = item["date"]
            region = item["region"]
            if dates_set is not None and date not in dates_set:
                continue

            x_date = item.get("x_date", date)
            src_y_path = self.xy_dir / "y" / date / f"{region}.npy"
            src_x_path = self.xy_dir / "X" / x_date / f"{region}.npy"
            if not src_y_path.exists():
                print(f"[警告] 找不到原始 y/{date}/{region}.npy，跳過")
                continue
            if not src_x_path.exists():
                print(f"[警告] 找不到原始 X/{x_date}/{region}.npy，跳過")
                continue

            self.kept_indices.setdefault(date, {}).setdefault(region, [])

            # 時間差輔助特徵：y 日期與配對到的 X 日期的天數差（>= 0）
            day_diff = (
                datetime.strptime(date, "%Y%m%d") - datetime.strptime(x_date, "%Y%m%d")
            ).days

            for meta in item["patches"]:
                if not self._keep_patch(meta):
                    n_filtered += 1
                    continue
                candidates.append((meta.get("water_ratio"), {
                    "date": date,
                    "region": region,
                    "idx": meta["idx"],
                    "patch_name": f"patch_{meta['idx']:04d}.npy",
                    "row": meta["row"],
                    "col": meta["col"],
                    "patch_size": self.patch_size,
                    "day_diff": day_diff,
                    "src_y_path": src_y_path,
                    "src_x_path": src_x_path,
                }))

        if n_filtered > 0:
            print(f"[OneSliceDataset] 篩選剔除 {n_filtered} 個 patch"
                  f"（filter_all_nodata={self.filter_all_nodata}, "
                  f"filter_no_water={self.filter_no_water}, "
                  f"cloud_rate_threshold={self.cloud_rate_threshold}）")

        # ── 第二階段：無灌水 patch 下採樣（train json 才有 water_ratio）──────
        # 含灌水像素（water_ratio > 0）的 n 張全保留，
        # water_ratio == 0 的只隨機保留 n × no_water_keep_ratio 張
        if self.no_water_keep_ratio is not None:
            n_water = sum(1 for wr, _ in candidates if wr is not None and wr > 0)
            zero_pos = [i for i, (wr, _) in enumerate(candidates) if wr == 0.0]
            k = min(len(zero_pos), int(round(n_water * self.no_water_keep_ratio)))
            keep_zero = set(random.Random(self.seed).sample(zero_pos, k))
            candidates = [
                (wr, s) for i, (wr, s) in enumerate(candidates)
                if wr != 0.0 or i in keep_zero
            ]
            print(f"[OneSliceDataset] 無灌水 patch 下採樣：有灌水 {n_water} 張全保留，"
                  f"無灌水 {len(zero_pos)} 張隨機保留 {k} 張"
                  f"（ratio={self.no_water_keep_ratio}, seed={self.seed}）")

        # ── 依原始順序建立 samples 與 kept_indices ───────────────────────────
        samples = []
        for _, s in candidates:
            self.kept_indices[s["date"]][s["region"]].append(s["idx"])
            samples.append(s)
        return samples

    def __getstate__(self):
        """DataLoader num_workers > 0（Windows spawn）時 dataset 會被 pickle 傳給
        worker：排除 mmap 快取（pickle np.memmap 會把整張大圖實際讀入記憶體），
        worker 端首次讀取時會自行重建快取。"""
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state

    def _get_memmap(self, path: Path) -> np.ndarray:
        """以 mmap 開啟原始大圖並快取，切 patch 時只實際讀取視窗範圍的資料。"""
        key = str(path)
        if key not in self._mmap_cache:
            self._mmap_cache[key] = np.load(path, mmap_mode="r")
        return self._mmap_cache[key]

    def _load_patch_from_source(self, s: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """依座標從原始大圖即時切出 (y_patch, x_patch)。
        超出原圖邊界的 padding 區域依 data_cut.py 的規則補值：y 補 nodata_value、X 補 0。
        """
        P = s["patch_size"]
        r, c = s["row"], s["col"]

        y_src = self._get_memmap(s["src_y_path"])
        if y_src.ndim == 3:
            y_src = y_src[0]  # (1, H, W) → (H, W)
        x_src = self._get_memmap(s["src_x_path"])  # (5, H, W)

        H, W = y_src.shape
        r2, c2 = min(r + P, H), min(c + P, W)

        y_data = np.full((P, P), float(self.nodata_value), dtype=np.float32)
        y_data[: r2 - r, : c2 - c] = y_src[r:r2, c:c2]

        x_data = np.zeros((x_src.shape[0], P, P), dtype=np.float32)
        x_data[:, : r2 - r, : c2 - c] = x_src[:, r:r2, c:c2]
        return y_data, x_data

    # ── Dataset 介面 ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        # A + B. 即時切割 y patch (P, P) 與 X patch (5, P, P) = [Green, Red, NIR, SWIR, SCL]
        y_data, x_data = self._load_patch_from_source(s)

        y_nodata_mask = y_data == self.nodata_value            # (P, P) bool

        # 原始連續型波段，依 train_dates 統計出的百分位數裁切離群值（衍生指數從此計算）
        x_raw = np.stack(
            [np.clip(x_data[i], self.clip_min[i], self.clip_max[i]) for i in range(4)],
            axis=0,
        )  # (4, P, P)
        G, R, NIR, SWIR = x_raw[0], x_raw[1], x_raw[2], x_raw[3]

        # 連續型波段標準化（shape 保持 (4, P, P)，索引對應 Green/Red/NIR/SWIR）
        X_numeric = torch.tensor(x_raw, dtype=torch.float32)
        if self.normalize:
            X_numeric = (X_numeric - self.mean) / (self.std + 1e-6)

        # 衍生光譜指數（由原始未標準化波段計算，值域 [-1, 1]）
        ndvi = (NIR - R)    / (NIR + R    + 1e-6)
        ndwi = (G   - NIR)  / (G   + NIR  + 1e-6)
        lswi = (NIR - SWIR) / (NIR + SWIR + 1e-6)

        # SCL cloud mask（1=有效，0=雲影/中機率雲/高機率雲；使用原始 SCL，未套用 nodata 覆蓋）
        scl_cloud_mask = torch.tensor(
            (~np.isin(x_data[4], [3, 8, 9])).astype(np.float32)
        )  # (P, P)

        # SCL one-hot（僅在 features 含 "SCL" 時計算）：
        # 真實類別 0 ~ scl_num_classes-1 之外，額外增加一類（index = scl_num_classes）
        # 放 nodata 位置與超出範圍的異常值（如 X nodata 的 65535），
        # 共 scl_num_classes + 1 個通道
        scl_one_hot = None
        if "SCL" in self.features:
            scl = x_data[4].copy()
            scl[y_nodata_mask] = self.scl_num_classes
            scl[(scl < 0) | (scl >= self.scl_num_classes)] = self.scl_num_classes
            t_scl = torch.tensor(scl, dtype=torch.long)
            scl_one_hot = F.one_hot(t_scl, num_classes=self.scl_num_classes + 1).permute(2, 0, 1).float()
            # scl_one_hot: (scl_num_classes + 1, P, P)

        # C. 依 self.features 選取並拼接通道
        _band_idx = {"Green": 0, "Red": 1, "NIR": 2, "SWIR": 3}
        _indices   = {"NDVI": ndvi, "NDWI": ndwi, "LSWI": lswi}
        channels: List[torch.Tensor] = []
        for feat in self.features:
            if feat in _band_idx:
                i = _band_idx[feat]
                channels.append(X_numeric[i: i + 1])            # (1, P, P)
            elif feat == "SCL":
                channels.append(scl_one_hot)                    # (scl_num_classes, P, P)
            else:
                arr = _indices[feat][np.newaxis].astype(np.float32)
                channels.append(torch.tensor(arr, dtype=torch.float32))  # (1, P, P)

        # 時間差通道固定附加於最後一個通道：
        # (y 日期 − X 日期) 天數差 / nday 的常數平面（反映資料新鮮度）
        P = s["patch_size"]
        day_diff_ch = torch.full(
            (1, P, P), s["day_diff"] / self.nday, dtype=torch.float32
        )
        channels.append(day_diff_ch)

        X_final = torch.cat(channels, dim=0)                    # (num_channels, P, P)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)
        return X_final, y_tensor, scl_cloud_mask


# ── 主程式（示範宣告 + EDA 查看切割結果）────────────────────────────────────

if __name__ == "__main__":
    from src.preprocessing.data_cut import cut_xy_patches
    from src.dataset.eda import (
        compute_irrigation_cloud_cover_rate,
        count_zero_nodata_patches,
        plot_irrigation_cloud_distribution,
    )

    XY_DIR = r"D:\研究一所\嘉璯\S2全台資料_Xy對應"

    TRAIN_DATES = ["20230130", "20230219", "20230301"]
    VALID_DATES = ["20230306"]
    TEST_DATES  = ["20230316"]

    # ── 先切割 patch（train / valid / test，皆只輸出 json，不做篩選）─────────
    P = 256
    FEATURES = ["Green", "Red", "NIR", "SWIR", "SCL", "NDVI", "NDWI", "LSWI"]

    train_info = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TRAIN_DATES,
        mode="train",
        patch_size=P,
        stride=P,
        keep_remainder=False,
    )
    valid_info = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=VALID_DATES,
        mode="valid",
        patch_size=P,
    )
    test_info = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TEST_DATES,
        mode="test",
        patch_size=P,
    )

    # ── 宣告 Dataset（皆即時切割，train 的篩選在此進行）──────────────────────
    train_dataset = OneSliceDataset(
        info_json=train_info,
        dates=TRAIN_DATES,
        features=FEATURES,
        xy_dir=XY_DIR,
        filter_all_nodata=True,
    )

    valid_dataset = OneSliceDataset(
        info_json=valid_info,
        dates=VALID_DATES,
        features=FEATURES,
        xy_dir=XY_DIR,
    )

    test_dataset = OneSliceDataset(
        info_json=test_info,
        dates=TEST_DATES,
        features=FEATURES,
        xy_dir=XY_DIR,
    )

    print(f"\nTrain: {len(train_dataset)} patches")
    print(f"Valid: {len(valid_dataset)} patches")
    print(f"Test:  {len(test_dataset)} patches")

    # ── EDA：訓練資料整體統計（不分日期／區塊，直接用 json 的比例欄位；
    #    kept_indices → 只統計 Dataset 篩選／下採樣後保留的 patch）───────────
    print("\n══════════════════════════════════════")
    print("  Train patch 整體統計（篩選後）")
    print("══════════════════════════════════════")
    count_zero_nodata_patches(
        info_json=train_info, dates=TRAIN_DATES,
        kept_indices=train_dataset.kept_indices,
    )

    df = compute_irrigation_cloud_cover_rate(
        info_json=train_info,
        dates=TRAIN_DATES,
        kept_indices=train_dataset.kept_indices,
    )
    plot_irrigation_cloud_distribution(df)
