"""
Dataset.py

改寫自 model_v1_base.ipynb 的 OneSliceDataset。

切割步驟需由外部先呼叫 data_cut.py 的 cut_xy_patches() 完成
（產生 training_data/ 、valid_data/ 、testing_data/ 其中之一），
宣告時直接傳入該已切割好的資料夾（data_dir），
__init__ 只負責建立 patch 索引供訓練使用，不會觸發任何切割 I/O。

X patch 格式（來自 data_cut.py）：(5, P, P)
    [0] Green, [1] Red, [2] NIR, [3] SWIR, [4] SCL

可選變數（features 參數，共 8 種）：
    連續型：Green、Red、NIR、SWIR（各 1 通道，可 Z-score 標準化）
    SCL：one-hot 編碼（13 通道）
    衍生指數：NDVI、NDWI、LSWI（各 1 通道，由原始未標準化值計算）
        NDVI = (NIR - Red) / (NIR + Red + ε)
        NDWI = (Green - NIR) / (Green + NIR + ε)
        LSWI = (NIR - SWIR) / (NIR + SWIR + ε)

固定附加通道（不可省略，永遠是最後一個通道）：
    nodata mask：1 通道，1=nodata（y==nodata_value），0=有效像素

連續型波段的離群值裁切（clip）邊界：
    預設為 [0, 10000]；建議改傳入 Preprocess.compute_band_percentiles()
    在 train_dates 上算出的 1st~99th 百分位數（clip_min / clip_max），
    避免使用寫死邊界，同時確保 val/test 沿用 train 的統計量、不洩漏資訊。

使用方式：
    from src.preprocessing.data_cut import cut_xy_patches

    train_dir = cut_xy_patches(xy_dir=r"...\\xy對應", mode="train", dates=[...], ...)
    train_dataset = OneSliceDataset(
        data_dir=train_dir,
        dates=["20230130", "20230219", "20230301"],
        features=["Green", "Red", "NIR", "SWIR", "SCL", "NDVI", "NDWI", "LSWI"],
    )
"""

import os
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
    讀取 data_cut.py 產生的 training_data / testing_data / valid_data 格式 patch。

    切割步驟需由外部先呼叫 cut_xy_patches() 完成；宣告時直接傳入該資料夾（data_dir），
    只負責建立 patch 索引，不會觸發任何切割 I/O。
    """

    def __init__(
        self,
        data_dir: str,
        dates: List[str] = None,
        features: List[str] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        clip_min: Optional[np.ndarray] = None,
        clip_max: Optional[np.ndarray] = None,
        scl_num_classes: int = 13,
        nodata_value: int = 999,
    ):
        """
        Args:
            data_dir:              已切割完成的資料夾（cut_xy_patches() 的回傳路徑，
                                   即 training_data/ 、valid_data/ 、testing_data/ 其中之一）
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
            scl_num_classes:       SCL one-hot 的類別數（預設 13）
            nodata_value:          無效像素值（預設 999）
        """
        self.data_dir = Path(data_dir)
        self.scl_num_classes = scl_num_classes
        self.nodata_value = nodata_value
        self.dates = dates

        self.clip_min = np.asarray(clip_min, dtype=np.float32) if clip_min is not None else np.zeros(4, dtype=np.float32)
        self.clip_max = np.asarray(clip_max, dtype=np.float32) if clip_max is not None else np.full(4, 10000.0, dtype=np.float32)

        # ── 變數選擇 ─────────────────────────────────────────────────────────
        if features is None:
            self.features = list(VALID_FEATURES)
        else:
            invalid = [f for f in features if f not in VALID_FEATURES]
            if invalid:
                raise ValueError(f"不支援的變數：{invalid}，可選：{VALID_FEATURES}")
            self.features = list(features)

        # 計算輸出通道數，供外部查閱（nodata mask 固定 +1，不可省略）
        self.num_channels = sum(
            self.scl_num_classes if f == "SCL" else 1
            for f in self.features
        ) + 1  # nodata mask 永遠是最後一個通道

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
            f"（data_dir={self.data_dir}）"
        )

    # ── 內部方法 ──────────────────────────────────────────────────────────────

    def _build_samples_list(self) -> List[Dict]:
        samples = []
        y_root = self.data_dir / "y"

        if self.dates is not None:
            date_dirs = [y_root / d for d in self.dates if (y_root / d).is_dir()]
        else:
            date_dirs = sorted(p for p in y_root.iterdir() if p.is_dir())

        for date_dir in date_dirs:
            date = date_dir.name
            region_dirs = sorted(p for p in date_dir.iterdir() if p.is_dir())
            for region_dir in region_dirs:
                region = region_dir.name
                for y_path in sorted(region_dir.glob("patch_*.npy")):
                    x_path = self.data_dir / "X" / date / region / y_path.name
                    if not x_path.exists():
                        continue
                    samples.append({
                        "date": date,
                        "region": region,
                        "patch_name": y_path.name,
                        "y_path": y_path,
                        "x_path": x_path,
                    })
        return samples

    # ── Dataset 介面 ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        # A. Y（地面真值）
        y_data = np.load(s["y_path"]).astype(np.float32)     # (P, P)
        if y_data.ndim == 3:
            y_data = y_data[0]
        y_nodata_mask = y_data == self.nodata_value            # (P, P) bool

        # B. X patch：(5, P, P) = [Green, Red, NIR, SWIR, SCL]
        x_data = np.load(s["x_path"]).astype(np.float32)     # (5, P, P)

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

        # SCL one-hot（nodata 位置強制設為最後一個類別）
        scl = x_data[4].copy()
        scl[y_nodata_mask] = self.scl_num_classes - 1
        t_scl = torch.tensor(scl, dtype=torch.long)
        scl_one_hot = F.one_hot(t_scl, num_classes=self.scl_num_classes).permute(2, 0, 1).float()
        # scl_one_hot: (scl_num_classes, P, P)

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

        # nodata mask 固定附加於最後一個通道（1=nodata，0=有效）
        nodata_ch = torch.tensor(
            y_nodata_mask[np.newaxis].astype(np.float32), dtype=torch.float32
        )                                                        # (1, P, P)
        channels.append(nodata_ch)

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

    # ── 先切割 patch（train / valid / test）────────────────────────────────
    P = 256
    FEATURES = ["Green", "Red", "NIR", "SWIR", "SCL", "NDVI", "NDWI", "LSWI"]

    train_dir = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TRAIN_DATES,
        mode="train",
        patch_size=P,
        stride_x=P,
        stride_y=P,
        keep_remainder=False,
        filter_all_nodata=True,
        filter_no_water=False,
    )
    valid_dir = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=VALID_DATES,
        mode="valid",
        patch_size=P,
    )
    test_dir = cut_xy_patches(
        xy_dir=XY_DIR,
        dates=TEST_DATES,
        mode="test",
        patch_size=P,
    )

    # ── 宣告 Dataset（只讀取已切割好的 patch）────────────────────────────────
    train_dataset = OneSliceDataset(
        data_dir=train_dir,
        dates=TRAIN_DATES,
        features=FEATURES,
    )

    valid_dataset = OneSliceDataset(
        data_dir=valid_dir,
        dates=VALID_DATES,
        features=FEATURES,
    )

    test_dataset = OneSliceDataset(
        data_dir=test_dir,
        dates=TEST_DATES,
        features=FEATURES,
    )

    print(f"\nTrain: {len(train_dataset)} patches")
    print(f"Valid: {len(valid_dataset)} patches")
    print(f"Test:  {len(test_dataset)} patches")

    # ── EDA：查看 train 切割結果 ────────────────────────────────────────────
    print("\n══════════════════════════════════════")
    print("  Train 全 0 / 全 nodata patch 統計")
    print("══════════════════════════════════════")
    count_zero_nodata_patches(data_dir=str(train_dataset.data_dir), dates=TRAIN_DATES)
    print("\n══════════════════════════════════════")
    print("  Valid 全 0 / 全 nodata patch 統計")
    print("══════════════════════════════════════")
    count_zero_nodata_patches(data_dir=str(valid_dataset.data_dir), dates=VALID_DATES)
    print("\n══════════════════════════════════════")
    print("  Test 全 0 / 全 nodata patch 統計")
    print("══════════════════════════════════════")
    count_zero_nodata_patches(data_dir=str(test_dataset.data_dir), dates=TEST_DATES)

    print("\n══════════════════════════════════════")
    print("  Train patch 灌水率與雲量遮蔽率")
    print("══════════════════════════════════════")
    df = compute_irrigation_cloud_cover_rate(
        data_dir=str(train_dataset.data_dir),
        dates=TRAIN_DATES,
    )
    print(df.groupby("date")[["water_ratio_%", "scl_cloud_ratio_%"]].mean().round(2))
    plot_irrigation_cloud_distribution(df)
