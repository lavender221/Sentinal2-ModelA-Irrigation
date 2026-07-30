---
title: Sentinel2_ModelA.md

---

# Sentinel-2 灌溉判識 Model A

文件分成三部分：**一、Base Model**（目前程式的完整設定，作為所有實驗的基準）、**二、衍生模型**（只描述與 Base Model 的差異，含測試結果比較表）、**三、待辦事項**。

---

# 一、Base Model

## 📁 原始資料路徑

### 2023年第一期作
`X:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\S2訓練資料\2023年一期作\2023全台`

- `S2特徵合併_原波長版_nday20.tif`
- `S2_原波長_Band說明.txt`
- `灌水網格合併_含弱相關.tif`
- `灌水網格合併Band說明.txt`

### 2023年第二期作
`X:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\2023年二期作訓練資料`

- `S2特徵合併_原波長版_nday20.nc`
- `灌水網格合併_含弱相關.tif`
- `灌水網格合併Band說明_含弱相關.txt`

### 2026年第一期作
`X:\01_資料專區\05_圖資GIS彙整版\06_衛星影像\2026年1期作全台`

- `S2特徵合併_原波長版_nday20.nc`
- `灌水網格合併_含弱相關.tif`
- `灌水網格合併Band說明_含弱相關.txt`

## 📂 整理後資料

存放於 `S2全台資料_Xy對應` 資料夾下。

### X（影像特徵）

```
X -- 20230110 -- N1.npy、N2.npy、...、W5.npy
   -- 20230115 -- N1.npy、N2.npy、...、W5.npy
    :
   -- 20230912 -- N1.npy、N2.npy、...、W5.npy
   -- 20260101 -- N1.npy、N2.npy、...、W5.npy
   -- 20260104 -- N1.npy、N2.npy、...、W5.npy
    :
   -- 20260509 -- N1.npy、N2.npy、...、W5.npy
```

- 每個 `.npy` 檔案的 `shape = [5, 。, 。]`，依序為 **Green、Red、NIR、SWIR、SCL**
- **像素值 = 999**：大部分位於非農地範圍；非農地範圍的 999 視為**實際測量值**
- 日期數量：2023年共 **37** 個日期、2026年共 **19** 個日期

### y（灌水標籤）

```
y -- 20230130 -- N1.npy、N2.npy、...、W5.npy
   -- 20230204 -- N1.npy、N2.npy、...、W5.npy
    :
   -- 20230824 -- N1.npy、N2.npy、...、W5.npy
   -- 20260104 -- N1.npy、N2.npy、...、W5.npy
   -- 20260119 -- N1.npy、N2.npy、...、W5.npy
    :
   -- 20260419 -- N1.npy、N2.npy、...、W5.npy
```

- 每個 `.npy` 檔案的 `shape = [1, 。, 。]`
- 標籤定義：

| 數值 | 意義 | 處理方式 |
|:---:|---|---|
| 0 | 農地，無灌水 | 負樣本 |
| 1 | 農地，有灌水 | 正樣本 |
| 0.5 | 當期作有灌水（弱相關） | 排除於 loss 與所有指標之外 |
| 999 | 非農地範圍 | 排除於 loss 與所有指標之外 |

- 日期數量：2023年共 **21** 個日期、2026年共 **12** 個日期

### 🔗 X-y 配對邏輯

由於 X 的日期數量（56個）多於 y（33個）,兩者並非一一對應,配對規則如下：

> 針對每一個 **y 的日期**,在 X 資料夾中尋找 **小於等於該 y 日期的最大日期** 所對應的資料夾,作為該筆樣本的輸入影像。

例如：y 日期為 `20230130`,若 X 資料夾中有 `20230115`、`20230125`、`20230201`,則會選擇 `20230125`（最接近且不超過 y 日期）作為配對的 X。

配對到的 X 日期會記錄於切割 json 的 `x_date` 欄位,**X-y 天數差** 由 Dataset 依 `date` 與 `x_date` 計算,作為「時間差輔助特徵」的輸入來源。

## 🔀 訓練、驗證、測試切分

### 時間獨立（Time-independent split，Case1）——Base Model 採用

- 資料範圍：**2023年與2026年的全部日期**
- 切分方式：用固定 seed 隨機打亂全部日期後，依 訓練 : 驗證 : 測試 = **6 : 2 : 2** 切分
  （不再以年份分組，三組皆可能同時包含 2023 與 2026 的日期）

### 空間獨立（Spatial-independent split，Case2）

- 資料範圍：2023年與2026年，共 **33 個日期**（2023年21個 + 2026年12個）、**11 個區塊**
  - 區塊清單：`N1`、`N2`、`EN`、`E1`、`E2`、`E3`、`W1`、`W2`、`W3`、`W4`、`W5`
- 切分比例：訓練 : 驗證 : 測試 = **7 : 2 : 2**（round(11×0.6)=7、round(11×0.2)=2）

> ⚠️ **已知限制**：切分是先對完整大圖依「時間」或「空間」進行切分,之後才進行 patch 切割。因此：
> - **時間獨立切分**時,空間上可能存在洩漏（同一區塊同時出現在不同時間切分的樣本中）
> - **空間獨立切分**時,時間上可能存在洩漏（同一時間點同時出現在不同區塊切分的樣本中）
>
> 此為資料特性所致的取捨,目前先接受此限制,不另外處理。

## ✂️ 資料切割（Patch 切割）

| 用途 | 影像大小 | 移動步輻 |
|---|:---:|:---:|
| 訓練 | 256 × 256 | 128（= patch_size × 50%） |
| 驗證 / 測試 | 256 × 256 | 256（= patch_size） |

> 📌 各 split 的儲存方式（X 來源日期使用「X-y 配對邏輯」配對到的 X 日期影像）：
> - **切割結果與模型解耦**：切割結果不跟隨模型名稱,依「**切割方式（time/spatial）+ patch_size + stride + keep_remainder**」命名,集中存於 `split_ways/` 下（例如 `split_ways/time_p256_s128_keepFalse/`）;**各模型透過 CONFIG 的這組參數選擇一個切割結果資料夾**,相同切割設定的模型共用同一份、不重切
> - **train / valid / test 一律相同**：整個 split 只寫一份 json（`train_info.json` / `valid_info.json` / `test_info.json`）,每個（日期, 區塊）為 json 中的一筆 item,記錄重建/即時切割用的切割座標與 padding 資訊,**不儲存 patch `.npy`**;Dataset 讀取時依座標從原始大圖即時切割（memmap,只讀取視窗範圍）
> - **train 額外欄位**：每個 patch 條目另記錄**灌水比例**（`water_ratio`,**分子只計 y=1**,弱相關的 0.5 不算灌水）、**SCL（=3, 8, 9）比例**（`scl_cloud_ratio`,以上分母為 y≠999 的有效像素）與 **999 比例**（`nodata_ratio`,分母為整張 patch 的像素數）,皆為 0~1 小數
> - **切割階段不做任何 patch 篩選**：篩選（`filter_all_nodata` / `filter_no_water` / `cloud_rate_threshold`）與下採樣改由宣告 Dataset 時依 `train_info.json` 的比例欄位進行
> - **正規化統計量**：以 train 統計出的 `clip_min`/`clip_max`（1st~99th 百分位）與 `mean`/`std` 存於同一個切割結果資料夾的 `norm_stats.json`,重跑（`generate_patch=False`）時直接讀取、不重算

## ⚙️ 前處理流程

適用波段：**Green、Red、NIR、SWIR**

所有統計量（百分位數上下界、mean / std）僅使用 **訓練集** 資料計算,再套用到驗證 / 測試集,避免資料洩漏。

**統計前的像素篩選**（Step 1、Step 2 共用）：

- **以 y 遮蔽非農地**：X 中的 999 因資料處理問題混雜了原本的有效值,無法直接剔除;但各日期同一區塊的 999 位置相同,因此以 **`y/20230130`** 各區塊中 **y = 999（非農地/無效區域）** 的位置作為固定遮罩,被遮蔽的 X 像素不計入統計
- **剔除 X 中值為 65535 的像素**（nodata,遮蔽後農地範圍內仍可能出現）
- X 本身的 999 視為可能有效值,**不剔除**

1. **Step 1**：取 **1st ~ 99th 百分位數** 作為上下界裁切（clip）邊界
2. **Step 2**：將像素 clip 到 Step 1 的邊界後,計算 **mean / std**,供 **Z-score 標準化**（`(X − mean) / std`）使用
   - Dataset 讀取 patch 時,對每個波段同樣先 clip 到 Step 1 的邊界,再以訓練集的 mean / std 做 Z-score

## 🧮 模型輸入（Dataset）

**輸入通道（共 5 通道）**：

| 通道 | 內容 | 說明 |
|---|---|---|
| 1~4 | Green、Red、NIR、SWIR | clip 後 Z-score 標準化 |
| 5 | 時間差輔助通道 | (y 日期 − X 日期) 天數差 ÷ `nday`（=20）的常數平面,反映資料新鮮度,固定為最後一個通道 |

- **SCL one-hot 暫不使用**：SCL（尤其水體類別 SCL=6,經驗證與灌水事件有高度關聯）可經 one-hot encoding（12 真實類別 + 1 類 nodata/異常值如 65535,共 12+1 通道）作為輔助輸入。功能已實作於 Dataset（features 加入 `"SCL"` 即啟用）,Base Model 不使用
- **NDVI / NDWI / LSWI 衍生指數暫不使用**：Dataset 已支援,Base Model 不使用
- **資料擴增：無**

**train patch 篩選與下採樣**（依 `train_info.json` 的比例欄位,於宣告 Dataset 時進行）：

1. `filter_all_nodata = True`：剔除整張皆為 nodata 的 patch（`nodata_ratio == 1`）
2. `filter_no_water = False`、`cloud_rate_threshold = None`（皆不啟用）
3. **無灌水 patch 下採樣**（於上述篩選**之後**套用）：含灌水像素（`water_ratio > 0`）的 patch（n 張）**全數保留**,無灌水（`water_ratio == 0`,只含 0/0.5/999）的 patch 只**隨機保留 n × 0.25 張**（`no_water_keep_ratio = 0.25`,固定 seed 可重現）

**Base Model 實際訓練資料量**（Case1,`time_p256_s128_keepFalse`）：
train json 共 419,520 patch → `filter_all_nodata` 剔除 167,180 → 剩 252,340（有灌水 48,902、無灌水 203,438）→ 下採樣後**實際用於訓練 61,128 patch**。

## 🧠 模型與訓練設定

| 項目 | 設定內容 |
|---|---|
| 切分方式 | **Case1 時間獨立**（6:2:2） |
| 切割結果 | `split_ways/time_p256_s128_keepFalse/` |
| 模型架構 | **UNet**，encoder = `resnet34`（ImageNet 預訓練） |
| 輸入通道 | 5（4 波段 + 時間差） |
| 損失函數 | **BCE × 0.5 + Dice × 0.5**（`pos_weight = 1.0` 等權；`scl_mask = False` 不遮雲） |
| 優化器 | **AdamW**（lr = 1e-4、weight_decay = 1e-5） |
| batch size | 4 |
| DataLoader | `num_workers = 4`、`persistent_workers = True`、`pin_memory`（有 GPU 時） |
| epochs / early stopping | 100 / patience 10（監測 val_loss） |
| 預測閾值 | sigmoid ≥ 0.5 → 1 |
| seed | 42（日期切分、下採樣抽樣、torch 皆用同一 seed） |

### 🔧 輸入通道調整（Encoder 修改）

- `resnet34` 預訓練權重的第一層卷積為 **3 通道**（RGB）設計,輸入通道數 > 3
- 修改方式：**前 3 通道沿用 RGB 預訓練權重,第 4 通道起（波段、SCL one-hot、時間差等新增通道）以 RGB 三通道權重的平均值複製填入**,其餘層沿用預訓練權重（實作於 `Model.build_model()`）

### 🎯 Loss 與指標的標籤排除規則

- **y = 999（非農地）**：排除,不計入 loss 與任何指標
- **y = 0.5（當期作有灌水,弱相關）**：排除,不計入 loss 與任何指標
- 預測重建圖上 y=999 的位置會蓋回 999；y=0.5 的位置保留模型預測

### ⚖️ 類別不平衡處理

灌水像素佔比極低（實測全域盛行率約 2.24%）。Base Model 採用**無灌水 patch 下採樣**（見「模型輸入」第 3 點）,`pos_weight` 維持 1.0 等權、不做 oversampling,以此建立基準。

## 📊 評估與輸出

- **模型輸出組織**：CONFIG 的 `MODEL_NAME`（如 `base_model`、之後的 `M1_augmentation`…）決定輸出資料夾 `model_output/{MODEL_NAME}/` 與 TensorBoard run 名稱;每個模型各自取名、互不覆蓋（切割結果在 `split_ways/`,與模型名稱無關）
- 指標：pixel-level confusion matrix（排除 999 與 0.5）計算 **accuracy / precision / recall / F1 / IoU**
- **EDA**（訓練前檢視,統計對象為篩選＋下採樣後**實際用於訓練**的 patch）：整體單列統計（total / all_nodata / all_zero / have_one / 平均灌水率 / 平均遮蔽率,直接讀 json 比例欄位）＋ 整體灌水率、遮蔽率分布直方圖各一張
- 結果檔案（都在 output_dir,中斷或重啟 kernel 皆保留）：
  - `train_history.json`：每 epoch 的 train/valid loss/acc/f1 + `best_val_loss` / `best_val_acc` / `best_val_f1` / `best_epoch`（逐 epoch 覆寫）
  - `test_metrics.json`：測試集 acc / precision / recall / f1 / IoU + confusion matrix
  - `eda.txt`（整體統計文字）、`eda_irrigation_cloud_distribution.png`（分布圖）、`confusion_map_*.png`
- **預測結果視覺化**：對「指定的 (date, region)」即時推論——載入 best_model 對該區域的 test patches 當場預測,依 json 座標重建完整圖（nodata 蓋回 999）,與 xy_dir 原始 y 大圖比對畫出 TP/TN/FP/FN confusion map;**不預先輸出 test_pred / test_gt 中間檔**
- TensorBoard：由「TensorBoard 匯整」cell 讀取上述檔案,一次重建單一 run（訓練曲線 + HPARAMS + EDA TEXT + confusion map IMAGES;視覺化 cell 在匯整之後,新畫的圖重跑匯整即可納入）

---

# 二、衍生模型

> 每個衍生模型**只描述與 Base Model 的差異**,其餘設定皆與 Base Model 相同。狀態：⬜ 規劃中 / 🔄 訓練中 / ✅ 已完成。

### M1：資料擴增 ⬜

- train 加入隨機水平/垂直翻轉 + 90° 旋轉（衛星影像標準擴增,針對 train/val 泛化差距）

### M2：雲遮蔽處理 ⬜

- `scl_mask = True`：loss 忽略 SCL∈{3, 8, 9}（雲影/中機率雲/高機率雲）像素,避免雲下標籤成為噪聲
- 變體 M2b：`cloud_rate_threshold = 0.3`,train 直接剔除高雲 patch

### M3：特徵擴充 ⬜

- features 加入 **SCL one-hot**（12+1 通道）與 **NDWI / LSWI / NDVI** 衍生指數
- 輸入通道 5 → 21（4 波段 + 13 SCL + 3 指數 + 1 時間差）

### M4：訓練設定調整 ⬜

- batch size 4 → 32~64（RTX 5090 32GB）,lr 等比例上調
- weight_decay 1e-5 → 1e-2
- 加入 lr scheduler（ReduceLROnPlateau 監測 val_loss）

### M5：類別不平衡強化 ⬜

- `pos_weight` 1.0 → 3~5（提升 recall）
- 高灌水比例 patch oversampling（依 json 的 `water_ratio` 欄位 + `WeightedRandomSampler`）

## 🏆 測試結果比較表

> 數值來源：各模型 output_dir 的 `test_metrics.json`（TensorBoard HPARAMS tab 有相同紀錄）。

| 模型 | accuracy | f1_score | precision | recall | IoU | 備註 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **base_model** | 0.9760 | 0.3465 | 0.4138 | 0.2981 | 0.2096 | Case1 時間切分；best_val_loss 0.4966（epoch 4）；run：`base_model` |
| M1 資料擴增 | — | — | — | — | — | |
| M2 雲遮蔽處理（scl_mask） | — | — | — | — | — | |
| M2b 雲遮蔽處理（高雲 patch 剔除） | — | — | — | — | — | |
| M3 特徵擴充 | — | — | — | — | — | |
| M4 訓練設定調整 | — | — | — | — | — | |
| M5 類別不平衡強化 | — | — | — | — | — | |

---

# 三、待辦事項

- [x] **base_model 訓練 + 測試**,結果已填入比較表（run：`base_model`;觀察：train loss 持續下降、val loss 停滯 ~0.50,train/val 泛化差距大 → M1~M5 的主要動機）
- [ ] **逐日期 val F1 × 雲量診斷**：把 val 各日期的 F1 分開統計,對照該日期雲量,確認是否為特定高雲日期拖垮平均
- [ ] **train / val / test 雲量分布檢查**：確認各切分之間的 SCL 品質分布相近,避免驗證結果過於樂觀
- [ ] 依診斷結果依序執行 M1~M5 衍生模型實驗,更新比較表
- [ ] **時序資訊**：評估加入前一次（或前兩次）X,讓模型學習變化趨勢（如 NDWI 時間差分）而非單一時間點快照
- [ ] **Loss 函數**：嘗試 Tversky Loss（分別調整 FP/FN 懲罰權重）、Focal Loss（加大難分類樣本權重）
- [ ] **後處理**：形態學操作（opening/closing）去除孤立小區域雜訊;全圖預測改用有重疊的滑動窗口 + 平均融合（overlap-average stitching）避免拼接邊界瑕疵
- [ ] **評估指標**：因類別極度不平衡,accuracy 參考價值低;補充 MCC、PR-AUC 作為主要評估指標
- [ ] **弱相關樣本**：「含弱相關」（y=0.5）的樣本,評估依信心程度給予較低 loss 權重,而非直接排除
- [ ] **預訓練消融**：比較「ImageNet 預訓練」vs「從頭訓練（random init）」,不預設預訓練一定有幫助（自然影像與衛星多光譜的 domain gap 大）
- [ ] **空間獨立切分（Case2）**：以 SPLIT_MODE=spatial 跑同一套流程,與 Case1 比較
