# 可配置視覺稽核助理（計畫書）
from https://github.com/google-research/google-research/tree/master/fvlm
圖源：Safety-Helmet-Wearing-Dataset、wikimedia

日期：2026-03-02  
作者/團隊：＿＿＿＿＿  
版本：v1.0

## Download the checkpoints.
Run the following commands from the root fvlm directory. 

```
cd ./checkpoints
./download.sh
```

For users who want to use the FLAX checkpoints directly rather than tf.SavedModel, we have prepared the checkpoints for downloading by the commands below:

```
MODEL="r50"  # Supported model: r50, r50x4, r50x16
wget "https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fvlm/jax_checkpoints/${MODEL}_checkpoint_184000"
```
We recommend users to run the above commands in the checkpoints directory.

```
python demo.py --demo_image_name="7.jpg" --category_name_string="person,hard hat,safety vest" --template construction_site
```
---

## 1. 主題
**InspectGPT Lite** 是一個「可配置」的影像稽核工具。  
使用者只要選一個稽核模板（或輸入要檢查的物件清單），系統就能：
- 在照片上標出偵測結果（證據圖）
- 產生一段簡短的稽核摘要（數量 + 大概位置 + 提醒）
- 產出 JSON 報表（方便留存與追蹤）

---

## 2. 要解決的問題
現場稽核/巡檢常見痛點：
1. **看照片很花時間**：要人工找重點、再寫報告。
2. **稽核項目常變**：不同工地/場域要檢查的東西不一樣。
3. **缺少可追溯資料**：需要「圖 + 文字 + 結構化結果」才能留存與追蹤改善。

本作品要解決的是：  
> 用最少的設定，把一張照片快速變成「可交付的稽核結果」。

---

## 3. 作法（簡單、做得完）
### 3.1 輸入
- 一張照片（Web 上傳）
- 稽核模板（例如 PPE / 消防），或一串關鍵字（逗號分隔）

### 3.2 流程
1. **偵測**：使用既有 F-VLM demo 推論，取得偵測結果（boxes / scores / classes）
2. **統計與整理**（不改模型）：
   - 分數太低的先過濾掉
   - 每個類別算「可見件數」
   - 每個類別抓一個代表位置（取最高分那個，粗分：左/中/右、上/中/下）
3. **輸出**：
   - 框選後圖片（證據）
   - 稽核摘要文字（固定模板產生）
   - JSON 報表（留存/串接）

---

## 4. 我們的創新點（用白話講）
1. **不用重新訓練也能換稽核項目**：用「模板/清單」就能換要檢查的東西，導入快。
2. **不只輸出框圖，直接給稽核結論**：把偵測結果整理成一句人看得懂的摘要。
3. **有報表可追溯**：每次稽核都輸出 JSON，方便後續追蹤改善。

> 備��：數量輸出採「可見件數」，遇到遮擋/重疊會加註提醒，避免誤導。

---

## 5. Web Demo 會長什麼樣子（MVP）
頁面包含：
- 上傳圖片
- 選擇模板（至少 2 個）
- 按「開始稽核」

輸出包含：
1. 框選結果圖（可下載）
2. 稽核摘要（可複製）
3. JSON 結果（可下載）

---

## 6. 兩個模板（先做最基本、好展示）
### 模板 A：工安 PPE（示例）
**關鍵字**：`person, helmet, safety vest`  
**摘要示例**：  
「偵測到 person 3 個（主要在中央），helmet 2 個（代表在左側）。疑似有 1 位未配戴安全帽，建議補拍近距離照片確認。」

### 模板 B：消防巡檢（示例）
**關鍵字**：`fire extinguisher, exit sign, obstacle`  
**摘要示例**：  
「偵測到 fire extinguisher 1 個（右下）。通道中央疑似有障礙物，建議清空後重新拍照存證。」

---

## 7. 交付內容（做完就能 Demo）
- 可執行的 Web（上傳→輸出結果）
- 兩個模板可切換
- 每張圖輸出：框圖 + 摘要 + JSON
- 一組展示圖片（20–30 張）與來源/授權說明
- 1–2 分鐘 Demo 影片 + 簡報（6–8 頁）

---

## 8. 待完成事項（1 週清單）
**Day 1**：決定模板與句型，準備展示圖片  
**Day 2-3**：完成後處理（計數、代表位置、摘要、JSON）  
**Day 4**：串上簡單 Web（上傳/回傳結果）  
**Day 5**：挑 5 個最佳案例做截圖與說明  
**Day 6**：錄 Demo 影片、做簡報  
**Day 7**：整理投件資料與彩排 Q&A

---
