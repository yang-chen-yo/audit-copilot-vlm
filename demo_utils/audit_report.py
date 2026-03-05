# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""自然語言稽核摘要產生器。

將 F-VLM 偵測結果轉換成人類可讀的自然語言稽核報告，
並輸出 JSON 結構化報表與純文字摘要。
適用於 demo.py 後處理，也可直接串接 Web 後端 import 使用。
"""

import json
import os
from datetime import datetime


# ── 信心程度語言映射 ────────────────────────────────────────
def _confidence_phrase(score: float) -> str:
  """根據信心分數回傳自然語言描述。"""
  if score >= 0.80:
    return '明確偵測到'
  elif score >= 0.60:
    return '偵測到'
  elif score >= 0.40:
    return '疑似發現'
  else:
    return '可能存在'


# ── 位置語言映射 ────────────────────────────────────────────
def _position_phrase(box, img_h: int, img_w: int) -> str:
  """將 bounding box 轉成人類可讀的位置描述。"""
  y1, x1, y2, x2 = box
  cy = (y1 + y2) / 2
  cx = (x1 + x2) / 2

  # 垂直分三區
  if cy < img_h / 3:
    vert = '上方'
  elif cy > 2 * img_h / 3:
    vert = '下方'
  else:
    vert = '中間'

  # 水平分三區
  if cx < img_w / 3:
    horiz = '左側'
  elif cx > 2 * img_w / 3:
    horiz = '右側'
  else:
    horiz = '中間'

  # 組合成自然語言
  if vert == '中間' and horiz == '中間':
    return '畫面正中央'
  elif horiz == '中間':
    return f'畫面{vert}中間'
  elif vert == '中間':
    return f'畫面{horiz}'
  else:
    return f'畫面{vert}{horiz}'


# ── 數量語言映射 ────────────────────────────────────────────
def _count_phrase(count: int, name: str) -> str:
  """把數量轉成自然語言片語。"""
  if count == 1:
    return f'1 件 {name}'
  elif count <= 3:
    return f'{count} 件 {name}'
  else:
    return f'共 {count} 件 {name}（數量較多）'


# ── 核心：統計偵測結果 ──────────────────────────────────────
def summarize_detections(
    detection_boxes,
    detection_scores,
    detection_classes,
    id_mapping: dict,
    img_h: int,
    img_w: int,
) -> dict:
  """輸入已過濾的偵測結果，回傳整理後的 class_info 字典。

  注意：不在這裡做信心值過濾，請在呼叫前先由 demo.py 的
  --min_score_thresh 控制，或使用 num_detections 截斷。

  Args:
    detection_boxes: shape (N, 4) 的 numpy array，格式為 [y1, x1, y2, x2]，
      座標對應縮放後的圖片空間（與 image_info 的 crop 尺寸一致）。
    detection_scores: shape (N,) 的 numpy array，偵測信心分數。
    detection_classes: shape (N,) 的 numpy array，偵測類別 id（int）。
    id_mapping: dict，將 class id 對應到類別名稱字串。
    img_h: 縮放後圖片的高度（crop_height），與 detection_boxes 座標空間一致。
    img_w: 縮放後圖片的寬度（crop_width），與 detection_boxes 座標空間一致。

  Returns:
    class_info: dict，結構為：
      {
        'helmet': {
            'count': 2,
            'best_score': 0.87,
            'confidence_phrase': '明確偵測到',
            'instances': [
                {'score': 0.87, 'position': '畫面上方左側'},
                {'score': 0.72, 'position': '畫面右側'},
            ],
        },
        ...
      }
  """
  class_info = {}

  for cls_id, score, box in zip(detection_classes, detection_scores,
                                detection_boxes):
    cls_name = id_mapping.get(int(cls_id), f'class_{cls_id}')
    if cls_name in ('background', 'empty'):
      continue

    if cls_name not in class_info:
      class_info[cls_name] = {
          'count': 0,
          'best_score': 0.0,
          'confidence_phrase': '',
          'instances': [],
      }

    class_info[cls_name]['count'] += 1
    class_info[cls_name]['instances'].append({
        'score': float(score),
        'position': _position_phrase(box, img_h, img_w),
    })
    if score > class_info[cls_name]['best_score']:
      class_info[cls_name]['best_score'] = float(score)
      class_info[cls_name]['confidence_phrase'] = _confidence_phrase(
          float(score)
      )

  return class_info


# ── 核心：產生自然語言摘要段落 ─────────────────────────────
def generate_natural_summary(
    class_info: dict,
    image_name: str = '',
    template_name: str = '',
) -> str:
  """輸入 class_info，產生一段完整、自然語言的稽核摘要。

  輸出適合直接顯示在 Web 頁面或終端機，以及儲存成 .txt 檔案。

  Args:
    class_info: 由 summarize_detections() 回傳的偵測統計字典。
    image_name: 圖片檔名（選填），用於摘要開頭說明。
    template_name: 使用的稽核模板名稱（選填），例如「工安 PPE」。

  Returns:
    natural_summary: 完整的自然語言稽核摘要字串。
  """
  lines = []

  # 開頭：說明這張圖在做什麼
  intro_parts = []
  if image_name:
    intro_parts.append(f'針對「{image_name}」')
  if template_name:
    intro_parts.append(f'使用「{template_name}」模板')
  intro_parts.append('完成影像稽核')
  lines.append('，'.join(intro_parts) + '，結果如下：')
  lines.append('')

  if not class_info:
    lines.append(
        '本次稽核未偵測到任何符合條件的目標，'
        '建議確認圖片品質或調整偵測閾值後重新執行。'
    )
    return '\n'.join(lines)

  # 主體：逐類別產生自然語言句子（每類別獨立一行）
  for cls_name, info in class_info.items():
    count_str = _count_phrase(info['count'], cls_name)
    conf_phrase = info['confidence_phrase']

    sentence = f'系統{conf_phrase} {count_str}'
    instances = info.get('instances', [])
    if instances:
      positions = [inst['position'] for inst in instances]
      unique_positions = list(dict.fromkeys(positions))  # 保持順序去重
      if len(positions) == 1:
        # 只有 1 件：直接說位置
        sentence += f'，位於{unique_positions[0]}'
      elif len(unique_positions) == 1:
        # 多件但都在同一區
        sentence += f'，均位於{unique_positions[0]}'
      elif len(positions) <= 3:
        # 少量（2–3 件）且位置不同：列出來還合理
        sentence += f'，分別位於{"、".join(unique_positions)}'
      elif len(unique_positions) <= 2:
        # 數量多但只分布在 1–2 個區域
        sentence += f'，主要分布於{"與".join(unique_positions)}'
      else:
        # 數量多且分散：用統計式描述
        sentence += '，分散於畫面各處'
    sentence += '。'
    lines.append(sentence)

  lines.append('')

  # 結尾：整體評估語句
  total_items = sum(v['count'] for v in class_info.values())

  assessment = (
      f'本次共偵測到 {len(class_info)} 類目標（{total_items} 件），'
      '請對照框選圖片進一步確認細節。'
  )

  lines.append(f'【稽核評估】{assessment}')

  return '\n'.join(lines)


# ── 輸出 JSON 報表 ──────────────────────────────────────────
def save_report(
    class_info: dict,
    natural_summary: str,
    image_name: str,
    model_name: str,
    output_dir: str = './output',
    file_stem: str = '',
) -> dict:
  """將稽核結果同時存成 JSON 報表與純文字摘要 .txt 檔。

  Args:
    class_info: 由 summarize_detections() 回傳的偵測統計字典。
    natural_summary: 由 generate_natural_summary() 產生的摘要字串。
    image_name: 原始圖片檔名。
    model_name: 使用的模型名稱。
    output_dir: 輸出資料夾路徑（預設 './output'）。
    file_stem: 輸出檔名前綴（不含副檔名）。若為空則自動根據圖片名稱產生。

  Returns:
    report: 完整的報表 dict（也已寫入 JSON 檔）。
  """
  os.makedirs(output_dir, exist_ok=True)

  if not file_stem:
    base = os.path.splitext(image_name)[0]
    model_tag = model_name.replace('resnet_', 'r')
    file_stem = f'{base}_{model_tag}'

  json_path = os.path.join(output_dir, f'{file_stem}_report.json')
  txt_path = os.path.join(output_dir, f'{file_stem}_summary.txt')

  report = {
      'timestamp': datetime.now().isoformat(),
      'image': image_name,
      'model': model_name,
      'summary': natural_summary,
      'detections': class_info,
  }

  # 儲存 JSON 報表
  with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

  # 儲存純文字摘要
  with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(natural_summary)
    f.write('\n')

  return report
