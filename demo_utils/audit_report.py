# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""自然語言稽核摘要產生器（強化版，支持彈性類別統計與合規資訊）。

將 F-VLM 偵測結果轉換人類可讀稽核報告，並支持彈性統計 summary_stats。
"""

import json
import os
from datetime import datetime

# 信心程度語言
def _confidence_phrase(score: float) -> str:
    if score >= 0.80:
        return '明確偵測到'
    elif score >= 0.60:
        return '偵測到'
    elif score >= 0.40:
        return '疑似發現'
    else:
        return '可能存在'

# 位置語言
def _position_phrase(box, img_h: int, img_w: int) -> str:
    y1, x1, y2, x2 = box
    cy = (y1 + y2) / 2
    cx = (x1 + x2) / 2
    if cy < img_h / 3:
        vert = '上方'
    elif cy > 2 * img_h / 3:
        vert = '下方'
    else:
        vert = '中間'
    if cx < img_w / 3:
        horiz = '左側'
    elif cx > 2 * img_w / 3:
        horiz = '右側'
    else:
        horiz = '中間'
    if vert == '中間' and horiz == '中間':
        return '畫面正中央'
    elif horiz == '中間':
        return f'畫面{vert}中間'
    elif vert == '中間':
        return f'畫面{horiz}'
    else:
        return f'畫面{vert}{horiz}'

# 數量語言
def _count_phrase(count: int, name: str) -> str:
    if count == 1:
        return f'1 件 {name}'
    elif count <= 3:
        return f'{count} 件 {name}'
    else:
        return f'共 {count} 件 {name}（數量較多）'

# 偵測結果摘要
def summarize_detections(
    detection_boxes,
    detection_scores,
    detection_classes,
    id_mapping: dict,
    img_h: int,
    img_w: int,
) -> dict:
    class_info = {}
    for cls_id, score, box in zip(detection_classes, detection_scores, detection_boxes):
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
            class_info[cls_name]['confidence_phrase'] = _confidence_phrase(float(score))
    return class_info

# 彈性 summary stats，統計所有類別數量，可加上外部傳入 summary_stats (如合規)
def get_summary_stats(class_info, extra: dict = None):
    stats = {k: v['count'] for k, v in class_info.items()}
    if extra:
        stats.update(extra)
    return stats

# 強化自然語言摘要，用彈性類別清單，summary_stats 顯示於段落
def generate_natural_summary(
    class_info: dict,
    image_name: str = '',
    template_name: str = '',
    summary_stats: dict = None
) -> str:
    lines = []

    # 開頭
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
            '本次稽核未偵測到任何符合條件的目標，建議確認圖片品質或調整偵測閾值後重新執行。'
        )
        return '\n'.join(lines)

    # 主體（每類分數與統計）
    for cls_name, info in class_info.items():
        count_str = _count_phrase(info['count'], cls_name)
        conf_phrase = info['confidence_phrase']
        sentence = f'系統{conf_phrase} {count_str}'
        instances = info.get('instances', [])
        if instances:
            positions = [inst['position'] for inst in instances]
            unique_positions = list(dict.fromkeys(positions))
            if len(positions) == 1:
                sentence += f'，位於{unique_positions[0]}'
            elif len(unique_positions) == 1:
                sentence += f'，均位於{unique_positions[0]}'
            elif len(positions) <= 3:
                sentence += f'，分別位於{"、".join(unique_positions)}'
            elif len(unique_positions) <= 2:
                sentence += f'，主要分布於{"與".join(unique_positions)}'
            else:
                sentence += '，分散於畫面各處'
        sentence += '。'
        lines.append(sentence)

    lines.append('')

    # 彈性 summary stats（如合規/違規等）
    if summary_stats:
        stat_line = "【統計摘要】" + "，".join(
            [f"{k.replace('_', ' ')}: {v}" for k, v in summary_stats.items()]
        )
        lines.append(stat_line)
        lines.append('')

    # 傳統結尾
    total_items = sum(v['count'] for v in class_info.values())
    assessment = (
        f'本次共偵測到 {len(class_info)} 類目標（{total_items} 件），'
        '請對照框選圖片進一步確認細節。'
    )
    lines.append(f'【稽核評估】{assessment}')
    return '\n'.join(lines)

# 輸出 JSON 報表，summary_stats 可彈性自訂
def save_report(
    class_info: dict,
    natural_summary: str,
    image_name: str,
    model_name: str,
    output_dir: str = './output',
    file_stem: str = '',
    summary_stats: dict = None
) -> dict:
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
        'summary_stats': summary_stats or get_summary_stats(class_info)
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(natural_summary)
        f.write('\n')
    return report
