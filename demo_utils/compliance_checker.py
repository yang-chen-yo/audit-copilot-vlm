# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""工安合規檢查模組。

完全由 JSON 的 compliance_rules 驅動，不寫死任何場域邏輯：

  compliance_rules[i].target        → 要檢查的人員類別
  compliance_rules[i].required_ppe  → 必須配戴的裝備清單
  compliance_rules[i].ppe_regions   → 每個裝備對應的身體區域
  compliance_rules[i].violation_msg → 違規時的提示文字

新場域只需新增 JSON，程式碼完全不用動。
"""

import json
import os


# ---------------------------------------------------------------------------
# 模板載入
# ---------------------------------------------------------------------------

def load_template(template_name: str, templates_dir: str = './templates') -> dict:
    """載入場域模板 JSON。"""
    path = os.path.join(templates_dir, f'{template_name}.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Template file not found: {path}. '
            f'Please ensure "{template_name}.json" exists under "{templates_dir}".'
        ) from None


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------

def _box_center(box):
    """回傳 box 的中心點 (cy, cx)，box 格式為 [y1, x1, y2, x2]。"""
    y1, x1, y2, x2 = box
    return (y1 + y2) / 2, (x1 + x2) / 2


def _is_ppe_near_person(person_box, ppe_box, ppe_region: str) -> bool:
    """判斷 PPE box 是否在 person box 的對應身體區域內。

    ppe_region 對應 JSON ppe_regions 的值：
      'head'  → 安全帽，person box 上方 40%
      'torso' → 反光背心，person box 中間 50%
      其他    → 整個 person box（通用）
    """
    py1, px1, py2, px2 = person_box
    person_h = py2 - py1
    person_w = px2 - px1

    ppe_cy, ppe_cx = _box_center(ppe_box)

    # 水平：PPE 中心在 person 左右範圍內（放寬 20%）
    margin_x = person_w * 0.2
    if not (px1 - margin_x <= ppe_cx <= px2 + margin_x):
        return False

    # 垂直：依 ppe_region 決定對應範圍
    if ppe_region == 'head':
        region_y1 = py1 - person_h * 0.1   # 往上延伸一點（帽子可能超出 person box）
        region_y2 = py1 + person_h * 0.4
    elif ppe_region == 'torso':
        region_y1 = py1 + person_h * 0.2
        region_y2 = py1 + person_h * 0.75
    else:
        region_y1 = py1
        region_y2 = py2

    return region_y1 <= ppe_cy <= region_y2


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def check_compliance(
    class_info: dict,
    template: dict,
    raw_boxes: dict,
) -> dict:
    """
    完全依照 compliance_rules 進行合規判斷。

    每條 rule 對應一種人員角色（target），
    判斷該角色的每個人是否配戴了 required_ppe 裡的所有裝備。

    Args:
        class_info : summarize_detections() 回傳的字典
        template   : load_template() 載入的模板
        raw_boxes  : dict {cls_name -> list of [y1, x1, y2, x2]}

    Returns:
        {
            'rules_results': [          # 每條 rule 一筆
                {
                    'target'         : 'construction worker',
                    'violation_msg'  : '未完整配戴安全帽與反光背心',
                    'total_persons'  : 19,
                    'compliant'      : 14,
                    'violations'     : [
                        {'person_index': 2, 'person_box': [...], 'missing_ppe': ['hard hat']},
                        ...
                    ],
                    'compliance_rate': 0.74,
                    'ppe_stats'      : {'hard hat': 17, 'safety vest': 14},
                },
                ...
            ]
        }
    """
    rules       = template.get('compliance_rules', [])
    ppe_aliases = template.get('ppe_aliases', {})
    rules_results = []

    for rule in rules:
        target       = rule.get('target', template.get('person_category', 'person'))
        required_ppe = rule.get('required_ppe', [])
        ppe_regions  = rule.get('ppe_regions', {})
        violation_msg = rule.get('violation_msg', '裝備不符規定')

        person_boxes = raw_boxes.get(target, [])
        total        = len(person_boxes)

        if total == 0:
            rules_results.append({
                'target'         : target,
                'violation_msg'  : violation_msg,
                'total_persons'  : 0,
                'compliant'      : 0,
                'violations'     : [],
                'compliance_rate': 1.0,
                'ppe_stats'      : {ppe: 0 for ppe in required_ppe},
            })
            continue

        violations      = []
        compliant_count = 0
        ppe_stats       = {ppe: 0 for ppe in required_ppe}

        for i, person_box in enumerate(person_boxes):
            missing = []
            for ppe_name in required_ppe:
                # 合併主名稱 + 別名的所有 box
                aliases = ppe_aliases.get(ppe_name, [ppe_name])
                if ppe_name not in aliases:
                    aliases = [ppe_name] + list(aliases)
                ppe_boxes = []
                for alias in aliases:
                    ppe_boxes.extend(raw_boxes.get(alias, []))

                region = ppe_regions.get(ppe_name, 'body')
                found  = any(
                    _is_ppe_near_person(person_box, pb, region)
                    for pb in ppe_boxes
                )
                if found:
                    ppe_stats[ppe_name] += 1
                else:
                    missing.append(ppe_name)

            if missing:
                violations.append({
                    'person_index': i + 1,
                    'person_box'  : list(person_box),
                    'missing_ppe' : missing,
                })
            else:
                compliant_count += 1

        rules_results.append({
            'target'         : target,
            'violation_msg'  : violation_msg,
            'total_persons'  : total,
            'compliant'      : compliant_count,
            'violations'     : violations,
            'compliance_rate': compliant_count / total,
            'ppe_stats'      : ppe_stats,
        })

    return {'rules_results': rules_results}


# ---------------------------------------------------------------------------
# 摘要文字產生
# ---------------------------------------------------------------------------

def generate_compliance_summary(
    compliance_result: dict,
    template: dict,
    class_info: dict,
    image_name: str = '',
    img_h: int = 1,
    img_w: int = 1,
) -> str:
    """
    依照每條 compliance_rule 的結果產生稽核摘要。
    violation_msg 直接從 JSON 讀取，不寫死在程式裡。
    """
    from demo_utils.audit_report import _position_phrase

    template_name = template.get('name', '影像稽核')
    rules         = template.get('compliance_rules', [])
    rules_results = compliance_result.get('rules_results', [])

    # 標題
    intro_parts = []
    if image_name:
        intro_parts.append(f'針對「{image_name}」')
    intro_parts.append(f'使用「{template_name}」模板')
    intro_parts.append('完成影像稽核')
    lines = ['，'.join(intro_parts) + '，結果如下：']

    if not rules_results:
        lines.append('\n未設定合規規則，僅輸出偵測數量。')
        for cls_name, info in class_info.items():
            lines.append(f'偵測到 {info["count"]} 件 {cls_name}。')
        return '\n'.join(lines)

    for rule_def, result in zip(rules, rules_results):
        required_ppe  = rule_def.get('required_ppe', [])
        violation_msg = result['violation_msg']   # 直接用 JSON 裡的文字
        total         = result['total_persons']
        compliant     = result['compliant']
        violations    = result['violations']
        rate          = result['compliance_rate']
        ppe_stats     = result['ppe_stats']

        lines.append('')
        lines.append(f'▌ 檢查對象：{result["target"]}')
        lines.append('')

        if total == 0:
            lines.append('  未偵測到此類人員，無法進行合規判斷。')
            continue

        lines.append(f'  共偵測到 {total} 人，其中：')
        lines.append('')

        # 第一層：每種 PPE 單獨統計
        for ppe_name in required_ppe:
            count   = ppe_stats.get(ppe_name, 0)
            missing = total - count
            icon    = '✅' if missing == 0 else '❌'
            suffix  = '，全員合規。' if missing == 0 else f'，疑似 {missing} 人未配戴。'
            lines.append(f'  {icon}  {ppe_name}：{count}/{total} 人配戴{suffix}')

        # 第二層：同時滿足所有 PPE
        lines.append('')
        ppe_list = '、'.join(required_ppe)
        lines.append(f'  🔍  同時配戴所有裝備（{ppe_list}）：{compliant}/{total} 人。')

        # 違規明細（violation_msg 來自 JSON）
        if violations:
            lines.append('')
            lines.append(f'  【違規明細】（{violation_msg}）')
            for v in violations:
                pos         = _position_phrase(v['person_box'], img_h, img_w)
                missing_str = '、'.join(v['missing_ppe'])
                lines.append(f'    · {pos} 的人員：缺少 {missing_str}')

        # 整體評估
        lines.append('')
        pct = rate * 100
        if rate >= 0.9:
            assessment = f'合規率 {pct:.0f}%，整體狀況良好。'
        elif rate >= 0.6:
            assessment = f'合規率 {pct:.0f}%，請要求違規人員立即補戴 PPE。'
        else:
            assessment = f'合規率 {pct:.0f}%，違規情況嚴重，請立即停工整改。'
        lines.append(f'  【稽核評估】{assessment}')

    return '\n'.join(lines)
