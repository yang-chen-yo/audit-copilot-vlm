# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""工安合規檢查模組。

根據場域模板，對每個偵測到的 person 判斷是否配戴了必要的 PPE。
"""

import json
import os


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


def _box_center(box):
    """回傳 box 的中心點 (cy, cx)，box 格式為 [y1, x1, y2, x2]。"""
    y1, x1, y2, x2 = box
    return (y1 + y2) / 2, (x1 + x2) / 2


def _is_ppe_near_person(person_box, ppe_box, ppe_region: str) -> bool:
    """判斷 PPE box 是否在 person box 的對應身體區域內。

    Args:
        person_box: [y1, x1, y2, x2]
        ppe_box: [y1, x1, y2, x2]
        ppe_region: 'head' 或 'torso'

    Returns:
        True 如果 PPE 在對應區域內
    """
    py1, px1, py2, px2 = person_box
    person_h = py2 - py1
    person_w = px2 - px1

    ppe_cy, ppe_cx = _box_center(ppe_box)

    # 水平方向：PPE 中心必須在 person box 左右範圍內（稍微放寬 20%）
    margin_x = person_w * 0.2
    if not (px1 - margin_x <= ppe_cx <= px2 + margin_x):
        return False

    # 垂直方向：依身體部位判斷
    if ppe_region == 'head':
        # 安全帽：PPE 中心在 person box 上方 40% 範圍內
        region_y1 = py1 - person_h * 0.1  # 稍微往上延伸（帽子可能超出 person box）
        region_y2 = py1 + person_h * 0.4
    elif ppe_region == 'torso':
        # 反光背心：PPE 中心在 person box 中間 50% 範圍內
        region_y1 = py1 + person_h * 0.2
        region_y2 = py1 + person_h * 0.75
    else:
        # 預設：整個 person box
        region_y1 = py1
        region_y2 = py2

    return region_y1 <= ppe_cy <= region_y2


def check_compliance(
    class_info: dict,
    template: dict,
    raw_boxes: dict,
) -> dict:
    """對每個 person 判斷是否合規。

    Args:
        class_info: summarize_detections() 回傳的字典
        template: load_template() 載入的模板
        raw_boxes: dict，key 為類別名稱，value 為該類別所有 box 的 list
                   格式：{'person': [[y1,x1,y2,x2], ...], 'hard hat': [...], ...}

    Returns:
        compliance_result: dict，結構為：
        {
            'total_persons': 5,
            'compliant': 3,
            'violations': [
                {
                    'person_index': 1,
                    'position': '畫面右側',
                    'missing_ppe': ['hard hat'],
                },
                ...
            ],
            'compliance_rate': 0.6,
        }
    """
    rules = template.get('compliance_rules', [])
    if not rules:
        return {}

    rule = rules[0]  # 目前只處理第一條規則
    required_ppe = rule.get('required_ppe', [])
    ppe_regions = rule.get('ppe_regions', {})

    person_boxes = raw_boxes.get('person', [])
    if not person_boxes:
        return {
            'total_persons': 0,
            'compliant': 0,
            'violations': [],
            'compliance_rate': 1.0,
        }

    violations = []
    compliant_count = 0

    # 初始化 PPE 統計
    ppe_stats = {ppe_name: 0 for ppe_name in required_ppe}

    for i, person_box in enumerate(person_boxes):
        missing = []
        for ppe_name in required_ppe:
            ppe_boxes = raw_boxes.get(ppe_name, [])
            region = ppe_regions.get(ppe_name, 'body')
            found = any(
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
                'person_box': list(person_box),
                'missing_ppe': missing,
            })
        else:
            compliant_count += 1

    total = len(person_boxes)
    return {
        'total_persons': total,
        'compliant': compliant_count,
        'violations': violations,
        'compliance_rate': compliant_count / total if total > 0 else 1.0,
        'ppe_stats': ppe_stats,  # 新增：每種 PPE 配戴人數
    }


def generate_compliance_summary(
    compliance_result: dict,
    template: dict,
    class_info: dict,
    image_name: str = '',
    img_h: int = 1,
    img_w: int = 1,
) -> str:
    """產生中文合規稽核摘要。"""
    from demo_utils.audit_report import _position_phrase

    template_name = template.get('name', '工安稽核')
    lines = []

    intro_parts = []
    if image_name:
        intro_parts.append(f'針對「{image_name}」')
    intro_parts.append(f'使用「{template_name}」模板')
    intro_parts.append('完成影像稽核')
    lines.append('，'.join(intro_parts) + '，結果如下：')
    lines.append('')

    total = compliance_result.get('total_persons', 0)
    compliant = compliance_result.get('compliant', 0)
    violations = compliance_result.get('violations', [])
    rate = compliance_result.get('compliance_rate', 1.0)

    if total == 0:
        lines.append('本次稽核未偵測到任何人員，無法進行合規判斷。')
        lines.append('')
        # 附上物件偵測結果
        for cls_name, info in class_info.items():
            if cls_name == 'person':
                continue
            lines.append(f'偵測到 {info["count"]} 件 {cls_name}。')
        return '\n'.join(lines)

    lines.append(f'共偵測到 {total} 名工作人員，其中：')
    lines.append(f'  ✅ {compliant} 人 已完整配戴 PPE')
    lines.append(f'  ❌ {len(violations)} 人 PPE 配戴不符規定')
    lines.append('')

    # PPE 配戴統計
    ppe_stats = compliance_result.get('ppe_stats', {})
    rules = template.get('compliance_rules', [])
    required_ppe = rules[0].get('required_ppe', []) if rules else []
    if ppe_stats and required_ppe:
        lines.append('【PPE 配戴統計】')
        for ppe_name in required_ppe:
            count = ppe_stats.get(ppe_name, 0)
            lines.append(f'  · {ppe_name}：{count} 人配戴 / {total} 人')
        lines.append('')

    if violations:
        lines.append('【違規明細】')
        for v in violations:
            box = v['person_box']
            pos = _position_phrase(box, img_h, img_w)
            missing_str = '、'.join(v['missing_ppe'])
            lines.append(f'  · {pos} 的工作人員：缺少 {missing_str}')
        lines.append('')

    rate_pct = rate * 100
    if rate >= 0.9:
        assessment = f'合規率 {rate_pct:.0f}%，整體狀況良好。'
    elif rate >= 0.6:
        assessment = f'合規率 {rate_pct:.0f}%，請要求違規人員立即補戴 PPE。'
    else:
        assessment = f'合規率 {rate_pct:.0f}%，違規情況嚴重，請立即停工整改。'

    lines.append(f'【稽核評估】{assessment}')

    return '\n'.join(lines)
