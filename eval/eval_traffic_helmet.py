# coding=utf-8
"""
eval_helmet.py — 交通安全帽稽查數量層級召回率評估

GT 對應：
  With Helmet    → helmet（系統應偵測到）
  Without Helmet → 不計入 helmet GT

執行方式：
  python eval_helmet.py --template helmet_traffic --n_images 10
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from collections import defaultdict
import tqdm

from absl import app, flags
import tensorflow as tf
import jax
from PIL import Image

from demo_utils import compliance_checker, input_utils
from utils import clip_utils

# ── Flags ─────────────────────────────────────────────────────────────────────
_TEMPLATE  = flags.DEFINE_string ('template',         'helmet_traffic', '模板名稱')
_N_IMAGES  = flags.DEFINE_integer('n_images',         10,               '評估圖片數')
_MODEL     = flags.DEFINE_enum   ('model',            'resnet_50x4',
                                  ['resnet_50','resnet_50x4','resnet_50x16'], '模型')
_MAX_CLS   = flags.DEFINE_integer('max_num_classes',  91,  '最大類別數')
_MIN_SCORE = flags.DEFINE_float  ('min_score_thresh', 0.2, '預設最低門檻')

# With Helmet → helmet（計入 GT）
# Without Helmet → 不計入
LABEL_MAP = {
    'with helmet'   : 'helmet',
    'without helmet': None,
}
EVAL_CLASSES = ['helmet']

DATASET_DIR = Path.home() / '.cache/kagglehub/datasets/andrewmvd/helmet-detection/versions/1'


# ── GT 載入 ───────────────────────────────────────────────────────────────────

def load_gt_counts(xml_path):
    root   = ET.parse(xml_path).getroot()
    counts = defaultdict(int)
    for obj in root.findall('object'):
        label = LABEL_MAP.get(obj.findtext('name').lower().strip())
        if label:
            counts[label] += 1
    return counts


# ── 文本嵌入（只建一次）──────────────────────────────────────────────────────

def build_embeddings(categories, clip_fn, embed_path, max_num_cls, alias_map):
    def _encode(cls_name):
        if alias_map and cls_name in alias_map:
            embs = np.concatenate([clip_fn(a) for a in alias_map[cls_name]], axis=0)
            mean = embs.mean(axis=0, keepdims=True)
            return mean / (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-8)
        return clip_fn(cls_name)

    feats = np.concatenate(
        [_encode(c) for c in tqdm.tqdm(categories, desc='建立嵌入')], axis=0)
    bg, empty = np.load(embed_path)
    return np.concatenate([
        bg[np.newaxis], feats,
        np.tile(empty[np.newaxis], (max_num_cls - len(categories) - 1, 1))
    ], axis=0)[np.newaxis]


# ── 單張推論，回傳各類別偵測數量 ─────────────────────────────────────────────

def infer_counts(np_img, model, txt_p, txt_o, map_p, map_o, id_map, thresh_cfg, min_score):
    parser = input_utils.get_maskrcnn_parser()

    def _data(img, txt):
        d = jax.tree.map(lambda x: x.numpy()[np.newaxis],
                         parser({'image': img, 'source_id': np.array([0])}))
        d.update({'text': txt, 'image': d.pop('images')})
        return d, d.pop('labels')

    out_p = model(_data(np_img, txt_p)[0])
    out_o = model(_data(np_img, txt_o)[0])

    def _extract(out, id_mapping):
        n  = int(np.squeeze(out['num_detections']))
        ss = np.squeeze(out['detection_scores'], 0)[:n]
        cs = np.squeeze(tf.cast(out['detection_classes'], tf.int32).numpy(), 0)[:n]
        keep = [i for i in range(n)
                if ss[i] >= thresh_cfg.get(id_mapping.get(cs[i], ''), min_score)]
        return cs[keep], ss[keep]

    c_p, s_p = _extract(out_p, map_p)
    c_o, s_o = _extract(out_o, map_o)

    name2id     = {v: k for k, v in id_map.items()}
    c_p_mapped  = np.array([name2id.get(map_p.get(c, ''), 0) for c in c_p], dtype=np.int32)
    c_o_mapped  = np.array([name2id.get(map_o.get(c, ''), 0) for c in c_o], dtype=np.int32)
    classes_all = np.concatenate([c_p_mapped, c_o_mapped])

    counts = defaultdict(int)
    for cls_id in classes_all:
        label = id_map.get(int(cls_id), '')
        if label in EVAL_CLASSES:
            counts[label] += 1
    return counts


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main(argv):
    # 1. 載入模板
    tmpl             = compliance_checker.load_template(_TEMPLATE.value)
    categories       = tmpl['categories']
    person_category  = tmpl['person_category']
    threshold_config = tmpl.get('thresholds', {})
    alias_map        = tmpl.get('ppe_aliases', {})
    print(f"\n模板：{tmpl['name']}  |  評估類別：{EVAL_CLASSES}\n")

    # 2. 載入模型 + 建立文本嵌入（只做一次）
    key        = _MODEL.value.replace('resnet_', 'r')
    model      = tf.saved_model.load(f'./checkpoints/{key}')
    embed_path = f'./data/{key}_bg_empty_embed.npy'
    clip_fn    = clip_utils.get_clip_text_fn(_MODEL.value)

    p_cats = [person_category]
    o_cats = [c for c in categories if c != person_category]
    txt_p  = build_embeddings(p_cats, clip_fn, embed_path, _MAX_CLS.value, alias_map)
    txt_o  = build_embeddings(o_cats, clip_fn, embed_path, _MAX_CLS.value, alias_map)
    map_p  = {0: 'background', 1: p_cats[0]}
    map_o  = {0: 'background', **{i+1: c for i, c in enumerate(o_cats)}}
    id_map = {0: 'background', **{i+1: c for i, c in enumerate(p_cats + o_cats)}}

    # 3. 迴圈跑圖片
    xml_files = sorted((DATASET_DIR / 'annotations').glob('*.xml'))[:_N_IMAGES.value]
    print(f"共 {len(xml_files)} 張圖片\n")

    print(f"{'圖片':<32} {'GT With Helmet':>14} {'偵測 helmet':>12}")
    print("-" * 62)

    total_gt  = defaultdict(int)
    total_det = defaultdict(int)

    for xml_path in tqdm.tqdm(xml_files, desc='推論中', leave=False):
        filename = ET.parse(xml_path).getroot().findtext('filename')
        img_path = DATASET_DIR / 'images' / filename
        if not img_path.exists():
            continue

        np_img    = np.array(Image.open(img_path).convert('RGB'))
        gt_counts = load_gt_counts(xml_path)
        dt_counts = infer_counts(np_img, model, txt_p, txt_o,
                                 map_p, map_o, id_map, threshold_config, _MIN_SCORE.value)

        for cls in EVAL_CLASSES:
            total_gt[cls]  += gt_counts.get(cls, 0)
            total_det[cls] += dt_counts.get(cls, 0)

        print(f"{filename:<32} "
              f"{gt_counts.get('helmet', 0):>14} "
              f"{dt_counts.get('helmet', 0):>12}")

    # 4. 總計與召回率
    gt     = total_gt['helmet']
    det    = total_det['helmet']
    recall = det / gt if gt > 0 else 0.0

    print("-" * 62)
    print(f"{'總計':<32} {gt:>14} {det:>12}")
    print("\n" + "=" * 45)
    print(f"{'類別':<20} {'GT':>6} {'偵測':>6} {'召回率':>8}")
    print("-" * 45)
    print(f"{'helmet':<20} {gt:>6} {det:>6} {recall:>7.1%}")
    print("=" * 45)
    print(f"\n評估圖片：{len(xml_files)} 張")
    print(f"備註：GT 僅計算 With Helmet，Without Helmet 不列入。\n")


if __name__ == '__main__':
    app.run(main)
