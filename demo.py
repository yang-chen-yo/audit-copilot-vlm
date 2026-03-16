import functools
import os
from absl import app, flags, logging
from demo_utils import audit_report, input_utils, vis_utils
import jax
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm
import json
from utils import clip_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))#取得demo資料夾路徑
OUTPUT_DIR = os.path.join(BASE_DIR, 'static', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

_DEMO_IMAGE_NAME = flags.DEFINE_string('demo_image_name', '7.jpg', '圖片檔名。')
_CATEGORY_NAME_STRING = flags.DEFINE_string('category_name_string', '', '類別清單 (逗號分隔)。')
_MODEL = flags.DEFINE_enum('model', 'resnet_50x4', ['resnet_50', 'resnet_50x4', 'resnet_50x16'], 'F-VLM 模型。')
_MAX_BOXES_TO_DRAW = flags.DEFINE_integer('max_boxes_to_draw', 100, '最大繪製框數。')
_MAX_NUM_CLS = flags.DEFINE_integer('max_num_classes', 91, '最大類別數。')
_MIN_SCORE_THRESH = flags.DEFINE_float('min_score_thresh', 0.2, '預設最低門檻。')
_TEMPLATE = flags.DEFINE_string('template', '', '稽核模板名稱。')


def _build_text_embeddings(categories, clip_text_fn, embed_path, max_num_cls,
                            alias_map=None):
    """建立文本嵌入。
    若 alias_map 有對應別名清單，對所有別名分別編碼後取平均（L2 正規化），
    提升 CLIP 對低頻類別（如工人）的召回率。
    alias_map=None 時行為與原版完全相同。
    """
    def _encode(cls_name):
        if alias_map and cls_name in alias_map:
            embs = np.concatenate(
                [clip_text_fn(a) for a in alias_map[cls_name]], axis=0)  # (N, D)
            mean = embs.mean(axis=0, keepdims=True)                       # (1, D)
            return mean / (np.linalg.norm(mean, axis=-1, keepdims=True) + 1e-8)
        return clip_text_fn(cls_name)

    class_clip_features = [_encode(cls_name) for cls_name in tqdm.tqdm(categories)]
    text_embeddings = np.concatenate(class_clip_features, axis=0)
    background_embedding, empty_embeddings = np.load(embed_path)
    text_embeddings = np.concatenate(
        (background_embedding[np.newaxis, ...],
         text_embeddings,
         np.tile(empty_embeddings[np.newaxis, ...], (max_num_cls - len(categories) - 1, 1))), axis=0)
    return text_embeddings[np.newaxis, ...]


def _build_np_data(np_image, text_embeddings):
    parser_fn = input_utils.get_maskrcnn_parser()
    data = parser_fn({'image': np_image, 'source_id': np.array([0])})
    np_data = jax.tree.map(lambda x: x.numpy()[np.newaxis, ...], data)
    np_data.update({'text': text_embeddings, 'image': np_data.pop('images')})
    return np_data, np_data.pop('labels'), np_data['image']


def _extract_detections(output, id_mapping, threshold_config):
    """執行類別差異化門檻過濾。"""
    n = int(np.squeeze(output['num_detections']))
    boxes = np.squeeze(output['detection_boxes'], axis=0)[:n]
    scores = np.squeeze(output['detection_scores'], axis=0)[:n]
    classes = np.squeeze(output['detection_classes'].astype(np.int32), axis=0)[:n]

    keep = []
    for i in range(n):
        name = id_mapping.get(classes[i], f'class_{classes[i]}')
        thresh = threshold_config.get(name, _MIN_SCORE_THRESH.value)
        if scores[i] >= thresh:
            keep.append(i)
    return boxes[keep], scores[keep], classes[keep]


def main(argv):
    clip_text_fn = clip_utils.get_clip_text_fn(_MODEL.value)
    pure_filename = os.path.basename(_DEMO_IMAGE_NAME.value)
    demo_image_path = os.path.join(BASE_DIR, 'static', 'uploads', _DEMO_IMAGE_NAME.value)
    file_base_name = os.path.splitext(pure_filename)[0]
    model_suffix = _MODEL.value.replace("resnet_", "r")
    output_filename = f"{file_base_name}_{model_suffix}.jpg"
    output_image_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"✅ 結果圖片將儲存至: {output_image_path}")

    with open(demo_image_path, 'rb') as f:
        np_image = np.array(Image.open(f).convert('RGB'))

    # 1. 讀取模板與門檻配置
    categories, person_category, threshold_config, alias_map = [], None, {}, {}
    tmpl = {}
    if _TEMPLATE.value:
        from demo_utils import compliance_checker
        is_path = _TEMPLATE.value.endswith('.json') or '/' in _TEMPLATE.value
        
        if is_path:
            # 這是從 translate.py 產生的動態路徑
            if os.path.exists(_TEMPLATE.value):
                with open(_TEMPLATE.value, 'r', encoding='utf-8') as f:
                    tmpl = json.load(f)
                print(f"成功載入動態指令檔案: {_TEMPLATE.value}")
            else:
                print(f"找不到路徑檔案: {_TEMPLATE.value}")
        else:
            # 這是原本內建的模板名稱 (例如 'construction_site')
            tmpl = compliance_checker.load_template(_TEMPLATE.value)
            print(f"✅ 成功載入內建模板: {_TEMPLATE.value}")
        categories       = tmpl.get('categories', [])
        person_category  = tmpl.get('person_category', None)
        threshold_config = tmpl.get('thresholds', {})
        alias_map        = tmpl.get('ppe_aliases', {})
    else:
        categories = _CATEGORY_NAME_STRING.value.split(',')

    model_sub_path = _MODEL.value.replace("resnet_", "r")
    model_full_path = os.path.join(BASE_DIR, 'checkpoints', model_sub_path)

    print(f"正在從絕對路徑載入模型: {model_full_path}")
    model = tf.saved_model.load(model_full_path)
    embed_filename = f'{model_sub_path}_bg_empty_embed.npy'
    embed_path = os.path.join(BASE_DIR, 'data', embed_filename)

    print(f"正在載入 Embedding 檔案: {embed_path}")

    # 2. 執行拆分推論以解決重疊物件的語義競爭
    use_split = (person_category in categories and len(categories) > 1)

    if use_split:
        p_cats, o_cats = [person_category], [c for c in categories if c != person_category]
        txt_p = _build_text_embeddings(p_cats, clip_text_fn, embed_path, _MAX_NUM_CLS.value, alias_map)
        txt_o = _build_text_embeddings(o_cats, clip_text_fn, embed_path, _MAX_NUM_CLS.value, alias_map)

        out_p = model(_build_np_data(np_image, txt_p)[0])
        out_o = model(_build_np_data(np_image, txt_o)[0])

        map_p = {0: 'background', 1: p_cats[0]}
        map_o = {0: 'background', **{i+1: c for i, c in enumerate(o_cats)}}

        b_p, s_p, c_p_l = _extract_detections(out_p, map_p, threshold_config)
        b_o, s_o, c_o_l = _extract_detections(out_o, map_o, threshold_config)

        id_map = {0: 'background', **{i+1: c for i, c in enumerate(p_cats + o_cats)}}
        name_to_id = {v: k for k, v in id_map.items()}
        c_p = np.array([name_to_id.get(map_p.get(c, ''), 0) for c in c_p_l], dtype=np.int32)
        c_o = np.array([name_to_id.get(map_o.get(c, ''), 0) for c in c_o_l], dtype=np.int32)

        boxes_all   = np.concatenate([b_p, b_o], axis=0)
        scores_all  = np.concatenate([s_p, s_o], axis=0)
        classes_all = np.concatenate([c_p, c_o], axis=0)
    else:
        txt = _build_text_embeddings(categories, clip_text_fn, embed_path, _MAX_NUM_CLS.value, alias_map)
        out = model(_build_np_data(np_image, txt)[0])
        id_map = {0: 'background', **{i+1: c for i, c in enumerate(categories)}}
        boxes_all, scores_all, classes_all = _extract_detections(out, id_map, threshold_config)

    # 3. NMS：各類別分開跑，避免 worker box 被 vest/helmet box 的 IoU 重疊刪掉
    if len(boxes_all) > 0:
        final_boxes, final_scores, final_classes = [], [], []
        for cls_id in np.unique(classes_all):
            mask = classes_all == cls_id
            cls_boxes  = boxes_all[mask]
            cls_scores = scores_all[mask]
            sel = tf.image.non_max_suppression(
                cls_boxes, cls_scores, _MAX_BOXES_TO_DRAW.value, iou_threshold=0.45
            ).numpy()
            final_boxes.append(cls_boxes[sel])
            final_scores.append(cls_scores[sel])
            final_classes.append(np.full(len(sel), cls_id, dtype=np.int32))

        boxes_all   = np.concatenate(final_boxes,   axis=0)
        scores_all  = np.concatenate(final_scores,  axis=0)
        classes_all = np.concatenate(final_classes, axis=0)

    # 4. 視覺化輸出
    output_vis = {
        'detection_boxes'  : boxes_all[np.newaxis, ...],
        'detection_scores' : scores_all[np.newaxis, ...],
        'detection_classes': classes_all[np.newaxis, ...].astype(np.float32),
        'num_detections'   : np.array([len(scores_all)]),
    }
    _, labels, image_raw = _build_np_data(np_image, np.zeros((1, _MAX_NUM_CLS.value, 1024)))
    vis_image = vis_utils.visualize_instance_segmentations(
        output_vis, image_raw, labels['image_info'],
        functools.partial(vis_utils.visualize_boxes_and_labels_on_image_array,
                          category_index=input_utils.get_category_index(id_map),
                          use_normalized_coordinates=False, min_score_thresh=0.01))
    Image.fromarray(vis_image).save(output_image_path)

    # 5. 生成稽核摘要
    img_h = int(labels['image_info'][0, 0, 0] * labels['image_info'][0, 2, 0])
    img_w = int(labels['image_info'][0, 0, 1] * labels['image_info'][0, 2, 1])

    if tmpl.get('compliance_rules'):
        from demo_utils import compliance_checker

        # 把各類別的 box 整理成 raw_boxes dict {cls_name -> list of [y1,x1,y2,x2]}
        raw_boxes = {}
        for cls_id, cls_name in id_map.items():
            if cls_name == 'background':
                continue
            mask = classes_all == cls_id
            raw_boxes[cls_name] = boxes_all[mask].tolist()

        class_info = audit_report.summarize_detections(
            boxes_all, scores_all, classes_all, id_map, img_h, img_w)

        result  = compliance_checker.check_compliance(class_info, tmpl, raw_boxes)
        summary = compliance_checker.generate_compliance_summary(
            result, tmpl, class_info,
            image_name=_DEMO_IMAGE_NAME.value,
            img_h=img_h, img_w=img_w)
    else:
        # 無合規規則 → 原版數量統計
        class_info = audit_report.summarize_detections(
            boxes_all, scores_all, classes_all, id_map, img_h, img_w)
        summary = audit_report.generate_natural_summary(
            class_info, _DEMO_IMAGE_NAME.value)

    print(f"\n{'='*50}\n{summary}\n{'='*50}")

if __name__ == '__main__':
    app.run(main)
