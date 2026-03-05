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

# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demo for the F-VLM paper (ICLR 2023).

Paper link: https://arxiv.org/abs/2209.15639.

This demo takes sample image and texts, and produce detections using pretrained
F-VLM models.


"""

from collections.abc import Sequence
import functools
import os

from absl import app
from absl import flags
from absl import logging
from demo_utils import audit_report
from demo_utils import input_utils
from demo_utils import vis_utils
import jax
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm
from utils import clip_utils


_DEMO_IMAGE_NAME = flags.DEFINE_string('demo_image_name', 'citrus.jpg',
                                       'The image file name under data/.')
_CATEGORY_NAME_STRING = flags.DEFINE_string(
    'category_name_string', '',
    'Comma separated list of categories, e.g. "person, car, oven".')
_MODEL = flags.DEFINE_enum('model', 'resnet_50',
                           ['resnet_50', 'resnet_50x4', 'resnet_50x16'],
                           'F-VLM model to use.')
_MAX_BOXES_TO_DRAW = flags.DEFINE_integer('max_boxes_to_draw', 25,
                                          'Max number of boxes to draw.')
_MAX_NUM_CLS = flags.DEFINE_integer('max_num_classes', 91,
                                    'Max number of classes users can input.')
_MIN_SCORE_THRESH = flags.DEFINE_float('min_score_thresh', 0.2,
                                       'Min score threshold.')
_TEMPLATE = flags.DEFINE_string(
    'template', '',
    'Template name under templates/ (e.g. construction_site). '
    'If set, runs compliance check instead of plain detection.')

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  clip_text_fn = clip_utils.get_clip_text_fn(_MODEL.value)

  demo_image_path = f'./data/{_DEMO_IMAGE_NAME.value}'
  output_image_path = demo_image_path.replace('data', 'output')
  output_image_path = (
      output_image_path[:-4]
      + f'_{_MODEL.value.replace("resnet_", "r")}'
      + output_image_path[-4:]
  )
  with open(demo_image_path, 'rb') as f:
    np_image = np.array(Image.open(f))

  if _CATEGORY_NAME_STRING.value:
    # Parse string.
    categories = _CATEGORY_NAME_STRING.value.split(',')
  elif _TEMPLATE.value:
    # Load categories from template when no explicit category string given.
    from demo_utils import compliance_checker
    _template_for_categories = compliance_checker.load_template(_TEMPLATE.value)
    categories = _template_for_categories.get('categories', [])
    if not categories:
      raise ValueError(
          f'Template "{_TEMPLATE.value}" has no categories defined.'
      )
  else:
    # Use default text prompts.
    try:
      categories = input_utils.category_dict[_DEMO_IMAGE_NAME.value]
    except KeyError:
      raise KeyError(
          'Default categories do not exist. Please specify!'
      ) from None

  class_clip_features = []
  logging.info('Computing custom category text embeddings.')
  for cls_name in tqdm.tqdm(categories, total=len(categories)):
    cls_feat = clip_text_fn(cls_name)
    class_clip_features.append(cls_feat)

  logging.info('Preparing input data.')
  text_embeddings = np.concatenate(class_clip_features, axis=0)
  embed_path = (
      f'./data/{_MODEL.value.replace("resnet_", "r")}_bg_empty_embed.npy'
  )
  background_embedding, empty_embeddings = np.load(embed_path)
  background_embedding = background_embedding[np.newaxis, Ellipsis]
  empty_embeddings = empty_embeddings[np.newaxis, Ellipsis]
  tile_empty_embeddings = np.tile(
      empty_embeddings, (_MAX_NUM_CLS.value - len(categories) - 1, 1)
  )
  # Concatenate 'background' and 'empty' embeddings.
  text_embeddings = np.concatenate(
      (background_embedding, text_embeddings, tile_empty_embeddings), axis=0
  )
  text_embeddings = text_embeddings[np.newaxis, Ellipsis]
  # Parse the image data.
  parser_fn = input_utils.get_maskrcnn_parser()
  data = parser_fn({'image': np_image, 'source_id': np.array([0])})
  np_data = jax.tree.map(lambda x: x.numpy()[np.newaxis, Ellipsis], data)
  np_data['text'] = text_embeddings
  np_data['image'] = np_data.pop('images')
  labels = np_data.pop('labels')
  image = np_data['image']

  logging.info('Loading saved model.')
  saved_model_dir = f'./checkpoints/{_MODEL.value.replace("resnet_","r")}'
  model = tf.saved_model.load(saved_model_dir)

  logging.info('Computing forward pass.')
  output = model(np_data)

  logging.info('Preparing visualization.')
  id_mapping = {(i + 1): c for i, c in enumerate(categories)}
  id_mapping[0] = 'background'
  for k in range(len(categories) + 2, _MAX_NUM_CLS.value):
    id_mapping[k] = 'empty'
  category_index = input_utils.get_category_index(id_mapping)
  maskrcnn_visualizer_fn = functools.partial(
      vis_utils.visualize_boxes_and_labels_on_image_array,
      category_index=category_index,
      use_normalized_coordinates=False,
      max_boxes_to_draw=_MAX_BOXES_TO_DRAW.value,
      min_score_thresh=_MIN_SCORE_THRESH.value,
      skip_labels=False)
  vis_image = vis_utils.visualize_instance_segmentations(
      output, image, labels['image_info'], maskrcnn_visualizer_fn
  )
  pil_vis_image = Image.fromarray(vis_image, mode='RGB')
  pil_vis_image.save(output_image_path)
  logging.info('Completed saving the output image at %s.', output_image_path)

  # ── 後處理：產生自然語言稽核摘要 ───────────────────────────
  _num_det = int(np.squeeze(output['num_detections']))
  _boxes = np.squeeze(output['detection_boxes'], axis=0)[:_num_det]
  _scores = np.squeeze(output['detection_scores'], axis=0)[:_num_det]
  _classes = np.squeeze(
      output['detection_classes'].astype(np.int32), axis=0
  )[:_num_det]

  # 過濾低信心偵測，與圖上畫的 box 數量保持一致
  _mask = _scores >= _MIN_SCORE_THRESH.value
  _boxes = _boxes[_mask]
  _scores = _scores[_mask]
  _classes = _classes[_mask]

  # 用 image_info 取得縮放後尺寸，與 detection_boxes 座標空間一致
  _image_info = labels['image_info']
  _img_h = int(float(_image_info[0, 0, 0]) * float(_image_info[0, 2, 0]))
  _img_w = int(float(_image_info[0, 0, 1]) * float(_image_info[0, 2, 1]))

  class_info = audit_report.summarize_detections(
      detection_boxes=_boxes,
      detection_scores=_scores,
      detection_classes=_classes,
      id_mapping=id_mapping,
      img_h=_img_h,
      img_w=_img_w,
  )

  if _TEMPLATE.value:
    # 合規稽核模式
    from demo_utils import compliance_checker
    template = compliance_checker.load_template(_TEMPLATE.value)

    # 整理每個類別的 raw box list
    raw_boxes = {}
    for cls_id, box in zip(_classes, _boxes):
      cls_name = id_mapping.get(int(cls_id), f'class_{cls_id}')
      if cls_name in ('background', 'empty'):
        continue
      if cls_name.startswith('class_'):
        logging.warning('Unknown class ID %d encountered during compliance check.', cls_id)
      if cls_name not in raw_boxes:
        raw_boxes[cls_name] = []
      raw_boxes[cls_name].append(box)

    compliance_result = compliance_checker.check_compliance(
        class_info=class_info,
        template=template,
        raw_boxes=raw_boxes,
    )

    natural_summary = compliance_checker.generate_compliance_summary(
        compliance_result=compliance_result,
        template=template,
        class_info=class_info,
        image_name=_DEMO_IMAGE_NAME.value,
        img_h=_img_h,
        img_w=_img_w,
    )
  else:
    # 一般物件偵測模式（原本邏輯）
    natural_summary = audit_report.generate_natural_summary(
        class_info=class_info,
        image_name=_DEMO_IMAGE_NAME.value,
    )

  print('\n' + '=' * 50)
  print(natural_summary)
  print('=' * 50 + '\n')

  base_stem = (
      os.path.splitext(_DEMO_IMAGE_NAME.value)[0]
      + f'_{_MODEL.value.replace("resnet_", "r")}'
  )
  audit_report.save_report(
      class_info=class_info,
      natural_summary=natural_summary,
      image_name=_DEMO_IMAGE_NAME.value,
      model_name=_MODEL.value,
      output_dir='./output',
      file_stem=base_stem,
  )
  logging.info('稽核報表已儲存至 ./output/%s_report.json', base_stem)
  logging.info('稽核摘要已儲存至 ./output/%s_summary.txt', base_stem)


if __name__ == '__main__':
  app.run(main)