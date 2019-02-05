import os
import numpy as np
import tensorflow as tf

from y3p.detector.maskrcnn.mrcnn import model

from y3p import PROJECT_DIR
from y3p.detector import Detector
from y3p.detector.maskrcnn.config import InferenceConfig
from y3p.detector.maskrcnn.classes import class_names

MODEL_PATH = os.path.join(PROJECT_DIR, 'data/model/mask_rcnn_coco.h5')
LOGS_PATH = os.path.join(PROJECT_DIR, 'logs')

class MaskRCNNDetector(Detector):
  model = None
  config = None

  def __init__(self):
    self.config = InferenceConfig()
    self.model = model.MaskRCNN(mode="inference", model_dir=LOGS_PATH, config=self.config)

    self.model.load_weights(MODEL_PATH, by_name=True)

  def forward(self, frame):
    result = self.model.detect([frame], verbose = 0)[0]

    detections = []

    boxes = result['rois']
    masks = result['masks']
    class_ids = result['class_ids']
    scores = result['scores']

    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0] == scores.shape[0]

    for i in range(boxes.shape[0]):
      # skip unwanted detections for now
      if class_names[class_ids[i]] != 'person':
        continue

      # skip invalid boundng boxes
      if not np.any(boxes[i]):
        continue

      # might want to skip scores < 0.5
      
      x1, y1, x2, y2 = boxes[i]

      width = abs(x1 - x2)
      height = abs(y1 - y2)

      image = frame[x1:x2, y1:y2, :]
      mask = masks[x1:x2, y1:y2, i]
      score = scores[i]

      detections.append((x1, y1, height, width, image, mask, score))
    
    return detections




