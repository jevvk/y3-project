import cv2
import numpy as np

from y3p.detector import Detector

class YoloDetector(Detector):
  def __init__(self, config):
    self._conf_threshold = config.get('detector').get('conf_threshold')
    self._nms_threshold = config.get('detector').get('nms_threshold')
    self._scale = config.get('detector').get('scale')
    self._input_h = config.get('detector').get('input').get('height')
    self._input_w = config.get('detector').get('input').get('width')

    self._config_path = config.get('model').get('config')
    self._classes_path = config.get('model').get('classes')
    self._weights_path = config.get('model').get('weights')

    self._open()

  def _open(self):
    self._net = cv2.dnn.readNet(self._weights_path, self._config_path)

    with open(self._classes_path, 'r') as f:
      self._classes = [line.strip() for line in f.readlines()]

    self._colors = np.random.uniform(0, 255, size=(len(self._classes), 3))
  
  def detect(self, image):
    Width = image.shape[1]
    Height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image, self._scale, (self._input_h, self._input_w), (0, 0, 0), True, crop = False)
    self._net.setInput(blob)
    outs = self._net.forward(self._get_output_layers())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, self._conf_threshold, self._nms_threshold)

    # don't change the read image
    image = image.copy()

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        self._draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    return image

  
  def _get_output_layers(self):
    layer_names = self._net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
  
    return output_layers

  # Warning: causes side-effects
  def _draw_prediction(self, image, cid, confidence, x0, y0, x1, y1):
    label = str(self._classes[cid])
    color = self._colors[cid]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, '%s (%.2f)' % (label, confidence), (x0 - 10, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
