"""
Usage: python3 -m y3p.detector.maskrcnn image [detection_index]
"""


import sys
import os
import skimage
import numpy as np
import datetime

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines

from y3p.detector.maskrcnn import MaskRCNNDetector

def apply_mask(image, mask, color, alpha=0.5):
  for c in range(3):
    image[:, :, c] = np.where(mask == 1,
                              image[:, :, c] *
                              (1 - alpha) + alpha * color[c] * 255,
                              image[:, :, c])
  return image

image_path = sys.argv[1]
image = skimage.io.imread(image_path)
image = image[:, :, 0:3]

detector = MaskRCNNDetector()

a = datetime.datetime.now()

detections = detector.forward(image)

b = datetime.datetime.now()
c = b - a

print(c)

n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
x, y, height, width, frame, mask, score = detections[n]

masked_frame = frame.astype(np.uint32).copy()
_, ax = plt.subplots(1)

height, width = image.shape[:2]
ax.set_ylim(height + 10, -10)
ax.set_xlim(-10, width + 10)
ax.axis('off')
ax.set_title('detection ' + str(n) + ' p' + str(score))

color = (1, 0, 0)

p = patches.Rectangle((x, y), width, height, linewidth=2,
                      alpha=0.7, linestyle="dashed",
                      edgecolor=color, facecolor='none')
ax.add_patch(p)

ax.text(x, y + 8, 'person', color='w', size=11, backgroundcolor="none")

apply_mask(masked_frame, mask, color)

padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
padded_mask[1:-1, 1:-1] = mask
contours = find_contours(padded_mask, 0.5)

for verts in contours:
  verts = np.fliplr(verts) - 1
  p = patches.Polygon(verts, facecolor="none", edgecolor=color)

ax.add_patch(p)
ax.imshow(masked_frame.astype(np.uint8))

plt.show()
