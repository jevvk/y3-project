import math
import numpy as np

class Player:
  def __init__(self, detection, camera):
    self.x = detection[0]
    self.y = detection[1]
    self.height = detection[2]
    self.width = detection[3]
    self.image = detection[4]
    # self.mask = detection[5]
    self.mask = None
    self.camera = camera

    self._calculate_feet_position()

  def _calculate_feet_position(self):
    # might do something different here with the mask
    self.feet_x = int(self.x + self.width / 2)
    self.feet_y = int(self.y + self.height)

  def get_position(self, field):
    feet = [self.feet_x, self.feet_y]
    position = field.get_position(self.camera, feet, use_homography=False)
    camera = field.get_camera(self.camera)
    T = np.dot(camera.R.T, -camera.T)
    diff = position + np.array([T[0], T[2]])
    diff /= field.size
    confidence = 1 + np.sum(diff * diff)
    # confidence = 1 / dist(diff)

    return position, confidence

def norm(v):
  return v / dist(v)

def dist(v):
  return math.sqrt(np.sum(v*v))
