import math
import numpy as np

from y3p.field import Field

class Player:
  def __init__(self, detection, camera: int):
    self.x = int(detection[0])
    self.y = int(detection[1])
    self.height = int(detection[2])
    self.width = int(detection[3])
    self.image = detection[4]
    # self.mask = detection[5]
    self.mask = None
    self.camera = camera
    self.team = -1

    self._calculate_feet_position()

  def _calculate_feet_position(self):
    # might do something different here with the mask
    self.feet_x = int(self.x + self.width / 2)
    self.feet_y = int(self.y + self.height)

  def get_position(self, field: Field, normalise=True):
    feet = [self.feet_x, self.feet_y]
    camera = field.get_camera(self.camera)

    position = field.get_position(self.camera, feet, use_homography=False)
    if normalise: position /= field.size[0]

    T = np.dot(camera.R.T, -camera.T)
    diff = np.array([position[0], 0, position[1]]) + T
    diff = np.dot(camera.R.T, diff)
    diff = np.array([diff[0], diff[1]]) / field.size
    confidence = 5 / dist(diff * diff)

    return position, confidence

def norm(v):
  return v / dist(v)

def dist(v):
  return math.sqrt(np.sum(v*v))
