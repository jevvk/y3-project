import numpy as np

class Player:
  def __init__(self, detection, camera):
    self.x = detection[0]
    self.y = detection[1]
    self.height = detection[2]
    self.width = detection[3]
    self.image = detection[4]
    self.mask = detection[5]
    self.camera = camera

    self._set_feet_position()

  def _set_feet_position(self):
    self.feet_x = int(self.x + self.width / 2)
    self.feet_y = int(self.y + self.height)

  def get_position(self, field):
    feet = [self.feet_x, self.feet_y]
    position = field.get_position(self.camera, feet)

    return position
