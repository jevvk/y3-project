from random import randint

class Tracklet:
  def __init__(self, start_time: int, camera: int):
    self.samples = []
    self.filtered_samples = []
    self.start_time = start_time
    self.last_time = start_time
    self.color = [randint(0, 255), randint(0, 255), randint(0, 255)]
    self.camera = camera
