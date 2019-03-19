from random import randint

class Track:
  def __init__(self, start_time: int):
    self.tracklets = []
    self.positions = []
    self.start_time = start_time
    self.last_time = start_time
    self.color = [randint(0, 255), randint(0, 255), randint(0, 255)]
