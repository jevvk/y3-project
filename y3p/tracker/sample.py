import numpy as np

from y3p.player import Player

class Sample:
  def __init__(self, player: Player, time: int, descriptor=None):
    self.x = player.x
    self.y = player.y
    self.height = player.height
    self.width = player.width
    self.camera = player.camera
    self.team = player.team
    self.time = time
    self.descriptor = descriptor
  
def distance(sample0: Sample, sample1: Sample):
  # TODO: could take into account the player appearance
  center0 = np.array([sample0.x + sample0.width / 2, sample0.y + sample0.height / 2])
  center1 = np.array([sample1.x + sample1.width / 2, sample1.y + sample1.height / 2])

  return dist(center0 - center1)

def dist(v):
  return np.sqrt(np.sum(v*v))
