import numpy as np

from y3p.teams.train import load_weights, single_classify

class TeamsClassifier:
  def __init__(self, config):
    self.weights = load_weights(config)

  def classify(self, image):
    return single_classify(image, self.weights)
