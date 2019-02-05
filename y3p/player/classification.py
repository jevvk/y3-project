"""
Linear logistic regression classifier.
"""
class TeamClassifier:
  def __init__(self):
    self.classifier = None

  def load(self, file):
    # TODO
    pass
  
  def save(self, file):
    # TODO
    pass

  """
  Given a list of images and masks, return a number for each player. Number
  0 represents no team, 1 represents team A and 2 represents team B.
  """
  def forward(self, players, masks):
    # TODO
    pass
  
  """
  Given a list of images and masks, use local search with outliers to classify
  into 2 teams and then use those to train the logistic classifier.
  """
  def train(self, players):
    # TODO
    pass
