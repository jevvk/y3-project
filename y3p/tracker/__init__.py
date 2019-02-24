from y3p.field import Field
from y3p.detector import Detector
from y3p.teams import TeamsClassifier

"""
Aggregates the results from the detector to identify and track players throught the
video.
"""
class Tracker:
  def __init__(self, detector: Detector, field: Field, classifier: TeamsClassifier, camera: int):
    self._detector = detector
    self._field = field
    self._classifier = classifier
    self._tracklets = []
    self._camera = camera

  def _get_players(self, frame):
    pass

  def reset(self, frame):
    # create initial tracklets from all the players detected
    # use the field to filter out spectators
    # use team classifier to filter out spectator and tag tracklet
    pass

  def forward(self, frame):
    # predict next move for each position of each tracklet
    # match the new players with the tracklets
    # update the position of tracklets (prediction/observation)
    # create new tracklets for unmatched new players
    # tracklets that didn't have a new match in N frames are moved
    #  and not used for matching anymore
    pass

  def save(self):
    pass
