from tracklet import Tracklet

class Tracker():
  def __init__(self, detector):
    self._detector = detector
    self._tracklets = []
  
  def forward(self, image):
    players = self._detector.detect(image)
    new_tracklets = [ Tracklet(player) for player in players ]

    self._tracklets = self._match(new_tracklets)

  def _match(self, tracklets):
    pass

