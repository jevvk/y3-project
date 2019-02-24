import pykalman

tcov = None
ocov = None

"""
Contains the following information:
  1. Position relative to court represented as an ellipse
  2. Image features used for identification
"""
class Tracklet:
  def __init__(self):
    self._filter = pykalman.KalmanFilter(transition_covariance=tcov, observation_covariance=ocov)
