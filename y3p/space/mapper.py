"""
Maps detections relative to the camera to positions relative the court.
"""

"""
Given the camera parameters and detections, this function returns an ellipse
and likelyhood for each detection. Each ellipse has the following properties:
  1. belongs to the plane of the court
  2. represents an area where the player detected is most likely located on
      the court 
"""
def map(camera, detections):
  pass

# Line and plane intersection
# T - camera translation
# V - vector from camera to projected point
# N - plane normal vector
# O - point on plane
# L - line from camera to projected point
# L(x) = T + xV

# solution condition
# N . L(s) = 0

# solution
# s = (-N . T) / (N . V)
