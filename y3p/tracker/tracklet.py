"""
Contains the following information:
  1. Position relative to court represented as an ellipse
  2. Image features used for identification
"""
class Tracklet:
  _image = None
  _mask = None
  position = None
  features = None

  def __init__(self, position, image, mask):
    self._image = image
    self._mask = mask

    self.position = position
    self.features = self.generate_features()


  """
  Generates features from the given image and mask.
  """
  def generate_features(self):
    pass
