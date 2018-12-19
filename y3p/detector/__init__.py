"""
Generic detector class.
"""
class Detector:
  def __init__(self):
    pass
  
  """
  Given a frame, this function returns an array representing the bounding box
  of the detected players, as well as the image and mask of the detection.

  The array elements are represented as tuples of the form:
    1. x coordinate of the top-left corner of the bounding box
    2. y coordinate of the top-left corner of the bounding box
    3. height of the bounding box
    4. width of the bounding box
    5. image of the player
    6. (optional) the mask representing the area of the image belonging to the detected
       player
  """
  def forward(self, frame):
    pass
