import cv2

from y3p.player import Player

# detector = cv2.xfeatures2d.SIFT_create()
# detector2 = cv2.ORB_create()
detector = cv2.KAZE_create()
bf_matcher = cv2.BFMatcher.create()

"""
Given an image of the player and a mask, return the features of the player.
"""
def calculate_descriptor(player: Player):
  image = cv2.cvtColor(player.image, cv2.COLOR_BGR2GRAY)
  kp, des = detector.detectAndCompute(image, player.mask)

  return des

def distance(descriptors1, descriptors2):
  if descriptors1 is None or descriptors2 is None:
    return 9999

  matches = bf_matcher.match(descriptors1, descriptors2)

  if not matches:
    return 9999

  distances = sorted(map(lambda m: m.distance, matches))
  distances = distances[:10]

  avg = sum(distances) / len(distances)

  # print(matches)

  return avg
