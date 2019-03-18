import cv2

from y3p.player import Player

MAX_DISTANCE = 99

detector = cv2.KAZE_create()
matcher = cv2.BFMatcher.create()

"""
Given an image of the player and a mask, return the features of the player.
"""
def calculate_descriptor(player: Player):
  image = cv2.cvtColor(player.image, cv2.COLOR_BGR2GRAY)
  _, des = detector.detectAndCompute(image, player.mask)

  return des

def distance(descriptors1, descriptors2):
  if descriptors1 is None and descriptors2 is None:
    return 0

  if descriptors1 is None or descriptors2 is None:
    return MAX_DISTANCE

  matches = matcher.match(descriptors1, descriptors2)

  if not matches:
    return MAX_DISTANCE

  distances = sorted(map(lambda m: m.distance, matches))
  distances = distances[:10]

  avg = sum(distances) / len(distances)

  return avg
