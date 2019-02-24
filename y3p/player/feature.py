import cv2

from y3p.player import Player

detector = cv2.xfeatures2d.SIFT_create()
detector2 = cv2.ORB_create()
detector3 = cv2.KAZE_create()
bf_matcher = cv2.BFMatcher.create()

"""
Given an image of the player and a mask, return the features of the player.
"""
def calculate_features(player: Player):
  image = cv2.cvtColor(player.image, cv2.COLOR_BGR2GRAY)
  kp, des = detector.detectAndCompute(image, player.mask)

  player_image = cv2.drawKeypoints(player.image, kp, des)

  print(des)
  print(player.height, player.width)
  cv2.imshow('sift', player_image)

  kp, des = detector2.detectAndCompute(image, player.mask)
  player_image = cv2.drawKeypoints(player.image, kp, des)

  print(des)
  print(player.height, player.width)

  cv2.imshow('orb', player_image)

  kp, des = detector3.detectAndCompute(image, player.mask)
  player_image = cv2.drawKeypoints(player.image, kp, des)

  print(des)
  print(player.height, player.width)

  cv2.imshow('kaze', player_image)

  cv2.waitKey(0)

  return None, None
  # return (kp, des)

def distance(features1, features2):
  keypoints1, descriptors1 = features1
  keypoints2, descriptors2 = features2

  # print(descriptors1)
  # print(descriptors2)

  matches = bf_matcher.match(descriptors1, descriptors2)

  # print(matches)

  pass
