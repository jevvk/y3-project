import cv2
import os
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

from y3p.field import Field
from y3p.player import Player
from y3p.tracker import Tracker
from y3p.teams.train import load_weights, single_classify
from y3p.player.feature import calculate_features, distance

FIELD_SCALE = 0.67
FIELD_OUT_THRESHOLD = 0.1
WINDOW_WIDTH = 470
WINDOW_HEIGHT = 250
CAMERA = 0

def main(config, detector, debug):
  captures = []
  cameras = []

  for camera_config in config['views']:
    camera = Camera(camera_config)

    cameras.append(camera)
    captures.append(cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file)))

  team_classifier = None
  field = Field(config, debug)
  # tracker = Tracker(detector, field, team_classifier)
  # teams_classifier = load_weights(config)

  while True:
    ret, frame0 = captures[0].read()
    ret, frame1 = captures[1].read()
    ret, frame2 = captures[2].read()

    if not ret:
      break

    court_image = np.zeros((field.size[1], field.size[0], 3), dtype=np.uint8)
    cv2.rectangle(court_image, (int(field.size[0] * (0.5 - FIELD_SCALE / 2)), int(field.size[1] * (0.5 - FIELD_SCALE / 2))), (int(field.size[0] * (0.5 + FIELD_SCALE / 2)), int(field.size[1] * (0.5 + FIELD_SCALE / 2))), (255, 255, 255), 5)

    detections0 = detector.forward(frame0)
    detections1 = detector.forward(frame1)
    detections2 = detector.forward(frame2)

    draw_detections(0, frame0, court_image, field, detections0, (0, 0, 255), debug)
    draw_detections(1, frame1, court_image, field, detections1, (0, 255, 255), debug)
    draw_detections(2, frame2, court_image, field, detections2, (255, 255, 0), debug)

    # cv2.imshow('frame0', cv2.resize(frame0, (0, 0), fx=0.67, fy=0.67))
    # cv2.imshow('frame1', cv2.resize(frame1, (0, 0), fx=0.67, fy=0.67))
    # cv2.imshow('frame2', cv2.resize(frame2, (0, 0), fx=0.67, fy=0.67))
    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)
    cv2.imshow('positions', cv2.resize(court_image, (0, 0), fx=0.25, fy=0.25))

    if cv2.waitKey(33) & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()

  for capture in captures:
    capture.release()

def draw_detections(camera: int, frame, court_image, field, detections, color, debug):
  # x = calculate_features(Player(detections[2], camera))
  # y = calculate_features(Player(detections[3], camera))
  # # print(x)
  # distance(x, y)

  for detection in detections:
    player = Player(detection, camera)
    position, confidence = player.get_position(field)

    if position is None:
      cv2.rectangle(frame, (player.x, player.y), (player.x + player.width, player.y + player.height), (0, 0, 0), 2)
      continue

    court_x, court_y = position

    # skip people too far away from the court
    if abs(court_x / field.size[0] - 0.5) - 0.5 > FIELD_OUT_THRESHOLD:
      cv2.rectangle(frame, (player.x, player.y), (player.x + player.width, player.y + player.height), (0, 0, 0), 2)
      continue

    if abs(court_y / field.size[1] - 0.5) - 0.5 > FIELD_OUT_THRESHOLD:
      cv2.rectangle(frame, (player.x, player.y), (player.x + player.width, player.y + player.height), (0, 0, 0), 2)
      continue

    overlay = court_image.copy()

    court_x = int(court_x * FIELD_SCALE + field.size[0] * (1 - FIELD_SCALE) / 2)
    court_y = int(court_y * FIELD_SCALE + field.size[1] * (1 - FIELD_SCALE) / 2)

    cv2.rectangle(frame, (player.x, player.y), (player.x + player.width, player.y + player.height), color, 2)
    cv2.circle(frame, (player.feet_x, player.feet_y), 3, color, thickness=-1)
    cv2.circle(court_image, (int(court_x), int(court_y)), 15, color, thickness=-1)
    cv2.circle(overlay, (int(court_x), int(court_y)), 15, color, thickness=-1)
    cv2.circle(overlay, (int(court_x), int(court_y)), int(15 * confidence), color, thickness=-1)
    cv2.addWeighted(overlay, 0.25, court_image, 1 - 0.25, 0, court_image)

    # court_x = int(WINDOW_WIDTH * court_x / 3 + WINDOW_WIDTH / 6)
    # court_y = int(WINDOW_HEIGHT * court_y / 3 + WINDOW_WIDTH / 6)

    # court_x = int(WINDOW_WIDTH * (court_x * FIELD_SCALE + FIELD_SCALE / 2))
    # court_y = int(WINDOW_HEIGHT * (court_y * FIELD_SCALE + FIELD_SCALE / 2))
    # court_x = int(court_x * 0.67 + WINDOW_WIDTH * 0.33 / 2)
    # court_y = int(court_y * 0.67 + WINDOW_HEIGHT * 0.33 / 2)
