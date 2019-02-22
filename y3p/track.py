import cv2
import os
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

from y3p.field import Field
from y3p.teams.train import load_weights, single_classify

FIELD_SCALE = 0.67
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

  field = Field(config, debug)
  # teams_classifier = load_weights(config)

  while True:
    ret, frame = captures[CAMERA].read()

    if not ret or not ret:
      break

    court_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(court_image, (int(WINDOW_WIDTH * (0.5 - FIELD_SCALE / 2)), int(WINDOW_HEIGHT * (0.5 - FIELD_SCALE / 2))), (int(WINDOW_WIDTH * (0.5 + FIELD_SCALE / 2)), int(WINDOW_HEIGHT * (0.5 + FIELD_SCALE / 2))), (255, 255, 255), 2)

    # new_frame = cv2.undistort(frame.copy(), camera.matrix, camera.kc)
    detections = detector.forward(frame)

    for detection in detections:
      x, y, height, width, image, _, _ = detection
      # player_class = single_classify(image, teams_classifier)

      feet_x = int(x + width / 2)
      feet_y = int(y + height)

      position = field.get_position(CAMERA, (feet_x, feet_y))

      if position is None:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
        continue

      court_x, court_y = position
      court_x /= field.size[0]
      court_y /= field.size[1]

      # skip people too far away from the court
      if abs(court_x - 0.5) > 1 / FIELD_SCALE / 2:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
        continue

      if abs(court_y - 0.5) > 1 / FIELD_SCALE / 2:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
        continue

      cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

      if debug: print(court_x, court_y)

      court_x = int(WINDOW_WIDTH * (court_x * FIELD_SCALE + FIELD_SCALE / 2))
      court_y = int(WINDOW_HEIGHT * (court_y * FIELD_SCALE + FIELD_SCALE / 2))
      court_x = int(court_x * 0.67 + WINDOW_WIDTH * 0.33 / 2)
      court_y = int(court_y * 0.67 + WINDOW_HEIGHT * 0.33 / 2)

      if debug: print(court_x, court_y, WINDOW_WIDTH, WINDOW_HEIGHT)

      cv2.circle(court_image, (court_x, court_y), 3, (0, 0, 255), thickness=-1)

    cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.67, fy=0.67))
    cv2.imshow('positions', court_image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
      break

  cv2.destroyAllWindows()

  for capture in captures:
    capture.release()
