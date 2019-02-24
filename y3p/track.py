import cv2
import os
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

from y3p.field import Field
from y3p.player import Player
from y3p.tracker import Tracker
from y3p.detector import Detector
from y3p.teams import TeamsClassifier

def main(config: dict, detector: Detector, debug: bool):
  cameras = []

  for camera_config in config['views']:
    cameras.append(Camera(camera_config))

  team_classifier = TeamsClassifier(config)
  field = Field(config, debug)

  stop = False

  for i in range(len(cameras)):
    if stop:
      break

    tracker = Tracker(detector, field, team_classifier, i)
    camera = cameras[i]
    capture = cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file))
    ret, frame = capture.read()

    assert ret

    cv2.imshow('frame', frame)
    tracker.reset(frame)

    while True:
      ret, frame = capture.read()

      if not ret:
        break

      cv2.imshow('frame', frame)
      tracker.forward(frame)

      if cv2.waitKey(33) & 0xFF == ord('q'):
        stop = True
        break

    cv2.destroyAllWindows()
    capture.release()
