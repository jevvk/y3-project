import cv2
import numpy as np

from y3p.space.camera import Camera

def main(config, detector):
  # for each frame
    # camera_positions = []
    # for each camera
      # undistort camera
      # detect players
      # perform team classification (A, B, unknown)
      # filter out unknown class
      # map each detection to the position relative to the court
      # append each position to camera_positions
    # aggregate camera_positions into positions
  # calculate statistics from positions
  # write results

  camera2_config = config['views'][0]
  camera = Camera(camera2_config)

  capture = cv2.VideoCapture(camera.file)

  while True:
    ret, frame = capture.read()

    if not ret: break

    new_frame = cv2.undistort(frame, camera.matrix, camera.kc)

    cv2.imshow('new_frame', new_frame)
    cv2.waitKey(1)

  capture.release()

  pass
