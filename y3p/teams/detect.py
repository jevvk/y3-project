import cv2
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

FRAMES_TO_SKIP = 36

def main(config, detector):
  """Sample humans from videos and manually classify into team A, team B, and spectators"""
  classifier_config = config['team_classification']
  views = config['views']
  max_samples = classifier_config['samples']
  out_dir = classifier_config['out_directory']

  samples = 0
  cameras = []
  captures = []
  labels = []

  try:
    os.mkdir(os.path.join(PROJECT_DIR, out_dir))
  except:
    pass

  # open video files
  for camera_config in views:
    camera = Camera(camera_config)

    cameras.append(camera)
    captures.append(cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file)))

  stop = False

  print('Press z for team A, x for team B, n for spectator, s to skip, q to exit...')

  team_a = 0
  team_b = 0
  spectators = 0
  frames_to_skip = 0

  while not stop:
    if samples >= max_samples:
      break

    for _, capture in enumerate(captures):
      ret, frame = capture.read()

      if not ret:
        stop = True
        break

      if frames_to_skip > 0:
        frames_to_skip -= 1
        continue

      detections = detector.forward(frame)

      for detection in detections:
        image = detection[4]
        mask = detection[5]

        img_path = os.path.join(PROJECT_DIR, out_dir, 'sample-%d.png' % samples)
        mask_path = os.path.join(PROJECT_DIR, out_dir, 'sample-%d.npy' % samples)

        cv2.imshow('Detection', image)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('z'):
          # team a
          labels.append(0)
          team_a += 1
        elif key == ord('x'):
          # team b
          labels.append(1)
          team_b += 1
        elif key == ord('n'):
          # spectator
          labels.append(2)
          spectators += 1
        elif key == ord('s'):
          continue
        elif key == ord('q'):
          print('Exiting.')
          sys.exit(0)
        else:
          print('Key not recognised, assumed as spectator. Skipping.')
          continue

        cv2.destroyAllWindows()
        cv2.imwrite(img_path, image)
        np.save(mask_path, mask)

        samples += 1

      frames_to_skip = FRAMES_TO_SKIP
      
      print('%d samples to do.' % (max_samples - samples))

  labels_path = os.path.join(PROJECT_DIR, out_dir, 'labels.npy')
  np.save(labels_path, labels)
  
  print('%d samples in todal.' % samples)
  print('Stats: %d in team A, %d in team B, %d spectators' % (team_a, team_b, spectators))

  # close video files
  for capture in captures:
    capture.release()

