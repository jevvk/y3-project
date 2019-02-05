import cv2
import os
import math
import matplotlib.pyplot as plt
import numpy as np

from y3p import PROJECT_DIR
from y3p.player.classification import TeamClassifier
from y3p.space.camera import Camera

def main(config, detector):
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

  print('Press z for team A, x for team B, n for spectator...')

  teamA = 0
  teamB = 0
  spectators = 0

  while not stop:
    if samples >= max_samples:
      break

    print('%d samples to do.' % (max_samples - samples))

    for i, capture in enumerate(captures):
      ret, frame = capture.read()

      if not ret:
        stop = True
        break

      detections = detector.forward(frame)

      for detection in detections:
        samples += 1
        image = detection[4]
        mask = detection[5]

        img_path = os.path.join(PROJECT_DIR, out_dir, 'sample-%d.png' % samples)
        mask_path = os.path.join(PROJECT_DIR, out_dir, 'sample-%d' % samples)

        cv2.imshow('Detection', image)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('z'):
          labels.append(0)
          teamA += 1
        elif key == ord('x'):
          labels.append(1)
          teamB += 1
        elif key == ord('n'):
          labels.append(2)
          spectators += 1
        else:
          labels.append(2)
          spectators += 1
          print('Key not recognised, assumed as spectator.')

        cv2.destroyAllWindows()
        cv2.imwrite(img_path, image)
        np.save(mask_path, mask)

  labels_path = os.path.join(PROJECT_DIR, out_dir, 'labels')
  np.save(labels_path, labels)

  print('Saved images.')
  print('Stats: %d in team A, %d in team B, %d spectators' % (teamA, teamB, spectators))

  # close video files
  for capture in captures:
    capture.release()

