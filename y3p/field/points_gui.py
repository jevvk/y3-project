import cv2
import os
import sys
import pickle
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

MAX_FRAMES_TO_AVERAGE = 40
WINDOW_NAME = 'points'
WINDOW_SCALE = 0.75

original_frame = None
point = None

def main(config, detector, debug):
  global point, original_frame

  captures = []
  cameras = []
  views = []
  all_field_points = []

  field_config = config['field']
  max_char = ord(max(field_config['points'].keys()))

  assert max_char < ord('q')

  for camera_config in config['views']:
    camera = Camera(camera_config)

    cameras.append(camera)
    captures.append(cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file)))
    views.append(None)

  print('Loaded cameras config.')
  print('Calculating average of %d frames for each camera...' % MAX_FRAMES_TO_AVERAGE)

  for index, capture in enumerate(captures):
    for _ in range(MAX_FRAMES_TO_AVERAGE):
      ret, frame = capture.read()

      if not ret:
        break

      # better not to undistort
      # frame = cv2.undistort(frame, camera.matrix, camera.kc)

      if views[index] is None:
        views[index] = (cv2.split(frame.astype('float')), 1)
      else:
        avg, total = views[index]
        b, g, r = cv2.split(frame.astype('float'))

        b_avg = avg[0] + b
        g_avg = avg[1] + g
        r_avg = avg[2] + r

        views[index] = ((b_avg, g_avg, r_avg), total + 1)

  for capture in captures:
    capture.release()

  print('Showing results.')
  print('To select a corner, press left mouse button on the image.')
  print('Press a, b, ..., %s for each corner visibile. Press r to reset, space to continue.' % chr(max_char))

  cv2.namedWindow(WINDOW_NAME)
  cv2.setMouseCallback(WINDOW_NAME, mouse_event)

  for i, view in enumerate(views):
    print('Showing camera %d.' % i)

    b, g, r = view[0]
    b /= view[1]
    g /= view[1]
    r /= view[1]

    field_points = {}
    frame = cv2.merge((b, g, r)).astype('uint8')
    original_frame = frame
    cv2.imshow(WINDOW_NAME, cv2.resize(frame, (0, 0), fx=WINDOW_SCALE, fy=WINDOW_SCALE))

    stop = False

    while not stop:
      key = cv2.waitKey(0) & 0xFF

      if key == 32: # space
        stop = True
      elif key == ord('q'):
        sys.exit(0)
      elif key == ord('r'):
        print('Points for this camera have been reset.')
        field_points = {}
      elif ord('a') <= key <= max_char:
        if point is None:
          print('No point selected.')
          continue

        print('Selected point %s as (%d, %d).' % (str(chr(key)).upper(), point[0], point[1]))

        index = key - ord('a')
        field_points[chr(key)] = point
        point = None

        cv2.imshow(WINDOW_NAME, cv2.resize(frame, (0, 0), fx=WINDOW_SCALE, fy=WINDOW_SCALE))

    all_field_points.append(field_points)

  print('Saving points.')

  out_dir = config['field']['out_directory']
  file_path = os.path.join(PROJECT_DIR, out_dir, 'points.data')

  try:
    os.mkdir(os.path.join(PROJECT_DIR, out_dir))
  except:
    pass

  with open(file_path, 'wb') as stream:
    pickle.dump(all_field_points, stream, protocol=pickle.HIGHEST_PROTOCOL)

def mouse_event(event, x, y, flags, param):
  global point, original_frame

  if event is not cv2.EVENT_FLAG_LBUTTON:
    return

  point = (int(x / WINDOW_SCALE), int(y / WINDOW_SCALE))
  frame = original_frame.copy()

  cv2.circle(frame, point, 3, (0, 0, 255), -1)
  cv2.imshow(WINDOW_NAME, cv2.resize(frame, (0, 0), fx=WINDOW_SCALE, fy=WINDOW_SCALE))
