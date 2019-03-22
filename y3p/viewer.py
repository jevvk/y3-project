import cv2
import os
import pickle
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera
from y3p.tracker.track import Track
from y3p.tracker.tracklet import Tracklet
from y3p.field import Field

FIELD_SCALE = 0.67
TRACK_MIN_LIFE = 12

def create_court(config, field):
  image_path = config['field_background']
  image = cv2.imread(os.path.join(PROJECT_DIR, image_path))
  court = np.zeros((field.size[1], field.size[0], 3), dtype=np.uint8)

  assert image is not None

  x0 = int(field.size[0] * (0.5 - FIELD_SCALE / 2))
  y0 = int(field.size[1] * (0.5 - FIELD_SCALE / 2))
  x1 = int(field.size[0] * (0.5 + FIELD_SCALE / 2))
  y1 = int(field.size[1] * (0.5 + FIELD_SCALE / 2))

  width = x1 - x0
  height = y1 - y0

  image = cv2.resize(image, (width, height))
  court[y0:y1, x0:x1] = image

  return court

def get_active_tracklets(tracks, time):
  tracklets = []

  for track in tracks:
    assert isinstance(track, Track)

    for tracklet in track.tracklets:
      assert isinstance(tracklet, Tracklet)

      if 0 <= time - tracklet.start_time < len(tracklet.filtered_samples):
        tracklet.color = track.color
        tracklets.append(tracklet)

  return tracklets

def draw_court_tracks(frame, tracks, field: Field, time: int):
  overlay = frame.copy()

  for track in tracks:
    assert isinstance(track, Track)

    position = track.positions[time - track.start_time]
    court_x, court_y = position

    court_x = court_x * field.size[0]
    court_y = court_y * field.size[0]
    court_x = int(court_x * FIELD_SCALE + field.size[0] * (1 - FIELD_SCALE) / 2)
    court_y = int(court_y * FIELD_SCALE + field.size[1] * (1 - FIELD_SCALE) / 2)

    cv2.circle(frame, (court_x, court_y), 15, track.color, thickness=-1)
    cv2.circle(overlay, (court_x, court_y), 15, track.color, thickness=-1)
    cv2.circle(overlay, (court_x, court_y), 35, track.color, thickness=-1)

  cv2.addWeighted(overlay, 0.25, frame, 1 - 0.25, 0, frame)

def draw_camera_tracks(frame, camera: Camera, camera_index: int, field: Field, tracks, time: int):
  for track in tracks:
    assert isinstance(track, Track)

    position = track.positions[time - track.start_time]
    position = [position[0] * field.size[0], 0, position[1] * field.size[0]]
    position = field.get_projection(camera_index, position)
    x, y = position
    x = int(x)
    y = int(y)

    if x < 0 or x > camera.width:
      continue

    if y < 0 or y > camera.height:
      continue

    cv2.circle(frame, (x, y), 5, track.color, thickness=-1)

def draw_camera_tracklets(frame, camera: Camera, camera_index: int, tracklets, time: int):
  tracklets = list(filter(lambda t: t.camera == camera_index, tracklets))
  overlay = frame.copy()

  for tracklet in tracklets:
    assert isinstance(tracklet, Tracklet)

    filtered_sample = tracklet.filtered_samples[time - tracklet.start_time]
    sample = next((s for s in tracklet.samples if s.time == time), None)

    cv2.rectangle(frame, (filtered_sample.x, filtered_sample.y), (filtered_sample.x + filtered_sample.width, filtered_sample.y + filtered_sample.height), tracklet.color, 2)

    if sample is not None:
      cv2.rectangle(overlay, (filtered_sample.x, filtered_sample.y), (filtered_sample.x + filtered_sample.width, filtered_sample.y + filtered_sample.height), tracklet.color, 2)
      cv2.rectangle(overlay, (sample.x, sample.y), (sample.x + sample.width, sample.y + sample.height), tracklet.color, 6)

  cv2.addWeighted(overlay, 0.25, frame, 1 - 0.25, 0, frame)

def main(config, detector, debug):
  cameras = []
  captures = []
  out_dir = config['out']

  for camera_config in config['views']:
    camera = Camera(camera_config)
    capture = cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file))

    cameras.append(camera)
    captures.append(capture)

  tracks = None
  active_tracks = []
  field = Field(config, False)
  court_orig = create_court(config, field)
  file_path = os.path.join(PROJECT_DIR, out_dir, 'tracks.data')

  with open(file_path, 'rb') as stream:
    tracks = pickle.load(stream)

  time = 0
  interval = 42
  stop = False

  while not stop:
    court = court_orig.copy()

    to_add = list(filter(lambda t: t.start_time == time, tracks))
    tracks = list(filter(lambda t: t.start_time != time, tracks))

    active_tracks = active_tracks + to_add
    active_tracks = list(filter(lambda t: t.last_time >= time, active_tracks))
    active_tracks = list(filter(lambda t: t.last_time - t.start_time >= TRACK_MIN_LIFE, active_tracks))
    active_tracklets = get_active_tracklets(active_tracks, time)

    draw_court_tracks(court, active_tracks, field, time)
    cv2.imshow('court', cv2.resize(court, (0, 0), fx=450.0/field.size[0], fy=450.0/field.size[0]))

    for i, camera, capture in zip(range(len(cameras)), cameras, captures):
      ret, frame = capture.read()

      if not ret:
        break

      draw_camera_tracks(frame, camera, i, field, active_tracks, time)
      draw_camera_tracklets(frame, camera, i, active_tracklets, time)

      cv2.imshow(camera.name, frame)

    key = cv2.waitKey(interval) & 0xFF

    if key == 32: # space
      interval = 42 if interval == 0 else 0
    if key == ord('q'):
      stop = True
      break

    time += 1
