import os
import cv2
import pickle
import numpy as np

from random import randint

from y3p import PROJECT_DIR

from y3p.detector import Detector
from y3p.field import Field
from y3p.player import Player
from y3p.player.feature import calculate_descriptor
from y3p.space.camera import Camera
from y3p.teams import TeamsClassifier
from y3p.tracker.sample import Sample

FIELD_OUT_THRESHOLD = 0.1
FIELD_WINDOW_WIDTH = 470
FIELD_SCALE = 0.67
FRAME_SCALE = 1
MIN_SHAPE_RATIO = 0.75
MAX_SHAPE_RATIO = 3.5

"""
Aggregates the results from the detector to identify and track players throught the
video.
"""
class PlayerDetector:
  def __init__(self, detector: Detector, field: Field, classifier: TeamsClassifier, camera: int):
    self._detector = detector
    self._field = field
    self._classifier = classifier
    self._samples = []
    self._camera = camera
    self._color = (randint(75, 255), randint(75, 255), 0)
    self._time = 0

  def _get_players(self, frame):
    return self._detector.forward(frame)

  def _convert_to_instances(self, players):
    return list(map(lambda x: Sample(x, self._time, calculate_descriptor(x)), players))

  def _filter_out_odd_shapes(self, detections):
    # TODO: use distance from camera for better thresholds
    _detections = []

    for detection in detections:
      player = Player(detection, self._camera)
      ratio = player.height / player.width

      if MIN_SHAPE_RATIO <= ratio <= MAX_SHAPE_RATIO:
        _detections.append(detection)

    return _detections

  def _filter_out_spectators(self, detections):
    players = []

    for detection in detections:
      player = Player(detection, self._camera)
      position, _ = player.get_position(self._field)

      # check if the detection is on the ground
      if position is None:
        continue

      # check is inside field
      if not self._field.is_inside(position):
        continue

      # check detection is a player (0 is A, 1 is B)
      # if self._classifier.classify(player.image) > 1:
      #   continue

      player.team = self._classifier.classify(player.image)
      players.append(player)

    return players

  def _draw_detections_and_court(self, frame, players):
    frame = frame.copy()
    field_width = self._field.size[0]
    field_height = self._field.size[1]
    width = FIELD_WINDOW_WIDTH
    height = int(FIELD_WINDOW_WIDTH * field_height / field_width)

    court_image = np.zeros((height, width, 3), dtype=np.uint8)

    top_left = (int(width * (1 - FIELD_SCALE) / 2), int(height * (1 - FIELD_SCALE) / 2))
    bottom_right = (int(width * (1 + FIELD_SCALE) / 2), int(height * (1 + FIELD_SCALE) / 2))

    cv2.rectangle(court_image, top_left, bottom_right, (255, 255, 255), 1)

    for player in players:
      position, confidence = player.get_position(self._field)
      overlay = court_image.copy()
      court_x, court_y = position
      court_x = int(width * (court_x / field_width * FIELD_SCALE + (1 - FIELD_SCALE) / 2))
      court_y = int(height * (court_y / field_height * FIELD_SCALE + (1 - FIELD_SCALE) / 2))

      color = None

      if player.team == 0:
        color = self._color
      elif player.team == 1:
        color = (self._color[1], 0, self._color[0])
      else:
        color = (0, 255, 255)

      cv2.rectangle(frame, (player.x, player.y), (player.x + player.width, player.y + player.height), color, 2)
      cv2.circle(frame, (player.feet_x, player.feet_y), 3, color, thickness=-1)
      cv2.circle(court_image, (court_x, court_y), 3, color, thickness=-1)
      cv2.circle(overlay, (court_x, court_y), 3, color, thickness=-1)
      cv2.circle(overlay, (court_x, court_y), int(3 * confidence), color, thickness=-1)
      cv2.addWeighted(overlay, 0.25, court_image, 1 - 0.25, 0, court_image)

    frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

    cv2.imshow('camera %d frame' % self._camera, frame)
    cv2.imshow('camera %d court' % self._camera, court_image)

  def forward(self, frame, debug: bool):
    players = self._get_players(frame)
    players = self._filter_out_odd_shapes(players)
    players = self._filter_out_spectators(players)

    if debug:
      self._draw_detections_and_court(frame, players)

    self._samples.append(self._convert_to_instances(players))
    self._time += 1

  def get_tracklets(self):
    return self._samples

def main(config: dict, detector: Detector, debug: bool):
  cameras = []
  out_dir = config['out']

  try:
    os.mkdir(os.path.join(PROJECT_DIR, out_dir))
  except:
    pass

  for camera_config in config['views']:
    cameras.append(Camera(camera_config))

  team_classifier = TeamsClassifier(config)
  field = Field(config, debug)

  stop = False

  for i, camera in enumerate(cameras):
    if stop:
      break

    print('Running detector on %s.' % camera.name)

    player_detector = PlayerDetector(detector, field, team_classifier, i)
    capture = cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file))

    while True:
      ret, frame = capture.read()

      if not ret:
        break

      player_detector.forward(frame, debug)

      if cv2.waitKey(33) & 0xFF == ord('q'):
        stop = True
        break

    if stop:
      capture.release()
      cv2.destroyAllWindows()
      break

    print('Finished %s.' % camera.name)

    players = player_detector.get_tracklets()
    file_path = os.path.join(PROJECT_DIR, out_dir, camera.name + '.detect.data')

    with open(file_path, 'wb') as stream:
      pickle.dump(players, stream, protocol=pickle.HIGHEST_PROTOCOL)

    cv2.destroyAllWindows()
    capture.release()
