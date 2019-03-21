import cv2
import os
import pickle
import numpy as np
from random import randint

from scipy.optimize import linear_sum_assignment

from pykalman import KalmanFilter

from y3p import PROJECT_DIR
from y3p.field import Field
from y3p.space.camera import Camera
from y3p.tracker.sample import Sample, distance as sample_distance
from y3p.tracker.track import Track
from y3p.tracker.tracklet import Tracklet
from y3p.player import Player
from y3p.player.feature import distance as descriptor_distance

INF_DISTANCE = 999999
DISTANCE_THRESHOLD = 75.0
MINIMUM_THRESHOLD = 3
TRACKLET_MIN_LIFE = 4
FIELD_SCALE = 0.67

STATE_TRANSITION = np.array([
  [1.0, 0.0, 1.0, 0.0],
  [0.0, 1.0, 0.0, 1.0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]
])

MEASUREMENT_FN = np.array([
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0]
])

STATE_COVAR = np.array([
  [15.0, 0.0,  0.0, 0.0],
  [0.0,  15.0, 0.0, 0.0],
  [0.0,  0.0,  5.0, 0.0],
  [0.0,  0.0,  0.0, 5.0]
])

MEASUREMENT_COVAR = np.eye(2) * 0.0094

class MultiViewTracker:
  def __init__(self, field: Field, camera_count: int):
    self._field = field
    self._camera_count = camera_count
    self._tracks = []
    self._active_tracks = []
    self._time = 0

  def forward(self, tracklets):
    assert len(tracklets) == self._camera_count

    for i, camera_tracklets in enumerate(tracklets):
      self._forward(i, camera_tracklets)

    self._filter_active_tracks()
    self._time += 1

  def _forward(self, camera: int, tracklets):
    if not tracklets:
      return

    if not self._active_tracks:
      self._create_tracks(camera, tracklets)
      return

    cost = self._create_cost_matrix(len(tracklets))
    dist_cost = np.zeros((len(tracklets), len(self._active_tracks)))

    for i, tracklet in enumerate(tracklets):
      assert isinstance(tracklet, Tracklet)

      positions = self._get_tracklet_positions(tracklet)

      for j, track in enumerate(self._active_tracks):
        dist_cost[i, j] = self._track_distance(camera, tracklet, positions, track)

    cost = np.concatenate((dist_cost, cost), axis=1)
    # calculate matches using hungarian algorithm
    rows, cols = linear_sum_assignment(cost)

    # keep a list of outliers that will be converted to tracklets
    outliers = []

    for i_tracklet, i_track in zip(rows, cols):
      tracklet = tracklets[i_tracklet]

      # add as an outlier if the index doesn't exist
      if i_track >= len(self._active_tracks):
        outliers.append(tracklet)
        continue

      self._update_track(camera, i_track, tracklet)

    self._create_tracks(camera, outliers)
    self._filter_active_tracks()

  def _create_tracks(self, camera: int, tracklets):
    for tracklet in tracklets:
      if tracklet.last_time - tracklet.start_time < TRACKLET_MIN_LIFE:
        continue

      track = Track(self._time)

      self._tracks.append(track)
      self._active_tracks.append(track)
      self._update_track(camera, -1, tracklet)

  def _update_track(self, camera: int, index: int, tracklet: Tracklet):
    track = self._active_tracks[index]

    assert isinstance(track, Track)

    track.last_time = max(track.last_time, tracklet.last_time)
    track.tracklets.append(tracklet)
    track.tracklet_positions.append(self._get_tracklet_positions(tracklet))

    tracklets = sorted(track.tracklets, key=lambda t: t.start_time)
    track.positions = self._flatten_positions(track.tracklet_positions, tracklets)

  def _get_tracklet_positions(self, tracklet: Tracklet):
    positions = []

    for sample in tracklet.filtered_samples:
      player = Player([sample.x, sample.y, sample.height, sample.width, None, None], tracklet.camera)
      position, _ = player.get_position(self._field)

      positions.append(position)

    if len(positions) > 2:
      kf = KalmanFilter(transition_matrices=STATE_TRANSITION, transition_covariance=STATE_COVAR, observation_matrices=MEASUREMENT_FN, observation_covariance=MEASUREMENT_COVAR)
      kf = kf.em(positions, n_iter=5)

      positions = kf.smooth(positions)[0][:, :2]

    return positions

  def _flatten_positions(self, positions, tracklets):
    expanded = list(map(lambda p: [p], positions[0]))
    start_time = tracklets[0].start_time

    for pos, tracklet in zip(positions[1:], tracklets[1:]):
      if len(expanded) < tracklet.start_time - start_time:
        raise 'Data invalid'

      for i, position in enumerate(pos):
        index = tracklet.start_time + i - start_time

        if index < len(expanded):
          expanded[index].append(position)
        else:
          expanded.append([position])

    flattened = list(map(lambda x: np.mean(x, axis=0), expanded))

    return flattened

  def _filter_active_tracks(self):
    self._active_tracks = list(filter(lambda t: t.last_time >= self._time, self._active_tracks))

  def _create_cost_matrix(self, n: int):
    cost = np.ones((n, n)) * INF_DISTANCE

    for i in range(n):
      cost[i, i] = DISTANCE_THRESHOLD / 100

    return cost

  def _track_distance(self, camera: int, tracklet, positions0, track: Track):
    if len(positions0) < TRACKLET_MIN_LIFE:
      return INF_DISTANCE

    # track must not have multiple tracklets belonging to the same camera at the same time
    for t_tracklet in track.tracklets:
      if tracklet.camera != t_tracklet.camera:
        continue

      if tracklet.start_time <= t_tracklet.start_time <= tracklet.last_time:
        return INF_DISTANCE

      if t_tracklet.start_time <= tracklet.start_time <= t_tracklet.last_time:
        return INF_DISTANCE

    dist = 0

    positions1 = track.positions
    index1 = self._time - track.start_time

    x_diff = positions0[0][0] - positions1[index1][0]
    y_diff = positions0[0][1] - positions1[index1][1]

    initial_dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
    delta_dist = 0
    delta_pos = np.array([0.0, 0.0])
    count = 0

    for index0 in range(1, len(positions0)):
      if len(positions1) <= index1 + index0:
        break

      x_diff0 = positions0[index0][0] - positions0[index0 - 1][0]
      y_diff0 = positions0[index0][1] - positions0[index0 - 1][1]
      x_diff1 = positions1[index1 + index0][0] - positions1[index1 + index0 - 1][0]
      y_diff1 = positions1[index1 + index0][1] - positions1[index1 + index0 - 1][1]
      x_diff = x_diff0 - x_diff1
      y_diff = y_diff0 - y_diff1

      delta_dist += np.sqrt(x_diff ** 2 + y_diff ** 2)

    for index0 in range(0, len(positions0)):
      if len(positions1) <= index1 + index0:
        break

      x_diff = positions0[index0][0] - positions1[index1 + index0][0]
      y_diff = positions0[index0][1] - positions1[index1 + index0][1]

      delta_pos += [x_diff, y_diff]
      count += 1

    if count < MINIMUM_THRESHOLD:
      return INF_DISTANCE

    mean_dist = np.sqrt(np.sum(delta_pos ** 2))
    dist = (mean_dist + delta_dist / count) / np.log(count)
    # dist = mean_dist

    # print(len(positions0), len(positions1), count, initial_dist, delta_dist, mean_dist, dist * 100)

    return dist

  def get_tracks(self):
    return self._tracks

  def _create_court(self):
    court_image = np.zeros((self._field.size[1], self._field.size[0], 3), dtype=np.uint8)
    cv2.rectangle(court_image, (int(self._field.size[0] * (0.5 - FIELD_SCALE / 2)), int(self._field.size[1] * (0.5 - FIELD_SCALE / 2))), (int(self._field.size[0] * (0.5 + FIELD_SCALE / 2)), int(self._field.size[1] * (0.5 + FIELD_SCALE / 2))), (255, 255, 255), 5)

    return court_image

  def _draw_position(self, frame, x, y, color):
    overlay = frame.copy()

    x = int(x * self._field.size[0] * FIELD_SCALE + self._field.size[0] * (1 - FIELD_SCALE) / 2)
    y = int(y * self._field.size[0] * FIELD_SCALE + self._field.size[1] * (1 - FIELD_SCALE) / 2)

    cv2.circle(frame, (x, y), 15, color, thickness=-1)
    cv2.circle(overlay, (x, y), 15, color, thickness=-1)
    cv2.circle(overlay, (x, y), 35, color, thickness=-1)
    cv2.addWeighted(overlay, 0.25, frame, 1 - 0.25, 0, frame)

  def draw_courts(self, camera_colors):
    mono = self._create_court()
    multi = self._create_court()
    multi2 = self._create_court()

    active_tracklets = []
    active_tracklets_index = []

    for index, track in enumerate(self._active_tracks):
      assert isinstance(track, Track)

      x, y = track.positions[self._time - track.start_time]

      for tracklet in track.tracklets:
        assert isinstance(tracklet, Tracklet)

        start = tracklet.start_time
        end = start + len(tracklet.filtered_samples) # last_time is actually for the last detection

        if start <= self._time < end:
          active_tracklets.append(tracklet)
          active_tracklets_index.append(index)

      self._draw_position(multi, x, y, track.color)

    for index, tracklet in zip(active_tracklets_index, active_tracklets):
      sample = tracklet.filtered_samples[self._time - tracklet.start_time]
      player = Player([sample.x, sample.y, sample.height, sample.width, None, None], tracklet.camera)
      position, _ = player.get_position(self._field)
      x, y = position

      self._draw_position(mono, x, y, camera_colors[tracklet.camera])
      self._draw_position(multi2, x, y, self._active_tracks[index].color)

    cv2.imshow('mono - tracklets', cv2.resize(mono, (0, 0), fx=450.0/self._field.size[0], fy=450.0/self._field.size[0]))
    cv2.imshow('multi - tracks', cv2.resize(multi, (0, 0), fx=450.0/self._field.size[0], fy=450.0/self._field.size[0]))
    cv2.imshow('multi - tracklets', cv2.resize(multi2, (0, 0), fx=450.0/self._field.size[0], fy=450.0/self._field.size[0]))

def main(config: dict, debug: bool):
  cameras = []
  out_dir = config['out']

  for camera_config in config['views']:
    cameras.append(Camera(camera_config))

  stop = False
  interval = 42
  field = Field(config, False)
  tracker = MultiViewTracker(field, len(cameras))
  tracklets = []
  current_time = 0
  camera_colors = [[randint(0, 255), randint(0, 255), randint(0, 255)] for _ in range(len(cameras))]

  print('Loading tracklets.')

  for camera in cameras:
    file_path = os.path.join(PROJECT_DIR, out_dir, camera.name + '.track.data')
    camera_tracklets = None

    with open(file_path, 'rb') as stream:
      camera_tracklets = pickle.load(stream)

    assert camera_tracklets is not None

    tracklets.append(camera_tracklets)

  print('Running tracker.')

  while not stop:
    new_tracklets = []
    has_more = False

    for i, camera_tracklets in enumerate(tracklets):
      if not camera_tracklets:
        new_tracklets.append([])
        continue

      has_more = True

      to_add = list(filter(lambda t: t.start_time == current_time, camera_tracklets))
      tracklets[i] = list(filter(lambda t: t.start_time != current_time, camera_tracklets))

      new_tracklets.append(to_add)

    if not has_more:
      break

    if debug:
      tracker.draw_courts(camera_colors)
      key = cv2.waitKey(interval) & 0xFF

      if key == 32: # space
        interval = 42 if interval == 0 else 0
      if key == ord('q'):
        return

    tracker.forward(new_tracklets)
    current_time += 1

  print('Saving results.')

  players = tracker.get_tracks()
  file_path = os.path.join(PROJECT_DIR, out_dir, 'tracks.data')

  with open(file_path, 'wb') as stream:
    pickle.dump(players, stream, protocol=pickle.HIGHEST_PROTOCOL)

  if debug:
    cv2.destroyAllWindows()
