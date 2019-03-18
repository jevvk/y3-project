import cv2
import os
import pickle
import numpy as np

from scipy.optimize import linear_sum_assignment

from y3p import PROJECT_DIR
from y3p.field import Field
from y3p.space.camera import Camera
from y3p.tracker.sample import Sample, distance as sample_distance
from y3p.tracker.track import Track
from y3p.tracker.tracklet import Tracklet
from y3p.player import Player
from y3p.player.feature import distance as descriptor_distance

INF_DISTANCE = 999999
DISTANCE_THRESHOLD = 50

STATE_TRANSITION = np.array([
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]
])

MEASUREMENT_FN = np.array([
  [1.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0, 1.0]
])

STATE_COVAR = np.array([15.0, 15.0, 5.0, 5.0])
MEASUREMENT_COVAR = np.eye(4) * 0.0094

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

      for j, track in enumerate(self._active_tracks):
        dist_cost[i, j] = self._track_distance(camera, tracklet, track)

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
      track = Track(self._time)

      self._tracks.append(track)
      self._active_tracks.append(track)
      self._update_track(camera, -1, tracklet)

  def _update_track(self, camera: int, index: int, tracklet: Tracklet):
    track = self._active_tracks[index]

    assert isinstance(track, Track)

    track.last_time = max(track.last_time, tracklet.last_time)
    track.tracklets.append(tracklet)

    tracklets = sorted(track.tracklets, key=lambda t: t.start_time)
    positions = self._get_tracklets_positions(camera, tracklets)
    track.positions = self._flatten_positions(positions, tracklets)

  def _get_tracklets_positions(self, camera: int, tracklets):
    positions = []

    for t in tracklets:
      t_positions = []

      for sample in t.filtered_samples:
        player = Player([sample.x, sample.y, sample.height, sample.width, None, None], camera)
        position, _ = player.get_position(self._field)

        t_positions.append(position)

      # TODO: use kalman smoother
      positions.append(t_positions)

    return positions

  def _flatten_positions(self, positions, tracklets):
    expanded = list(map(lambda p: [p], positions[0]))
    end_time = tracklets[0].end_time

    for pos, tracklet in zip(positions[1:], tracklets[1:]):
      count = tracklet.start_time - end_time

      if count > 0:
        for i in range(count):
          expanded[tracklet.start_time + i].append(pos[i])
      else:
        for i in range(count):
          expanded.append([])

      for i in range(max(0, count), len(pos)):
        expanded[tracklet.start_time + i].append(pos[i])

      end_time = max(end_time, tracklet.end_time)

    # might need changing because of empty arrays
    flattened = list(map(np.average, expanded))

    return flattened

  def _filter_active_tracks(self):
    self._active_tracks = list(filter(lambda t: t.last_time >= self._time, self._active_tracks))

  def _create_cost_matrix(self, n: int):
    cost = np.ones((n, n)) * INF_DISTANCE

    for i in range(n):
      cost[i, i] = DISTANCE_THRESHOLD

    return cost

  def _track_distance(self, camera: int, tracklet, track: Track):
    dist = 0

    positions0 = self._get_tracklets_positions(camera, [tracklet])[0]
    positions1 = track.positions
    index1 = self._time - track.start_time

    x_diff = positions0[0][0] - positions1[index1][0]
    y_diff = positions0[0][1] - positions1[index1][1]

    initial_dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
    delta_dist = 0

    for index0 in range(1, len(positions0)):
      if len(positions1) >= index1 + index0:
        break

      x_diff0 = positions0[index0][0] - positions0[index0 - 1][0]
      y_diff0 = positions0[index0][1] - positions0[index0 - 1][1]
      x_diff1 = positions1[index1 + index0][0] - positions1[index1 + index0 - 1][0]
      y_diff1 = positions1[index1 + index0][1] - positions1[index1 + index0 - 1][1]
      x_diff = x_diff0 - x_diff1
      y_diff = y_diff0 - y_diff1

      delta_dist += np.sqrt(x_diff ** 2 + y_diff ** 2)

    # dist = initial_dist + np.sqrt(delta_dist)
    dist = initial_dist + delta_dist

    return dist

  def get_tracks(self):
    return self._tracks

def main(config: dict, debug: bool):
  cameras = []
  out_dir = config['out']

  for camera_config in config['views']:
    cameras.append(Camera(camera_config))

  stop = False
  interval = 42
  field = Field(config, debug)
  tracker = MultiViewTracker(field, len(cameras))
  tracklets = []
  current_time = 0

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
      tracklets[i] = list(filter(lambda t: t.start_time == current_time, camera_tracklets))

      new_tracklets.append(to_add)

    if not has_more:
      break

    tracker.forward(new_tracklets)
    current_time += 1

  players = tracker.get_tracks()
  file_path = os.path.join(PROJECT_DIR, out_dir, 'tracks.data')

  with open(file_path, 'wb') as stream:
    pickle.dump(players, stream, protocol=pickle.HIGHEST_PROTOCOL)

  if debug:
    cv2.destroyAllWindows()
    # capture.release()
