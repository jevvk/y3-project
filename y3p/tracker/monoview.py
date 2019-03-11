import cv2
import os
import pickle
import numpy as np

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from y3p import PROJECT_DIR
from y3p.field import Field
from y3p.space.camera import Camera
from y3p.tracker.sample import Sample, distance as sample_distance
from y3p.tracker.tracklet import Tracklet
from y3p.player import Player
from y3p.player.feature import distance as descriptor_distance

INF_DISTANCE = 999999
DISTANCE_THRESHOLD = 500
TRACKLET_MAX_LIFETIME = 4
FILTER_HISTORY_SIZE = 4
FIELD_SCALE = 0.67

# no velocity state
# STATE_TRANSITION = np.eye(4, dtype=np.float32)
# MEASUREMENT_FN = np.eye(4, dtype=np.float32)
# STATE_COVAR = np.array([10.0, 10.0, 2.0, 2.0])
# MEASUREMENT_COVAR = np.eye(4) * 0.094

# with velocity state
STATE_TRANSITION = np.array([
  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.8, 0.0],
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
])

MEASUREMENT_FN = np.array([
  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
])

STATE_COVAR = np.array([15.0, 15.0, 5.0, 5.0, 2.5, 2.5])
MEASUREMENT_COVAR = np.eye(4) * 0.0094

class MonoViewTracker:
  def __init__(self, camera: Camera):
    self._camera = camera
    self._tracklets = []
    self._active_tracklets = []
    self._time = 0
    self._last_samples = []

  def _create_tracklets(self, samples):
    for sample in samples:
      assert isinstance(sample, Sample)

      tracklet = Tracklet(self._time)

      tracklet.samples.append(sample)
      tracklet.filtered_samples.append(sample)

      self._active_tracklets.append(tracklet)
      self._tracklets.append(tracklet)

  def _update_tracklets(self, already_matched):
    # add the predictions of the kalman filters to all active tracklets
    for i, tracklet in enumerate(self._active_tracklets):
      # skip tracklets already matched
      if i in already_matched:
        continue

      tracklet.filtered_samples.append(self._get_next_sample(tracklet))

  def _match_samples_and_update(self, samples):
    self._last_samples = samples

    # if the are no active tracklets, just create new ones
    if not self._active_tracklets:
      self._create_tracklets(samples)
      return

    # find matches between current samples and observations made by the
    # kalman filter, update matched tracklets and create new tracklets for
    # unmatched new samples

    # create cost matrix
    cost = self._create_cost_matrix(len(samples))
    dist_cost = np.zeros((len(samples), len(self._active_tracklets)))

    for i, sample in enumerate(samples):
      assert isinstance(sample, Sample)

      for j, tracklet in enumerate(self._active_tracklets):
        dist_cost[i, j] = self._tracklet_distance(tracklet, sample)

    cost = np.concatenate((dist_cost, cost), axis=1)
    # calculate matches using hungarian algorithm
    rows, cols = linear_sum_assignment(cost)

    # check matches length
    assert len(rows) == len(samples)
    assert len(cols) == len(samples)

    # keep a list of outliers that will be converted to tracklets
    outliers = []

    for i_sample, i_tracklet in zip(rows, cols):
      # add as an outlier if the index doesn't exist
      if i_tracklet >= len(self._active_tracklets):
        outliers.append(samples[i_sample])
        continue

      sample = samples[i_sample]
      tracklet = self._active_tracklets[i_tracklet]

      # linting
      assert isinstance(sample, Sample)
      assert isinstance(tracklet, Tracklet)

      last_sample = tracklet.samples[-1]

      # append samples to tracklet
      tracklet.samples.append(sample)
      tracklet.filtered_samples.append(sample)
      # tracklet.filtered_samples.append(self._get_next_sample(tracklet))

      # update time
      tracklet.last_time = self._time

    self._update_tracklets(cols)
    self._create_tracklets(outliers)

  def _tracklet_distance(self, tracklet: Tracklet, sample: Sample):
    tracklet_sample = self._get_next_sample(tracklet)
    center_dist = sample_distance(sample, tracklet_sample)
    sample_dist = 1

    lam = 0.8
    t = 5
    fac = lam

    t_samples = reversed(tracklet.samples[:t])

    for t_sample in t_samples:
      sample_dist += fac * descriptor_distance(t_sample.descriptor, sample.descriptor)
      fac *= lam

    center_dist *= center_dist
    sample_dist = np.log(sample_dist)
    dist = sample_dist * center_dist

    return dist

  def _filter_tracklets(self):
    self._active_tracklets = list(filter(lambda t: self._time - t.last_time <= TRACKLET_MAX_LIFETIME, list(self._active_tracklets)))

  def _create_cost_matrix(self, n: int):
    cost = np.ones((n, n)) * INF_DISTANCE

    for i in range(n):
      cost[i, i] = DISTANCE_THRESHOLD

    return cost

  def _get_next_sample(self, tracklet: Tracklet):
    samples = tracklet.filtered_samples[-FILTER_HISTORY_SIZE:]
    sample = samples[0]
    samples = list(map(lambda s: [s.x, s.y, s.height, s.width], samples[1:]))

    kf = KalmanFilter(dim_x=6, dim_z=4)
    kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=1.0, block_size=3)
    kf.x = np.array([sample.x, sample.y, sample.height, sample.width, 0, 0])
    kf.F = STATE_TRANSITION
    kf.H = MEASUREMENT_FN
    kf.P *= STATE_COVAR
    kf.R = MEASUREMENT_COVAR

    means, covs, means_pred, covs_pred = kf.batch_filter(samples)

    # print(list(map(lambda sample: [sample.x, sample.y, sample.height, sample.width], samples)))
    # print(kf.x, len(samples))

    # create a detection from filter; image and mask are None
    detection = list(kf.x[0:4]) + [None, None]
    player = Player(detection, self._camera)

    return Sample(player, self._time)

  def forward(self, samples):
    self._match_samples_and_update(samples)
    self._filter_tracklets()

    self._time += 1

  def get_tracklets(self):
    return self._tracklets

  def draw_active_tracklets(self, frame):
    for tracklet in self._active_tracklets:
      assert isinstance(tracklet, Tracklet)

      # sample = self._get_next_sample(tracklet)
      sample = tracklet.filtered_samples[-1]
      cv2.rectangle(frame, (sample.x, sample.y), (sample.x + sample.width, sample.y + sample.height), tracklet.color, 2)

  def draw_samples(self, frame):
    for sample in self._last_samples:
      assert isinstance(sample, Sample)

      cv2.rectangle(frame, (sample.x, sample.y), (sample.x + sample.width, sample.y + sample.height), (0, 0, 255), 2)

  def draw_court(self, frame, field: Field, camera: int):
    for tracklet in self._active_tracklets:
      assert isinstance(tracklet, Tracklet)

      sample = tracklet.filtered_samples[-1]
      player = Player([sample.x, sample.y, sample.height, sample.width, None, None], camera)

      position, confidence = player.get_position(field)

      if position is None or not field.is_inside(position):
        continue

      overlay = frame.copy()

      court_x, court_y = position
      court_x = int(court_x * FIELD_SCALE + field.size[0] * (1 - FIELD_SCALE) / 2)
      court_y = int(court_y * FIELD_SCALE + field.size[1] * (1 - FIELD_SCALE) / 2)

      cv2.circle(frame, (court_x, court_y), 15, tracklet.color, thickness=-1)
      cv2.circle(overlay, (court_x, court_y), 15, tracklet.color, thickness=-1)
      cv2.circle(overlay, (court_x, court_y), int(15 * confidence), tracklet.color, thickness=-1)
      cv2.addWeighted(overlay, 0.25, frame, 1 - 0.25, 0, frame)
  
def main(config: dict, debug: bool):
  cameras = []
  out_dir = config['out']

  for camera_config in config['views']:
    cameras.append(Camera(camera_config))

  stop = False
  interval = 42

  # cameras = cameras[2:]

  for i, camera in enumerate(cameras):
    if stop:
      break

    frames = None
    file_path = os.path.join(PROJECT_DIR, out_dir, camera.name + '.detect.data')

    with open(file_path, 'rb') as stream:
      frames = pickle.load(stream)

    capture = None
    tracker = MonoViewTracker(camera)
    field = Field(config, debug)

    if debug:
      capture = cv2.VideoCapture(os.path.join(PROJECT_DIR, camera.file))

    for samples in frames:
      tracker.forward(samples)

      if debug:
        ret, frame = capture.read()

        if not ret:
          break

        frame2 = frame.copy()
        court_image = np.zeros((field.size[1], field.size[0], 3), dtype=np.uint8)
        cv2.rectangle(court_image, (int(field.size[0] * (0.5 - FIELD_SCALE / 2)), int(field.size[1] * (0.5 - FIELD_SCALE / 2))), (int(field.size[0] * (0.5 + FIELD_SCALE / 2)), int(field.size[1] * (0.5 + FIELD_SCALE / 2))), (255, 255, 255), 5)

        # cv2.imshow(camera.name, frame)

        tracker.draw_active_tracklets(frame)
        tracker.draw_samples(frame2)
        tracker.draw_court(court_image, field, i)

        cv2.imshow('%s - tracked' % camera.name, frame)
        cv2.imshow('%s - samples' % camera.name, frame2)
        cv2.imshow('%s - positions' % camera.name, cv2.resize(court_image, (0, 0), fx=450.0/field.size[0], fy=450.0/field.size[0]))

        key = cv2.waitKey(interval) & 0xFF

        if key == 32: # space
          interval = 42 if interval == 0 else 0
        if key == ord('q'):
          stop = True
          break

    players = tracker.get_tracklets()
    file_path = os.path.join(PROJECT_DIR, out_dir, camera.name + '.track.data')

    with open(file_path, 'wb') as stream:
      pickle.dump(players, stream, protocol=pickle.HIGHEST_PROTOCOL)

    if debug:
      cv2.destroyAllWindows()
      capture.release()
