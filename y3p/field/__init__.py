import os
import cv2
import math
import pickle
from random import randint
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera_extended import CameraExtended

Y_NORMAL = np.array([0, 1, 0])
ACCURACY_TEST_SAMPLES = 500

class Field:
  _cameras = []
  _camera_points = []
  _points = None
  size = None

  def __init__(self, config: dict, debug: bool):
    self._load_cameras(config['views'])
    self._load_field(config['field'])
    self._load_camera_points(config['field'])
    self._position_cameras()
    self._calculate_field_size()
    self._calculate_homography()
    self._test_accuracy(debug)

  def _load_cameras(self, cameras: list):
    print('Field: loading cameras.')

    for camera_config in cameras:
      self._cameras.append(CameraExtended(camera_config))

  def _load_field(self, field_config: dict):
    print('Field: loading field points.')

    self._points = field_config['points']

  def _load_camera_points(self, field_config: dict):
    print('Field: loading camera points.')

    file_path = os.path.join(PROJECT_DIR, field_config['out_directory'], 'points.data')

    with open(file_path, 'rb') as stream:
      self._camera_points = pickle.load(stream)

  def _position_cameras(self):
    print('Field: positioning cameras.')

    for camera, camera_points in zip(self._cameras, self._camera_points):
      print('Field: calculating for %s.' % camera.name)

      object_points = []
      image_points = []

      for key, _ in self._points.items():
        if key not in camera_points:
          continue

        object_points.append(point_2d_3d(self._points[key]))
        image_points.append(camera_points[key])

      print('Field: got %d points.' % len(object_points))

      object_points = np.array(object_points, dtype=np.float64)
      image_points = np.array(image_points, dtype=np.float64)

      assert len(image_points) > 3 and len(object_points) > 3

      retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera.matrix, camera.kc, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False)

      assert retval

      camera.R = cv2.Rodrigues(rvec)[0]
      camera.T = tvec.reshape((-1))

      print('Field: calculated R and t for %s.' % camera.name)

  def _calculate_field_size(self):
    self.size = [0, 0]

    for point in self._points.values():
      self.size[0] = max(self.size[0], point[0])
      self.size[1] = max(self.size[1], point[1])

  def _calculate_homography(self):
    for i in range(len(self._cameras)):
      camera = self._cameras[i]

      positions = []
      ground_truth = []

      for point, key in zip(self._camera_points[i].values(), self._camera_points[i].keys()):
        positions.append(self.get_position(i, point))
        ground_truth.append(self._points[key])

      positions = np.array(positions, dtype=np.float32)
      ground_truth = np.array(ground_truth, dtype=np.float32)

      H, mask = cv2.findHomography(positions, ground_truth)

      assert H is not None

      camera.H = H
      camera.mask = mask
  
  def _test_accuracy(self, debug: bool):
    for i in range(len(self._cameras)):
      camera = self._cameras[i]
      proj_points = []
      reproj_points = []

      for s in range(ACCURACY_TEST_SAMPLES):
        x = randint(0, camera.width)
        y = randint(0, camera.height)

        proj_points.append((x, y))
      
      for s in range(ACCURACY_TEST_SAMPLES):
        x = randint(0, self.size[0])
        y = randint(0, self.size[1])

        reproj_points.append((x, y))

      self._test_projection_accuracy(i, proj_points, debug)
      self._test_reprojection_accuracy(i, reproj_points, False, debug)
      self._test_reprojection_accuracy(i, reproj_points, True, debug)

  def _test_projection_accuracy(self, i, points, debug: bool):
    camera = self._cameras[i]
    distances = []

    for point in points:
      position = self.get_position(i, point)

      if position is None:
        continue

      position = [position[0], 0, position[1]]
      projection = self.get_projection(i, position)
      diff = np.array(point, dtype=np.uint32) - np.array(projection, dtype=np.uint32)

      distances.append(100 * dist(diff / [camera.width, camera.height]))

    mean = np.mean(distances)
    std = np.std(distances)

    print('Field: (projection test) %s has %.3f%% mean and %.3f%% standard deviation error.' % (self._cameras[i].name, mean, std))

  def _test_reprojection_accuracy(self, i, points, use_homography: bool, debug: bool):
    camera = self._cameras[i]
    distances = []

    for point, key in zip(self._camera_points[i].values(), self._camera_points[i].keys()):
      position = np.array(self.get_position(i, point, use_homography=use_homography), dtype=np.float32)

      diff = position - np.array(self._points[key])
      distances.append(100 * dist(diff / self.size))

      if debug:
        print(camera.name, point, self._points[key], self.get_position(i, point, use_homography=use_homography), self.get_projection(i, (self._points[key][0], 0, self._points[key][1])), diff)

    mean = np.mean(distances)
    std = np.std(distances)

    print('Field: (samples=%d, homography=%r, reprojection field points) %s has %.3f%% mean and %.3f%% standard deviation error.' % (len(distances), use_homography, self._cameras[i].name, mean, std))

    distances = []

    for point in points:
      position = np.array(point, dtype=np.float32)
      point = np.array([position[0], 0, position[1]])
      projection = self.get_projection(i, point)

      if not 0 <= projection[0] <= self.size[0]:
        continue

      if not 0 <= projection[0] <= self.size[1]:
        continue

      position = self.get_position(i, projection, use_homography=use_homography)

      if position is None:
        continue

      diff = position - np.array([point[0], point[2]])

      distances.append(100 * dist(diff / self.size))

    mean = np.mean(distances)
    std = np.std(distances)

    print('Field: (samples=%d, homography=%r, reprojection test) %s has %.3f%% mean and %.3f%% standard deviation error.' % (len(distances), use_homography, self._cameras[i].name, mean, std))

  def camera_count(self):
    return len(self._cameras)
  
  def get_camera(self, camera_index):
    assert 0 <= camera_index < len(self._cameras)

    return self._cameras[camera_index]

  def get_projection(self, camera_index: int, point: tuple):
    assert 0 <= camera_index < len(self._cameras)

    camera = self._cameras[camera_index]
    point = np.array(point, dtype=np.float64).reshape((1, 1, 3))
    point = cv2.projectPoints(point, camera.R, camera.T, camera.matrix, camera.kc)
    point = point[0].reshape((2))

    return point

  def get_position(self, camera_index: int, point: tuple, use_homography: bool = False):
    assert 0 <= camera_index < len(self._cameras)

    camera = self._cameras[camera_index]
    # undistort point then convert to homogeneos
    point = np.array(point, dtype=np.float64).reshape((1, 1, 2))
    point = cv2.undistortPoints(point, camera.matrix, camera.kc)
    point = cv2.convertPointsToHomogeneous(point)
    point = point.reshape((3))

    # apply the camera transformations in reverse
    origin = -np.dot(camera.R.T, camera.T)
    direction = np.dot(camera.R.T, point)
    direction = norm(direction)

    denominator = np.dot(Y_NORMAL, direction)

    if denominator == 0:
      return None

    # find where line intersects the y plane
    s = np.dot(-1 * Y_NORMAL, origin) / denominator

    # line is directed
    if s < 0:
      return None

    position = point_3d_2d(origin + s * direction)

    if use_homography:
      position = cv2.perspectiveTransform(position.reshape((1, 1, 2)), camera.H)
      position = position.reshape((2))

    return position

def point_2d_3d(point):
  return np.array([point[0], 0, point[1]])

def point_3d_2d(point):
  return np.array([point[0], point[2]])

def norm(v):
  return v / dist(v)

def dist(v):
  return math.sqrt(np.sum(v*v))
