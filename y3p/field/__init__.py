import os
import cv2
import math
import pickle
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

Y_NORMAL = np.array([0, 1, 0])

class Field:
  _cameras = []
  _camera_points = []
  _points = None

  def __init__(self, config: dict):
    self._load_cameras(config['views'])
    self._load_field(config['field'])
    self._load_camera_points(config['field'])
    self._position_cameras()

  def _load_cameras(self, cameras: list):
    print('Field: loading cameras.')

    for camera_config in cameras:
      self._cameras.append(Camera(camera_config))

  def _load_field(self, field_config: dict):
    print('Field: loading field points.')

    self._points = field_config['points']

  def _load_camera_points(self, field_config: dict):
    print('Field: loading camera points.')

    file_path = os.path.join(PROJECT_DIR, field_config['out_directory'], 'points.data')

    with open(file_path, 'rb') as stream:
      self._camera_points = pickle.load(stream)

    # print(self._camera_points)

  def _position_cameras(self):
    print('Field: positioning cameras.')

    for camera, camera_points in zip(self._cameras, self._camera_points):
      print('Field: calculating for %s.' % camera.name)

      object_points = []
      image_points = []

      for key, _ in self._points.items():
        if key not in camera_points:
          continue

        object_points.append(self._points[key])
        image_points.append(camera_points[key])

      print('Field: got %d points.' % len(object_points))

      object_points = np.array(object_points, dtype=np.float64)
      image_points = np.array(image_points, dtype=np.float64)

      assert len(image_points) > 3 and len(object_points) > 3

      retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera.matrix, camera.kc)

      assert retval

      camera.R = cv2.Rodrigues(rvec)[0]
      camera.T = tvec.reshape((-1))

      # print(camera)

  def camera_count(self):
    return len(self._cameras)

  def get_projection(self, camera_index: int, point: tuple):
    assert 0 <= camera_index < len(self._cameras)

    camera = self._cameras[camera_index]
    point = np.array(point, dtype=np.float64).reshape((1, 1, 3))
    point = cv2.projectPoints(point, camera.R, camera.T, camera.matrix, camera.kc)
    point = point[0].reshape((2))

    return point

  def get_position(self, camera_index: int, point: tuple):
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

    # find where line intersects the y plane
    s = np.dot(-1 * Y_NORMAL, origin) / np.dot(Y_NORMAL, direction)

    return origin + s * direction

def norm(v):
  return v / dist(v)

def dist(v):
  return math.sqrt(np.sum(v*v))
