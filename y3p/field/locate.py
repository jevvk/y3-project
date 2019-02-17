import os
import pickle
import numpy as np

from y3p import PROJECT_DIR
from y3p.space.camera import Camera

def get_field_y(cameras, field_dir):
  # read field points
  # for each field point
    # 

  field_points = None
  file_path = os.path.join(PROJECT_DIR, field_dir, 'points.data')

  with open(file_path, 'rb') as stream:
    field_points = pickle.load(stream)

  c0 = cameras[0]
  c1 = cameras[1]
  p0 = field_points[0]['i']
  p1 = field_points[1]['i']

  # r0 = normalise(np.array([(p0[0] - c0.cc[0]) / c0.width, (p0[1] - c0.cc[1]) / c0.height, 1]))
  # r1 = normalise(np.array([(p1[0] - c1.cc[0]) / c1.width, (p1[1] - c1.cc[1]) / c1.height, 1]))
  r0 = normalise(np.array([p0[0] - c0.cc[0], c0.height - p0[1] - c0.cc[1], c0.fc[0]]))
  r1 = normalise(np.array([p1[0] - c1.cc[0], c1.height - p1[1] - c1.cc[1], c1.fc[0]]))

  c0d = np.array([c0.R[0][2], c0.R[1][2], c0.R[2][2]])
  c1d = np.array([c1.R[0][2], c1.R[1][2], c1.R[2][2]])

  r0 = np.dot(c0.R, r0)
  r1 = np.dot(c1.R, r1)

  s = 0
  t = 0

  dc = dist(c0.T - c1.T)
  direction = -0.2

  print('c0d, c1d', c0d, c1d)
  print('r0, r1', r0, r1)
  # print(c0.T / 100, get_point(c0.T, c0d, 100) / 100)
  # print(c1.T / 100, get_point(c1.T, c1d, 100) / 100)

  for i in range(10000):
    if i%200 == 0: print('s, t', s, t)

    p = get_point(c0.T, r0, s)
    q = get_point(c1.T, r1, t)
    pq = q - p
    d_pq = dist(pq)

    if i%200 == 0: print('distance', dist(pq))

    # if dc < d_pq:
    #   direction *= -1

    s0 = np.dot(pq, r0)
    s1 = np.dot(pq, r1)

    if i%200 == 0: print('s0, s1', s0, s1)

    if i % 2 == 0:
      s += abs(s0) * direction
    else:
      t += abs(s1) * direction
    
    dc = d_pq

  p = get_point(c0.T, r0, s)
  q = get_point(c1.T, r1, t)

  print(s, t)
  print(p, q)

  print(dist(q - p))
  print(dist(np.array(c0.T) - np.array(c1.T)))

  return 1.0

def get_point(v, u, t):
  return v + u * t

def normalise(v):
  return v / dist(v)

def dist(v):
  return np.sqrt(np.sum(v**2))

