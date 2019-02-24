import numpy as np

class Camera:
  def __init__(self, config):
    self.name = config['name']
    self.file = config['file']
    # self.index = config['index']
    self.width = config['width']
    self.height = config['height']

    calibration = config['calibration']

    self.T = np.array(calibration['T'], dtype=np.float64)
    self.R = np.array(calibration['R'], dtype=np.float64).reshape(3, 3)
    self.kc = np.array(calibration['kc'], dtype=np.float64)
    self.fc = np.array(calibration['fc'], dtype=np.float64)
    self.cc = np.array(calibration['cc'], dtype=np.float64)

    self.matrix = np.array([ self.fc[0], 0, self.cc[0], 0, self.fc[1], self.cc[1], 0, 0, 1 ])
    self.matrix = self.matrix.reshape(3, 3)

  def __str__(self):
    return 'Camera(T=%s, R=%s, kc=%s, fc=%s, cc=%s)' % (str(self.T), str(self.R), str(self.kc), str(self.fc), str(self.cc))
