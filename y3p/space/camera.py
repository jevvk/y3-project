import numpy as np

class Camera:
  def __init__(self, config):
    self.name = config['name']
    self.file = config['file']
    self.index = config['index']
    self.width = config['width']
    self.height = config['height']

    calibration = config['calibration']

    self.T = np.array(calibration['T'])
    self.R = np.array(calibration['R']).reshape(3, 3)
    self.kc = np.array(calibration['kc'])
    self.fc = np.array(calibration['fc'])
    self.cc = np.array(calibration['cc'])

    self.matrix = np.array([ self.fc[0], 0 , self.cc[0], 0 , self.fc[1], self.cc[1], 0, 0, 1 ])
    self.matrix = self.matrix.reshape(3, 3)
