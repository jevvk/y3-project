import os
import cv2
import math
from random import random
import numpy as np
from functools import reduce

from y3p import PROJECT_DIR

ITERATIONS = 5000
ITERATIONS_PRINT_PERIOD = 1000
LEARNING_RATE = 0.1
TRAINING_SAMPLE_PROBABILITY = 0.75
NORMALIZED_IMAGE_SIZE = 60

def main(config, _):
  classifier_config = config['team_classification']
  out_dir = classifier_config['out_directory']

  weights = np.shape([3, 768])

  print('Loading labels.')

  labels_path = os.path.join(PROJECT_DIR, out_dir, 'labels.npy')
  labels = np.load(labels_path)
  total_labels = len(labels)

  features = np.ndarray((total_labels, 768))

  print('Loading samples.')

  for i in range(total_labels):
    path = os.path.join(PROJECT_DIR, out_dir, 'sample-%d.png' % i)
    image = cv2.imread(path)
    features[i] = calculate_features(image)

  weights = np.random.rand(3, 768)
  training_features, training_labels, testing_features, testing_labels = prepare_data(features, labels)

  test_accuracy(testing_features, testing_labels, weights)

  print('Training started.')
  weights, _ = train_all(training_features, training_labels, weights, ITERATIONS, LEARNING_RATE)
  test_accuracy(testing_features, testing_labels, weights)

  training_features, training_labels, testing_features, testing_labels = prepare_data(features, labels)
  weights, _ = train_all(training_features, training_labels, weights, ITERATIONS, LEARNING_RATE)
  test_accuracy(testing_features, testing_labels, weights)

  training_features, training_labels, testing_features, testing_labels = prepare_data(features, labels)
  weights, _ = train_all(training_features, training_labels, weights, ITERATIONS, LEARNING_RATE)
  print('Done training.')

  test_accuracy(testing_features, testing_labels, weights)
  test_accuracy(features, labels, weights)

def prepare_data(features, labels):
  total_labels = len(labels)

  training_features = []
  testing_features = []
  training_labels = []
  testing_labels = []

  for i in range(total_labels):
    if random() < TRAINING_SAMPLE_PROBABILITY:
      training_features.append(features[i])
      training_labels.append(labels[i])
    else:
      testing_features.append(features[i])
      testing_labels.append(labels[i])

  print('Stats: %d training samples, %d testing samples' % (len(training_labels), len(testing_labels)))

  return training_features, training_labels, testing_features, testing_labels

def test_accuracy(features, labels, weights):
  predictions = classify(features, weights)
  accuracy = np.sum([p == gt for p, gt in zip(predictions, labels)]) / len(labels)

  print('Accuracy is %.2f%%' % (accuracy * 100))

def convert_histogram(hist):
  return [x[0] for x in hist]

def calculate_features(image):
  # hist = cv2.calcHist([image], [0], np.where(mask, 255, 0).astype(np.uint8), [256], [0, 256])
  # image = cv2.resize(image, (NORMALIZED_IMAGE_SIZE, NORMALIZED_IMAGE_SIZE))

  b_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
  b_hist = cv2.normalize(b_hist, b_hist, 1, 0, cv2.NORM_L1)
  b_hist = convert_histogram(b_hist)

  r_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
  r_hist = cv2.normalize(r_hist, r_hist, 1, 0, cv2.NORM_L1)
  r_hist = convert_histogram(r_hist)

  g_hist = cv2.calcHist([image], [2], None, [256], [0, 256])
  g_hist = cv2.normalize(g_hist, g_hist, 1, 0, cv2.NORM_L1)
  g_hist = convert_histogram(g_hist)

  return np.concatenate((r_hist, b_hist, g_hist))

def sigmoid(x):
  """sigmoid(number) -> number"""
  return 1 / (1 + math.exp(-x))

def predict(features, weights):
  """Single class predict
  
  features - array(N, F)
  weights - array(F, 1)

  returns array(N, 1)
  """
  z = np.dot(features, weights)

  return np.array(list(map(sigmoid, z)))

def cost_fn(features, labels, weights):
  """Single class cost function"""
  observations = len(labels)
  predictions = predict(features, weights)

  #Take the error when label=1
  class1_cost = -labels * np.log(predictions)
  #Take the error when label=0
  class2_cost = (1 - labels) * np.log(1 - predictions)

  #Take the sum of both costs
  cost = class1_cost - class2_cost

  #Take the average cost
  cost = cost.sum() / observations

  return cost

def train_all(features, labels, weights, iterations, lr):
  team_a_weights = weights[0]
  team_b_weights = weights[1]
  spectators_weights = weights[2]

  team_a_labels = np.array(list(map(lambda x: 1 if x == 0 else 0, labels)))
  team_b_labels = np.array(list(map(lambda x: 1 if x == 1 else 0, labels)))
  spectators_labels = np.array(list(map(lambda x: 1 if x == 2 else 0, labels)))

  for i in range(iterations):
    team_a_weights = update_weights(features, team_a_labels, weights[0], lr)
    team_b_weights = update_weights(features, team_b_labels, weights[1], lr)
    spectators_weights = update_weights(features, spectators_labels, weights[2], lr)

    if i % ITERATIONS_PRINT_PERIOD == 0:
      cost = cost_fn(features, team_a_labels, team_a_weights)
      cost += cost_fn(features, team_b_labels, team_b_weights)
      cost += cost_fn(features, spectators_labels, spectators_weights)
      cost /= 3

      print('Iteration %d: cost %.2f' % (i, cost))

  weights = [team_a_weights, team_b_weights, spectators_weights]

  cost = cost_fn(features, team_a_labels, team_a_weights)
  cost += cost_fn(features, team_b_labels, team_b_weights)
  cost += cost_fn(features, spectators_labels, spectators_weights)
  cost /= 3

  return weights, cost

def update_weights(features, labels, weights, lr):
  predictions = predict(features, weights)

  #2 Transpose features from (200, 3) to (3, 200)
  # So we can multiply w the (200,1)  cost matrix.
  # Returns a (3,1) matrix holding 3 partial derivatives --
  # one for each feature -- representing the aggregate
  # slope of the cost function across all observations
  gradient = np.dot(np.transpose(features), predictions - labels)

  #3 Take the average cost derivative for each feature
  gradient /= len(features)

  #4 - Multiply the gradient by our learning rate
  gradient *= lr

  #5 - Subtract from our weights to minimize cost
  weights -= gradient

  return weights

def classify(features, weights):
  team_a_weights = weights[0]
  team_b_weights = weights[1]
  spectators_weights = weights[2]

  team_a_pred = predict(features, team_a_weights)
  team_b_pred = predict(features, team_b_weights)
  spectators_pred = predict(features, spectators_weights)

  pred = []

  for predictions in zip(team_a_pred, team_b_pred, spectators_pred):
    pred.append(np.argmax(predictions))

  return pred

