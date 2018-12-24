import argparse
import yaml

from y3p.logic import main

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('config')
parser.add_argument('--model', choices = ['mrcnn'], default = 'mrcnn')

args = parser.parse_args()

detector = None
config = None

if args.model == 'mrcnn':
  from y3p.detector.maskrcnn import MaskRCNNDetector
  detector = MaskRCNNDetector()
else:
  # TODO yolo detector
  pass

with open(args.config, 'r') as stream:
  config = yaml.load(stream)

  main(config, detector)
