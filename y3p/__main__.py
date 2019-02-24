import sys
import yaml
import argparse

from y3p.track import main as track_main
from y3p.field.points_gui import main as field_points_main
from y3p.teams.detect import main as teams_detect_main
from y3p.teams.train import main as teams_train_main
from y3p.demo import main as demo_main

def main(args):
  detector = None
  config = None

  # since mask rcnn uses python2.7 and yolo uses python3
  # imports are done inside the if
  if args.model == 'mrcnn':
    from y3p.detector.maskrcnn import MaskRCNNDetector
    detector = MaskRCNNDetector()
  elif args.model == 'yolo':
    from y3p.detector.yolo import YoloDetector
    detector = YoloDetector(config)
  elif args.model == 'none':
    pass
  else:
    # this shouldn't happen
    sys.exit(1)

  with open(args.config, 'r') as stream:
    config = yaml.load(stream)

    if args.mode == 'track':
      track_main(config, detector, args.debug)
    elif args.mode == 'field-points':
      field_points_main(config, detector, args.debug)
    elif args.mode == 'teams-detect':
      teams_detect_main(config, detector, args.debug)
    elif args.mode == 'teams-train':
      teams_train_main(config, detector, args.debug)
    elif args.mode == 'demo':
      demo_main(config, detector, args.debug)
    else:
      # this shouldn't happen
      sys.exit(1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Short sample app')

  parser.add_argument('config')
  parser.add_argument('--model', choices=['mrcnn', 'none'], default='none')
  parser.add_argument('--mode', choices=['teams-detect', 'teams-train', 'field-points', 'track', 'demo'], default='track')
  parser.add_argument('--debug', action='store_true')
  parser.set_defaults(debug=False)

  args = parser.parse_args()

  main(args)
