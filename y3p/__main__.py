import sys
import yaml
import argparse

from y3p.tracker.detector import main as detect_main
from y3p.tracker.monoview import main as monoview_main
from y3p.field.points_gui import main as field_points_main
from y3p.teams.detect import main as teams_detect_main
from y3p.teams.train import main as teams_train_main
from y3p.demo import main as demo_main

def main(args):
  detector = None
  config = None

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

    if args.mode == 'stage1':
      f_y = input('Do you wish to set field points? [y/N]: ')

      if not f_y or f_y == 'y' or f_y == 'Y':
        field_points_main(config, args.debug)

      t_y = input('Do you wish to label detections for team classifier? [y/N]: ')

      if not t_y or t_y == 'y' or t_y == 'Y':
        teams_detect_main(config, detector, args.debug)

      t_y = input('Do you wish to train team classifier? [y/N]: ')

      if not t_y or t_y == 'y' or t_y == 'Y':
        teams_train_main(config, args.debug)
    elif args.mode == 'stage2':
      detect_main(config, detector, args.debug)
    elif args.mode == 'stage3':
      monoview_main(config, args.debug)
    elif args.mode == 'stage4':
      print('Not implemented.')
      sys.exit(0)
    elif args.mode == 'demo':
      demo_main(config, detector, args.debug)
    else:
      # this shouldn't happen
      sys.exit(1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Third year project')

  parser.add_argument('config')
  parser.add_argument('--model', choices=['mrcnn', 'none'], default='none')
  parser.add_argument('--mode', choices=['demo', 'stage1', 'stage2', 'stage3', 'stage4'], default='demo')
  parser.add_argument('--debug', action='store_true')
  parser.set_defaults(debug=False)

  arguments = parser.parse_args()

  main(arguments)
