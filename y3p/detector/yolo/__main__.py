import cv2
import argparse
import yaml

from y3p.detector.yolo import YoloDetector

ap = argparse.ArgumentParser()
ap.add_argument('image', metavar='image', type=str, help = 'path to image')
ap.add_argument('-c', '--config', required = True, help = 'path to config file')
ap.add_argument('-o', '--out', required = False, help = 'outfile file name')
args = ap.parse_args()

config = None

with open(args.config, 'r') as f:
    config = yaml.load(f).get('yolo')

yolov3 = YoloDetector(config)

image = cv2.imread(args.image)
newimage = yolov3.detect(image)

if args.out is not None:
    cv2.imwrite(args.out, newimage)
else:
    # cv2.imshow('original', image)
    cv2.imshow('detection', newimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
