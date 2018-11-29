import numpy as np
import cv2

def detect(image, model, scale_factor, min_neighbours):
    image = image.copy()
    classifier = cv2.CascadeClassifier(model)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    players = classifier.detectMultiScale(gray, scale_factor, min_neighbours)

    for (x, y, w, h) in players:
        cv2.rectangle(image, (x,y), (x + w, y + h), (255, 0, 0), 2)

    return image

if __name__ == "__main__":
    import argparse
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument('image', metavar='image', type=str, help = 'path to image')
    ap.add_argument('-c', '--config', required = True, help = 'path to config file')
    ap.add_argument('-o', '--out', required = False, help = 'outfile file name')
    args = ap.parse_args()

    config = None

    with open(args.config, 'r') as f:
        config = yaml.load(f).get('haarcascade')

    scale_factor = config.get('detector').get('scale_factor')
    min_neighbours = config.get('detector').get('min_neighbours')

    image = cv2.imread(args.image)
    newimage = detect(image, config.get('model'), scale_factor, min_neighbours)

    if args.out is not None:
        cv2.imwrite(args.out, newimage)
    else:
        # cv2.imshow('original', image)
        cv2.imshow('detection', newimage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()