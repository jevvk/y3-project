from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('config', help = 'path to config file')
args = parser.parse_args()


# for each frame
  # camera_positions = []
  # for each camera
    # detect players
    # perform team classification (A, B, unknown)
    # filter out unknown class
    # map each detection to the position relative to the court
    # append each position to camera_possitions
  # aggregate camera_positions into positions
# calculate statistics from positions
# write results