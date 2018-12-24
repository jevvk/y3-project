from mrcnn.config import Config

class InferenceConfig(Config):
  NAME = "coco"
  NUM_CLASSES = 1 + 80 # COCO has 80 classes
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
