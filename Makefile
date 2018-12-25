init:
	pip install -r requirements.txt

download:
	mkdir -p data/model/
	wget -q --show-progress https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -O data/model/mask_rcnn_coco.h5.tmp
	gdown https://drive.google.com/uc?id=1cVWJE1hv1M_KxzyJN6NE52L2JKqjW133 -O data/model/yolov3.h5.tmp

spiroudome:
	python3 -m y3p data/spiroudome/config.yml
