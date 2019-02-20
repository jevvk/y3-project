init:
	pip install -r requirements.txt

download:
	mkdir -p data/model/
	wget -q --show-progress https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -O data/model/mask_rcnn_coco.h5
	gdown https://drive.google.com/uc?id=1cVWJE1hv1M_KxzyJN6NE52L2JKqjW133 -O data/model/yolov3.h5

apidis-field-points:
	python -m y3p --model none --mode field-points data/apidis/config.yml

spiroudome-field-points:
	python -m y3p --model none --mode field-points data/spiroudome/config.yml

apidis-teams-detect:
	rm -rf data/apidis/data/sample*
	python -m y3p --model mrcnn --mode teams-detect data/apidis/config.yml

spiroudome-teams-detect:
	rm -rf data/spiroudome/teams
	python -m y3p --model mrcnn --mode teams-detect data/spiroudome/config.yml

apidis-teams-train:
	python -m y3p --model none --mode teams-train data/apidis/config.yml

spiroudome-teams-train:
	python -m y3p --model none --mode teams-train data/spiroudome/config.yml

spiroudome-teams: spiroudome-teams-detect, spiroudome-teams-train

spiroudome:
	python3 -m y3p --model mrcnn data/spiroudome/config.yml
