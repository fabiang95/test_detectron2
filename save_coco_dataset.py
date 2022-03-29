import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import requests
import time
from io import BytesIO
from PIL import Image

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# im_resp = requests.get("http://images.cocodataset.org/val2017/000000439715.jpg")
# im_array = np.frombuffer(im_resp.content, np.uint8)
# im = cv2.cvtColor(cv2.imdecode(im_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# plt.imshow(im)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)


# outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)

# We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.imshow(out.get_image()[:, :, ::-1])

from collections import deque

d = deque(maxlen=20)

cap = cv2.VideoCapture(0)
count = 0
last_time = -1

while True:
    ret, im = cap.read()
    print(ret)

    if not ret:
        continue

    outputs = predictor(im)

    data = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    print(data)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow('frame', out.get_image())
    # cv2.imshow('frame', im)

    curr_time = time.time()
    d.append(curr_time - last_time)
    last_time = curr_time

    count += 1

    if len(d) == 20 and count % 10 == 0:
        print(f"FPS: {1 / (sum(d) / len(d))}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
