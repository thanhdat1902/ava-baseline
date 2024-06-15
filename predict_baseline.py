import os
import cv2
import json
import random
import numpy as np
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

# detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.engine import DefaultPredictor

# Ignore warnings
import warnings
warnings.filterwarnings('ignore') 

# Setup logger
setup_logger()

Data_Resister_test="test_xworld_kps";
from detectron2.data.datasets import register_coco_instances

register_coco_instances(Data_Resister_test,{},'dataset/test_xworld.json', Path("dataset/test_xworld"))

dataset_test = DatasetCatalog.get(Data_Resister_test)

keypoint_names = ["crl_hips__C",
    "crl_spine__C",
    "crl_spine01__C",
    "crl_shoulder__L",
    "crl_arm__L",
    "crl_foreArm__L",
    "crl_hand__L",
    "crl_handThumb__L",
    "crl_handThumb01__L",
    "crl_handThumb02__L",
    "crl_handThumbEnd__L",
    "crl_handIndex__L",
    "crl_handIndex01__L",
    "crl_handIndex02__L",
    "crl_handIndexEnd__L",
    "crl_handMiddle__L",
    "crl_handMiddle01__L",
    "crl_handMiddle02__L",
    "crl_handMiddleEnd__L",
    "crl_handRing__L",
    "crl_handRing01__L",
    "crl_handRing02__L",
    "crl_handRingEnd__L",
    "crl_handPinky__L",
    "crl_handPinky01__L",
    "crl_handPinky02__L",
    "crl_handPinkyEnd__L",
    "crl_neck__C",
    "crl_Head__C",
    "crl_eye__L",
    "crl_eye__R",
    "crl_shoulder__R",
    "crl_arm__R",
    "crl_foreArm__R",
    "crl_hand__R",
    "crl_handThumb__R",
    "crl_handThumb01__R",
    "crl_handThumb02__R",
    "crl_handThumbEnd__R",
    "crl_handIndex__R",
    "crl_handIndex01__R",
    "crl_handIndex02__R",
    "crl_handIndexEnd__R",
    "crl_handMiddle__R",
    "crl_handMiddle01__R",
    "crl_handMiddle02__R",
    "crl_handMiddleEnd__R",
    "crl_handRing__R",
    "crl_handRing01__R",
    "crl_handRing02__R",
    "crl_handRingEnd__R",
    "crl_handPinky__R",
    "crl_handPinky01__R",
    "crl_handPinky02__R",
    "crl_handPinkyEnd__R",
    "crl_thigh__R",
    "crl_leg__R",
    "crl_foot__R",
    "crl_toe__R",
    "crl_toeEnd__R",
    "crl_thigh__L",
    "crl_leg__L",
    "crl_foot__L",
    "crl_toe__L",
    "crl_toeEnd__L"
                 ]

keypoint_flip_map = [("crl_shoulder__L", "crl_shoulder__R"),
    ("crl_arm__L", "crl_arm__R"),
    ("crl_foreArm__L", "crl_foreArm__R"),
    ("crl_hand__L", "crl_hand__R"),
    ("crl_handThumb__L", "crl_handThumb__R"),
    ("crl_handThumb01__L", "crl_handThumb01__R"),
    ("crl_handThumb02__L", "crl_handThumb02__R"),
    ("crl_handThumbEnd__L", "crl_handThumbEnd__R"),
    ("crl_handIndex__L", "crl_handIndex__R"),
    ("crl_handIndex01__L", "crl_handIndex01__R"),
    ("crl_handIndex02__L", "crl_handIndex02__R"),
    ("crl_handIndexEnd__L", "crl_handIndexEnd__R"),
    ("crl_handMiddle__L", "crl_handMiddle__R"),
    ("crl_handMiddle01__L", "crl_handMiddle01__R"),
    ("crl_handMiddle02__L", "crl_handMiddle02__R"),
    ("crl_handMiddleEnd__L", "crl_handMiddleEnd__R"),
    ("crl_handRing__L", "crl_handRing__R"),
    ("crl_handRing01__L", "crl_handRing01__R"),
    ("crl_handRing02__L", "crl_handRing02__R"),
    ("crl_handRingEnd__L", "crl_handRingEnd__R"),
    ("crl_handPinky__L", "crl_handPinky__R"),
    ("crl_handPinky01__L", "crl_handPinky01__R"),
    ("crl_handPinky02__L", "crl_handPinky02__R"),
    ("crl_handPinkyEnd__L", "crl_handPinkyEnd__R"),
    ("crl_eye__L", "crl_eye__R"),
    ("crl_thigh__L", "crl_thigh__R"),
    ("crl_leg__L", "crl_leg__R"),
    ("crl_foot__L", "crl_foot__R"),
    ("crl_toe__L", "crl_toe__R"),
    ("crl_toeEnd__L", "crl_toeEnd__R")
                    ]

keypoint_connection_rules = [
    ("crl_eye__L", "crl_Head__C", (0, 255, 128)),
    ("crl_eye__R", "crl_Head__C", (0, 255, 128)),
    ("crl_Head__C", "crl_neck__C", (0, 255, 128)),
    ("crl_neck__C", "crl_spine01__C", (0, 255, 128)),
    ("crl_spine01__C", "crl_spine__C", (0, 255, 128)),
    ("crl_spine__C", "crl_hips__C", (0, 255, 128)),

    ("crl_hips__C", "crl_thigh__R", (0, 128, 255)),
    ("crl_hips__C", "crl_thigh__L", (255, 128, 0)),
    ("crl_thigh__R", "crl_leg__R", (0, 128, 255)),
    ("crl_thigh__L", "crl_leg__L", (255, 128, 0)),
    ("crl_leg__R", "crl_foot__R", (0, 128, 255)),
    ("crl_leg__L", "crl_foot__L", (255, 128, 0)),
    ("crl_foot__R", "crl_toe__R", (0, 128, 255)),
    ("crl_foot__L", "crl_toe__L", (255, 128, 0)),
    ("crl_toe__R", "crl_toeEnd__R", (0, 128, 255)),
    ("crl_toe__L", "crl_toeEnd__L", (255, 128, 0)),


    ("crl_spine01__C", "crl_shoulder__R", (0, 128, 255)),
    ("crl_shoulder__R", "crl_arm__R", (0, 128, 255)),
    ("crl_arm__R", "crl_foreArm__R", (0, 128, 255)),
    ("crl_foreArm__R", "crl_hand__R", (0, 128, 255)),

    ("crl_hand__R", "crl_handThumb__R", (0, 128, 255)),
    ("crl_handThumb__R", "crl_handThumb01__R", (0, 128, 255)),
    ("crl_handThumb01__R", "crl_handThumb02__R", (0, 128, 255)),
    ("crl_handThumb02__R", "crl_handThumbEnd__R", (0, 128, 255)),

    ("crl_hand__R", "crl_handIndex__R", (0, 128, 255)),
    ("crl_handIndex__R", "crl_handIndex01__R", (0, 128, 255)),
    ("crl_handIndex01__R", "crl_handIndex02__R", (0, 128, 255)),
    ("crl_handIndex02__R", "crl_handIndexEnd__R", (0, 128, 255)),

    ("crl_hand__R", "crl_handMiddle__R", (0, 128, 255)),
    ("crl_handMiddle__R", "crl_handMiddle01__R", (0, 128, 255)),
    ("crl_handMiddle01__R", "crl_handMiddle02__R", (0, 128, 255)),
    ("crl_handMiddle02__R", "crl_handMiddleEnd__R", (0, 128, 255)),

    ("crl_hand__R", "crl_handRing__R", (0, 128, 255)),
    ("crl_handRing__R", "crl_handRing01__R", (0, 128, 255)),
    ("crl_handRing01__R", "crl_handRing02__R", (0, 128, 255)),
    ("crl_handRing02__R", "crl_handRingEnd__R", (0, 128, 255)),

    ("crl_hand__R", "crl_handPinky__R", (0, 128, 255)),
    ("crl_handPinky__R", "crl_handPinky01__R", (0, 128, 255)),
    ("crl_handPinky01__R", "crl_handPinky02__R", (0, 128, 255)),
    ("crl_handPinky02__R", "crl_handPinkyEnd__R", (0, 128, 255)),


    ("crl_spine01__C", "crl_shoulder__L", (255, 128, 0)),
    ("crl_shoulder__L", "crl_arm__L", (255, 128, 0)),
    ("crl_arm__L", "crl_foreArm__L", (255, 128, 0)),
    ("crl_foreArm__L", "crl_hand__L", (255, 128, 0)),
    
    ("crl_hand__L", "crl_handThumb__L", (255, 128, 0)),
    ("crl_handThumb__L", "crl_handThumb01__L", (255, 128, 0)),
    ("crl_handThumb01__L", "crl_handThumb02__L", (255, 128, 0)),
    ("crl_handThumb02__L", "crl_handThumbEnd__L", (255, 128, 0)),

    ("crl_hand__L", "crl_handIndex__L", (255, 128, 0)),
    ("crl_handIndex__L", "crl_handIndex01__L", (255, 128, 0)),
    ("crl_handIndex01__L", "crl_handIndex02__L", (255, 128, 0)),
    ("crl_handIndex02__L", "crl_handIndexEnd__L", (255, 128, 0)),

    ("crl_hand__L", "crl_handMiddle__L", (255, 128, 0)),
    ("crl_handMiddle__L", "crl_handMiddle01__L", (255, 128, 0)),
    ("crl_handMiddle01__L", "crl_handMiddle02__L", (255, 128, 0)),
    ("crl_handMiddle02__L", "crl_handMiddleEnd__L", (255, 128, 0)),

    ("crl_hand__L", "crl_handRing__L", (255, 128, 0)),
    ("crl_handRing__L", "crl_handRing01__L", (255, 128, 0)),
    ("crl_handRing01__L", "crl_handRing02__L", (255, 128, 0)),
    ("crl_handRing02__L", "crl_handRingEnd__L", (255, 128, 0)),

    ("crl_hand__L", "crl_handPinky__L", (255, 128, 0)),
    ("crl_handPinky__L", "crl_handPinky01__L", (255, 128, 0)),
    ("crl_handPinky01__L", "crl_handPinky02__L", (255, 128, 0)),
    ("crl_handPinky02__L", "crl_handPinkyEnd__L", (255, 128, 0)),
]

KEYPOINT_OKS_SIGMAS = [
    0.197, 0.212, 0.298, 0.155, 0.206, 0.364, 0.103, 0.103, 0.114, 0.166, 0.435, 0.101, 0.129,
    0.085, 0.131, 0.067, 0.077, 0.077, 0.135, 0.080, 0.451, 0.127, 0.137, 0.084, 0.089, 0.058, 
    0.108, 0.124, 0.153, 0.112, 0.112, 0.156, 0.211, 0.462, 0.108, 0.072, 0.075, 1.00, 0.062, 
    0.058, 0.139, 0.794, 0.082, 0.056, 0.055, 0.059, 0.053, 0.147, 0.051, 0.066, 0.051, 0.073,
    0.063, 0.048, 0.050, 0.210, 0.483, 0.548, 0.580, 0.893, 0.205, 0.546, 0.340, 0.175, 0.278
    ]


cfg = get_cfg()
config_name = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml" 
cfg.merge_from_file(model_zoo.get_config_file(config_name))


# cfg.MODEL.WEIGHTS ="detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
cfg.MODEL.WEIGHTS ="./keypoint/model_final.pth"
# cfg.MODEL.WEIGHTS ="./YOLOV8/best.pt"

cfg.MODEL.DEVICE = "cuda:0"

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 65
cfg.TEST.KEYPOINT_OKS_SIGMAS = KEYPOINT_OKS_SIGMAS

cfg.DATALOADER.NUM_WORKERS = 8

cfg.SOLVER.IMS_PER_BATCH = 8 
cfg.SOLVER.BASE_LR = 0.01 
cfg.SOLVER.WARMUP_ITERS = 0
cfg.SOLVER.MAX_ITER = 5
cfg.SOLVER.STEPS = (500, 1000) 
cfg.SOLVER.CHECKPOINT_PERIOD=1

cfg.TEST.EVAL_PERIOD = 1

cfg.OUTPUT_DIR = "./keypoint-test"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("test_xworld_kps")
predictions = []
for d in dataset_dicts:
    # Load image
    img_path = d["file_name"]
    img = cv2.imread(img_path)
    
    # Make predictions
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    # Loop through each instance
    for i in range(len(instances)):
        instance = instances[i]

        keypoints = instance.pred_keypoints.numpy().flatten().tolist()
        score = instance.scores.item()
        bbox = instance.pred_boxes.tensor.numpy().flatten().tolist()
        # Prepare the prediction dictionary
        prediction = {
            "image_id": d["image_id"],
            "bbox": bbox,
            "category_id": 1,  # assuming 1 for person in keypoint detection task
            "keypoints": keypoints,
            "score": score
        }
        # Add the prediction to the list
        predictions.append(prediction)
    
    print(len(predictions))

with open("predictions-baseline.json", "w") as f:
    json.dump(predictions, f)
