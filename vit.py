import os
import timm
import warnings
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# detectron2
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.data.datasets import register_coco_instances
from detectron2.config import CfgNode as CN


class ViTBackbone(Backbone):
    def __init__(self, name, out_features, output_channels):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=True)
        self.output_channels = output_channels
        self.out_features = out_features
        self.input_size = 256

        # Ensure the convolution adapts from the correct number of input channels
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.adapt_conv = nn.Conv2d(768, output_channels, kernel_size=1, stride=1)

        # Necessary attributes for Detectron2 to understand the output features
        self._out_features = out_features
        self._out_feature_channels = {
            k: self.output_channels for k in self.out_features
        }
        self._out_feature_strides = {
            k: 16 for k in self.out_features
        }  # Example stride, adjust as needed
        set_trainable(self.vit, False)

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = self.vit.forward_features(x)
        # Assuming the output is [batch, num_tokens, feature_dim] and needs to be treated as [batch, feature_dim, H, W]
        x = x.permute(0, 2, 1)  # Change to [batch, feature_dim, num_tokens]

        # Apply global pooling to reduce token dimension
        x = self.global_pool(x).squeeze(-1)  # Output shape [batch, feature_dim]

        # Since we no longer have spatial dimensions, create a pseudo spatial dimension (1x1)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Now [batch, feature_dim, 1, 1]

        x = self.adapt_conv(x)  # Adapt the channel dimensions as necessary

        # Collect and return the features as expected by Detectron2
        features = {f: x for f in self.out_features}
        return features

    def output_shape(self):
        # This method should return a dict of ShapeSpec for each feature
        return {
            f: ShapeSpec(channels=self.output_channels, stride=16)
            for f in self.out_features
        }


@BACKBONE_REGISTRY.register()
def build_vit_backbone(cfg, input_shape):
    name = cfg.MODEL.VIT.NAME
    out_features = cfg.MODEL.VIT.OUT_FEATURES
    return ViTBackbone(name, out_features, 1024)


keypoint_names = [
    "crl_hips__C",
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
    "crl_toeEnd__L",
]

keypoint_flip_map = [
    ("crl_shoulder__L", "crl_shoulder__R"),
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
    ("crl_toeEnd__L", "crl_toeEnd__R"),
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
    0.197,
    0.212,
    0.298,
    0.155,
    0.206,
    0.364,
    0.103,
    0.103,
    0.114,
    0.166,
    0.435,
    0.101,
    0.129,
    0.085,
    0.131,
    0.067,
    0.077,
    0.077,
    0.135,
    0.080,
    0.451,
    0.127,
    0.137,
    0.084,
    0.089,
    0.058,
    0.108,
    0.124,
    0.153,
    0.112,
    0.112,
    0.156,
    0.211,
    0.462,
    0.108,
    0.072,
    0.075,
    1.00,
    0.062,
    0.058,
    0.139,
    0.794,
    0.082,
    0.056,
    0.055,
    0.059,
    0.053,
    0.147,
    0.051,
    0.066,
    0.051,
    0.073,
    0.063,
    0.048,
    0.050,
    0.210,
    0.483,
    0.548,
    0.580,
    0.893,
    0.205,
    0.546,
    0.340,
    0.175,
    0.278,
]


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators(
            [
                COCOEvaluator(
                    dataset_name,
                    output_dir=output_folder,
                    kpt_oks_sigmas=KEYPOINT_OKS_SIGMAS,
                )
            ]
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logger()

    def set_trainable(model, is_trainable):
        for param in model.parameters():
            param.requires_grad = is_trainable

    # DATA LOADING
    Data_Training = "train_xworld_kps"
    Data_Valid = "val_xworld_kps"

    register_coco_instances(
        Data_Training,
        {},
        "dataset/train_xworld.json",
        Path("dataset/train_xworld"),
    )
    register_coco_instances(
        Data_Valid,
        {},
        "dataset/val_xworld.json",
        Path("dataset/val_xworld"),
    )

    MetadataCatalog.get("train_xworld_kps").thing_classes = ["person"]
    MetadataCatalog.get("train_xworld_kps").thing_dataset_id_to_contiguous_id = {1: 0}
    MetadataCatalog.get("train_xworld_kps").keypoint_names = keypoint_names
    MetadataCatalog.get("train_xworld_kps").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("train_xworld_kps").keypoint_connection_rules = (
        keypoint_connection_rules
    )
    MetadataCatalog.get("train_xworld_kps").evaluator_type = "coco"

    MetadataCatalog.get("val_xworld_kps").thing_classes = ["person"]
    MetadataCatalog.get("val_xworld_kps").thing_dataset_id_to_contiguous_id = {1: 0}
    MetadataCatalog.get("val_xworld_kps").keypoint_names = keypoint_names
    MetadataCatalog.get("val_xworld_kps").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("train_xworld_kps").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("val_xworld_kps").evaluator_type = "coco"

    metadata = MetadataCatalog.get(Data_Training)

    cfg = get_cfg()
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.NAME = "hf_hub:timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
    cfg.MODEL.VIT.PRETRAINED = True
    cfg.MODEL.VIT.OUT_FEATURES = ["res4"]
    cfg.MODEL.RPN.IN_FEATURES = ["res4"]
    config_name = "config.yaml"
    cfg.merge_from_file(config_name)

    cfg.DATASETS.TRAIN = (Data_Training,)
    cfg.DATASETS.TEST = (Data_Valid,)

    # cfg.MODEL.WEIGHTS ="detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    cfg.MODEL.DEVICE = "cuda:0"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 65
    cfg.TEST.KEYPOINT_OKS_SIGMAS = KEYPOINT_OKS_SIGMAS


    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.WARMUP_ITERS = 10
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (500, 1000)
    cfg.SOLVER.CHECKPOINT_PERIOD=1000

    cfg.TEST.EVAL_PERIOD = 1000

    cfg.OUTPUT_DIR = "./keypoint-vit"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
