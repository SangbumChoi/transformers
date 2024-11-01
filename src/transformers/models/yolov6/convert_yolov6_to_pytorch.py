# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert YOLOS checkpoints from the original repository. URL: https://github.com/hustvl/YOLOS"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import Yolov6Config, Yolov6ForObjectDetection, Yolov6ImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_yolov6_config(yolov6_name: str) -> Yolov6Config:
    config = Yolov6Config()
    config.model_type = yolov6_name

    # size of the architecture
    if yolov6_name == "yolov6n":
        config.image_size = 640
    elif yolov6_name == "yolov6s":
        config.image_size = 640
        config.backbone_out_channels = [32, 64, 128, 256, 512]
        config.neck_out_channels = [128, 64, 64, 128, 128, 256]
        config.iou_type = "giou"
        config.class_loss_coefficient = 1.0
        config.dfl_loss_coefficient = 1.0
    elif yolov6_name == "yolov6m":
        config.image_size = 640
        config.backbone_num_repeats = [1, 4, 7, 11, 4]
        config.backbone_out_channels = [48, 96, 192, 384, 768]
        config.backbone_csp_e = float(2) / 3
        config.backbone_cspsppf = False
        config.backbone_stage_block_type = "Yolov6BepC3"
        config.neck_num_repeats = [7, 7, 7, 7]
        config.neck_out_channels = [192, 96, 96, 192, 192, 384]
        config.neck_csp_e = float(2) / 3
        config.neck_stage_block_type = "Yolov6BepC3"
        config.iou_type = "giou"
        config.use_dfl = True
        config.reg_max = 16
    elif yolov6_name == "yolov6l":
        config.image_size = 640
        config.block_type = "Yolov6ConvBNSilu"
        config.backbone_num_repeats = [1, 6, 12, 18, 6]
        config.backbone_out_channels = [64, 128, 256, 512, 1024]
        config.backbone_csp_e = float(1) / 2
        config.backbone_cspsppf = False
        config.backbone_stage_block_type = "Yolov6BepC3"
        config.neck_num_repeats = [12, 12, 12, 12]
        config.neck_out_channels = [256, 128, 128, 256, 256, 512]
        config.neck_csp_e = float(1) / 2
        config.neck_stage_block_type = "Yolov6BepC3"
        config.iou_type = "giou"
        config.use_dfl = True
        config.reg_max = 16
    elif yolov6_name == "yolov6l6":
        config.image_size = 1280
        config.block_type = "Yolov6ConvBNSilu"
        config.backbone_num_repeats = [1, 6, 12, 18, 6, 6]
        config.backbone_out_channels = [64, 128, 256, 512, 768, 1024]
        config.backbone_csp_e = float(1) / 2
        config.backbone_cspsppf = False
        config.backbone_stage_block_type = "Yolov6BepC3"
        config.neck_num_repeats = [12, 12, 12, 12, 12, 12]
        config.neck_out_channels = [512, 256, 128, 256, 512, 1024]
        config.neck_csp_e = float(1) / 2
        config.neck_stage_block_type = "Yolov6BepC3"
        config.head_in_channels = [128, 256, 512, 1024]
        config.head_num_layers = 4
        config.head_anchors = 1
        config.head_strides = [8, 16, 32, 64]
        config.iou_type = "giou"
        config.atss_warmup_epoch = 4
        config.use_dfl = True
        config.reg_max = 16

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def rename_key(name: str) -> str:
    if "conv." in name:
        name = name.replace("conv.", "convolution.")
    if "bn." in name:
        name = name.replace("bn.", "normalization.")
    if "block.convolution" in name:
        name = name.replace("block.convolution", "convolution")
    if "block.normalization" in name:
        name = name.replace("block.normalization", "normalization")
    if "backbone.stem" in name:
        name = name.replace("backbone.stem", "model.embedder.embedder")
    if "backbone.ERBlock" in name:
        name = name.replace("backbone.ERBlock", "model.encoder.blocks.ERBlock")
    if "2.cspsppf" in name:
        name = name.replace("2.cspsppf", "channel_merge_layer")
    if "2.sppf" in name:
        name = name.replace("2.sppf", "channel_merge_layer")
    if "neck" in name:
        name = name.replace("neck", "model.neck.stage")
    if "detect" in name:
        name = name.replace("detect", "head")
    if "proj_convolution" in name:
        name = name.replace("proj_convolution", "proj_conv")
    if "reg_preds_dist" in name:
        name = name.replace("reg_preds_dist", "reg_preds")

    return name


def convert_state_dict(orig_state_dict: dict, model: Yolov6ForObjectDetection) -> dict:
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        orig_state_dict[rename_key(key)] = val

    return orig_state_dict


# We will verify our results on an image of cute cats
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_yolov6_checkpoint(
    yolov6_name: str,
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    push_to_hub: bool = False,
):
    """
    Copy/paste/tweak model's weights to our YOLOV6 structure.
    """
    config = get_yolov6_config(yolov6_name)

    # load original state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # load 🤗 model
    model = Yolov6ForObjectDetection(config)
    model.eval()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # Check outputs on an image, prepared by YolosImageProcessor
    size = config.image_size
    image_processor = Yolov6ImageProcessor(format="coco_detection", size={"shortest_edge": size, "longest_edge": size})
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits, pred_boxes = outputs.logits, outputs.pred_boxes
    original_size = encoding["pixel_values"].shape
    pred_boxes *= torch.tensor([original_size[-1], original_size[-2], original_size[-1], original_size[-2]])

    expected_slice_logits, expected_slice_boxes = None, None
    if yolov6_name == "yolov6n":
        expected_slice_logits = torch.tensor(
            [
                [-3.79127, -5.42277, -4.43975],
                [-4.40880, -5.64879, -4.93791],
                [-4.51388, -5.69715, -4.93552],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [11.27861, 22.33572, 22.95666],
                [23.49722, 20.68348, 47.31107],
                [32.20158, 19.24104, 65.46049],
            ]
        )
    elif yolov6_name == "yolov6s":
        expected_slice_logits = torch.tensor(
            [
                [-4.46909, -5.56590, -5.12657],
                [-4.49251, -5.80001, -5.15828],
                [-4.52727, -5.81888, -5.12548],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [11.56186, 13.51999, 22.47980],
                [14.94720, 11.38480, 30.77328],
                [22.81690, 11.56549, 44.71961],
            ]
        )
    elif yolov6_name == "yolov6m":
        expected_slice_logits = torch.tensor(
            [[-4.4237, -5.7332, -5.0298], [-4.5706, -5.8418, -5.1312], [-4.6787, -5.8133, -5.2177]]
        )
        expected_slice_boxes = torch.tensor(
            [[15.73717, 25.04552, 31.35688], [19.21497, 18.68022, 39.56915], [23.27861, 14.67001, 47.11521]]
        )
    elif yolov6_name == "yolov6l":
        expected_slice_logits = torch.tensor(
            [[-4.0884, -5.3208, -4.6791], [-4.1873, -5.3827, -4.7223], [-4.5806, -5.5509, -4.9667]]
        )
        expected_slice_boxes = torch.tensor(
            [[11.7442, 28.9719, 25.2751], [22.2533, 28.0677, 52.8962], [33.4531, 19.2840, 68.9814]]
        )
    elif yolov6_name == "yolov6l6":
        expected_slice_logits = torch.tensor(
            [[-4.76734, -5.98765, -5.42296], [-5.04849, -6.27099, -5.64376], [-5.28514, -6.38343, -5.75267]]
        )
        expected_slice_boxes = torch.tensor(
            [[19.42040, 26.32382, 40.39944], [25.69743, 24.37918, 54.64775], [33.79816, 14.53956, 69.25398]]
        )
    else:
        logger.warning(f"Skipping logit tests: {yolov6_name}")
        expected_slice_logits = None
        expected_slice_boxes = None

    if expected_slice_logits is not None:
        if not torch.allclose(logits[0, :3, :3], expected_slice_logits, atol=1e-3):
            logger.warning("logit value mismatch")
    if expected_slice_boxes is not None:
        if not torch.allclose(pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-1):
            logger.warning("bbox value mismatch")

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {yolov6_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_mapping = {
            "yolov6n": "yolov6n",
            "yolov6s": "yolov6s",
            "yolov6m": "yolov6m",
            "yolov6l": "yolov6l",
            "yolov6l6": "yolov6l6",
        }

        print("Pushing to the hub...")
        model_name = model_mapping[yolov6_name]
        image_processor.push_to_hub(model_name, organization="superb-ai")
        model.push_to_hub(model_name, organization="superb-ai")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--yolov6_name",
        default="yolov6n",
        type=str,
        help=(
            "Name of the YOLOV6 model you'd like to convert. Should be one of 'yolov6n', 'yolov6s',"
            " 'yolov6m', 'yolov6l', 'yolov6l6'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default="",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_yolov6_checkpoint(
        args.yolov6_name,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
