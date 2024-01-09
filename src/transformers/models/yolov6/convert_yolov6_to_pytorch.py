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

    # size of the architecture
    if yolov6_name == "yolov6n":
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
        config.image_size = [800, 1333]
        config.use_mid_position_embeddings = False
    elif yolov6_name == "yolov6s":
        config.hidden_size = 330
        config.num_hidden_layers = 14
        config.num_attention_heads = 6
        config.intermediate_size = 1320
    elif yolov6_name == "yolov6m":
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif yolov6_name == "yolov6s=l":
        config.image_size = [800, 1344]

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
    yolov6_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our YOLOS structure.
    """
    config = get_yolov6_config(yolov6_name)

    # load original state_dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # load 🤗 model
    model = Yolov6ForObjectDetection(config)
    model.eval()
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    # Check outputs on an image, prepared by YolosImageProcessor
    size = 640 if yolov6_name in ("yolov6n", "yolov6s") else 1280
    image_processor = Yolov6ImageProcessor(format="coco_detection", size={'shortest_edge': size, 'longest_edge': size})
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)
    logits, pred_boxes = outputs.logits, outputs.pred_boxes
    print(logits.shape, pred_boxes.shape)

    expected_slice_logits, expected_slice_boxes = None, None
    if yolov6_name == "yolov6n":
        expected_slice_logits = torch.tensor(
            [[-39.5022, -11.9820, -17.6888], [-29.9574, -9.9769, -17.7691], [-42.3281, -20.7200, -30.6294]]
        )
        expected_slice_boxes = torch.tensor(
            [[11.27861, 22.33572, 22.95666],
            [23.49722, 20.68348, 47.31107],
            [32.20158, 19.24104, 65.46049]]
        )
    # elif yolov6_name == "yolos_s_200_pre":
    #     expected_slice_logits = torch.tensor(
    #         [[-24.0248, -10.3024, -14.8290], [-42.0392, -16.8200, -27.4334], [-27.2743, -11.8154, -18.7148]]
    #     )
    #     expected_slice_boxes = torch.tensor(
    #         [[0.2559, 0.5455, 0.4706], [0.2989, 0.7279, 0.1875], [0.7732, 0.4017, 0.4462]]
    #     )
    # elif yolov6_name == "yolos_s_300_pre":
    #     expected_slice_logits = torch.tensor(
    #         [[-36.2220, -14.4385, -23.5457], [-35.6970, -14.7583, -21.3935], [-31.5939, -13.6042, -16.8049]]
    #     )
    #     expected_slice_boxes = torch.tensor(
    #         [[0.7614, 0.2316, 0.4728], [0.7168, 0.4495, 0.3855], [0.4996, 0.1466, 0.9996]]
    #     )
    # elif yolov6_name == "yolos_s_dWr":
    #     expected_slice_logits = torch.tensor(
    #         [[-42.8668, -24.1049, -41.1690], [-34.7456, -14.1274, -24.9194], [-33.7898, -12.1946, -25.6495]]
    #     )
    #     expected_slice_boxes = torch.tensor(
    #         [[0.5587, 0.2773, 0.0605], [0.5004, 0.3014, 0.9994], [0.4999, 0.1548, 0.9994]]
    #     )
    # elif yolov6_name == "yolos_base":
    #     expected_slice_logits = torch.tensor(
    #         [[-40.6064, -24.3084, -32.6447], [-55.1990, -30.7719, -35.5877], [-51.4311, -33.3507, -35.6462]]
    #     )
    #     expected_slice_boxes = torch.tensor(
    #         [[0.5555, 0.2794, 0.0655], [0.9049, 0.2664, 0.1894], [0.9183, 0.1984, 0.1635]]
    #     )
    else:
        raise ValueError(f"Unknown yolov6_name: {yolov6_name}")

    assert torch.allclose(logits[0, :3, :3], expected_slice_logits, atol=1e-4)
    assert torch.allclose(pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4)

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
            " 'yolov6m', 'yolov6l'."
        ),
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original state dict (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_yolov6_checkpoint(args.yolov6_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
