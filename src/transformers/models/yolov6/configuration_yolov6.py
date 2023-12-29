# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" YOLOV6 model configuration"""
import math
from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/yolos-small": "https://huggingface.co/hustvl/yolos-small/resolve/main/config.json",
    # See all YOLOS models at https://huggingface.co/models?filter=yolos
}


class YolosConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YolosModel`]. It is used to instantiate a YOLOS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the YOLOS
    [hustvl/yolos-base](https://huggingface.co/hustvl/yolos-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`List[int]`, *optional*, defaults to `[512, 864]`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        num_detection_tokens (`int`, *optional*, defaults to 100):
            The number of detection tokens.
        use_mid_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to use the mid-layer position encodings.
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        class_cost (`float`, *optional*, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.

    Example:

    ```python
    >>> from transformers import Yolov6Config, Yolov6Model

    >>> # Initializing a YOLOV6 hustvl/yolos-base style configuration
    >>> configuration = YolosConfig()

    >>> # Initializing a model (with random weights) from the hustvl/yolos-base style configuration
    >>> model = YolosModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "yolov6n"

    def __init__(
        self,
        depth_multiple=1.0,
        width_multiple=1.0,
        backbone_type="CSPBepBackbone_P6",
        backbone_num_repeats=[1, 6, 12, 18, 6, 6],
        backbone_out_channels=[64, 128, 256, 512, 768, 1024],
        backbone_csp_e=float(1) / 2,
        backbone_fuse_P2=True,
        neck_type="CSPRepBiFPANNeck_P6",
        neck_num_repeats=[12, 12, 12, 12, 12, 12],
        neck_out_channels=[512, 256, 128, 256, 512, 1024],
        neck_csp_e=float(1) / 2,
        head_type="EffiDeHead",
        head_in_channels=[128, 256, 512, 1024],
        head_num_layers=4,
        head_anchors=1,
        head_strides=[8, 16, 32, 64],
        head_atss_warmup_epoch=4,
        iou_type="giou",
        use_dfl=True,
        reg_max=16,  # if use_dfl is False, please set reg_max to 0
        class_loss_coefficient=1.0,
        iou_loss_coefficient=1.0,
        dfl_loss_coefficient=0.5,
        auxiliary_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        backbone_num_repeats = [
            (max(round(i * depth_multiple), 1) if i > 1 else i)
            for i in backbone_num_repeats
        ]
        neck_num_repeats = [
            (max(round(i * depth_multiple), 1) if i > 1 else i)
            for i in neck_num_repeats
        ]

        backbone_out_channels = [
            math.ceil(i * width_multiple / 8) * 8 for i in backbone_out_channels
        ]
        neck_out_channels = [
            math.ceil(i * width_multiple / 8) * 8 for i in neck_out_channels
        ]

        self.backbone_type = backbone_type
        self.backbone_num_repeats = backbone_num_repeats
        self.backbone_out_channels = backbone_out_channels
        self.backbone_csp_e = backbone_csp_e
        self.backbone_fuse_P2 = backbone_fuse_P2
        self.neck_type = neck_type
        self.neck_num_repeats = neck_num_repeats
        self.neck_out_channels = neck_out_channels
        self.neck_csp_e = neck_csp_e
        self.head_type = head_type
        self.head_in_channels = head_in_channels
        self.head_num_layers = head_num_layers
        self.head_anchors = head_anchors
        self.head_strides = head_strides
        self.head_atss_warmup_epoch = head_atss_warmup_epoch
        self.iou_type = iou_type
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.auxiliary_loss = auxiliary_loss

        # Loss coefficients
        self.class_loss_coefficient = class_loss_coefficient
        self.iou_loss_coefficient = iou_loss_coefficient
        self.dfl_loss_coefficient = dfl_loss_coefficient


class Yolov6OnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                (
                    "pixel_values",
                    {0: "batch", 1: "num_channels", 2: "height", 3: "width"},
                ),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    @property
    def default_onnx_opset(self) -> int:
        return 12
