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
from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

YOLOV6_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "superb-ai/yolov6n": "https://huggingface.co/superb-ai/yolov6n/blob/main/config.json",
    "superb-ai/yolov6s": "https://huggingface.co/superb-ai/yolov6s/blob/main/config.json",
    "superb-ai/yolov6l6": "https://huggingface.co/superb-ai/yolov6l6/blob/main/config.json",
    # See all YOLOV6 models at https://huggingface.co/models?filter=yolov6
}


class Yolov6Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YolosModel`]. It is used to instantiate a YOLOV6
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the YOLOV6
    [hustvl/yolos-base](https://huggingface.co/hustvl/yolos-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        in_channels (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        block_type (`<fill_type>`, *optional*, defaults to `"Yolov6RepVGGBlock"`): <fill_docstring>
        backbone_num_repeats (`<fill_type>`, *optional*, defaults to `[1, 2, 4, 6, 2]`): <fill_docstring>
        backbone_out_channels (`<fill_type>`, *optional*, defaults to `[16, 32, 64, 128, 256]`): <fill_docstring>
        backbone_csp_e (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
        backbone_fuse_P2 (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        backbone_cspsppf (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        backbone_stage_block_type (`<fill_type>`, *optional*, defaults to `"Yolov6RepBlock"`): <fill_docstring>
        neck_num_repeats (`<fill_type>`, *optional*, defaults to `[4, 4, 4, 4]`): <fill_docstring>
        neck_out_channels (`<fill_type>`, *optional*, defaults to `[64, 32, 32, 64, 64, 128]`): <fill_docstring>
        neck_csp_e (`<fill_type>`, *optional*, defaults to 0.0): <fill_docstring>
        neck_stage_block_type (`<fill_type>`, *optional*, defaults to `"Yolov6RepBlock"`): <fill_docstring>
        head_in_channels (`<fill_type>`, *optional*, defaults to `[128, 256, 512]`): <fill_docstring>
        head_num_layers (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        head_anchors (`<fill_type>`, *optional*, defaults to 3): <fill_docstring>
        head_strides (`<fill_type>`, *optional*, defaults to `[8, 16, 32]`): <fill_docstring>
        atss_warmup_epoch (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        iou_type (`<fill_type>`, *optional*, defaults to `"siou"`): <fill_docstring>
        use_dfl (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        reg_max (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        reg_max_proj (`<fill_type>`, *optional*, defaults to 16): <fill_docstring>
        class_loss_coefficient (`<fill_type>`, *optional*, defaults to 1.0): <fill_docstring>
        iou_loss_coefficient (`<fill_type>`, *optional*, defaults to 2.5): <fill_docstring>
        dfl_loss_coefficient (`<fill_type>`, *optional*, defaults to 0.5): <fill_docstring>
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        forward_fuse (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        export (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>

    Example:

    ```python
    >>> from transformers import Yolov6Config, Yolov6Model

    >>> # Initializing a YOLOV6 hustvl/yolos-base style configuration
    >>> configuration = Yolov6Config()

    >>> # Initializing a model (with random weights) from the hustvl/yolos-base style configuration
    >>> model = Yolov6Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "yolov6n"

    def __init__(
        self,
        in_channels=3,
        block_type="Yolov6RepVGGBlock",
        backbone_num_repeats=[1, 2, 4, 6, 2],
        backbone_out_channels=[16, 32, 64, 128, 256],
        backbone_csp_e=float(0),
        backbone_fuse_P2=True,
        backbone_cspsppf=True,
        backbone_stage_block_type="Yolov6RepBlock",
        neck_num_repeats=[4, 4, 4, 4],
        neck_out_channels=[64, 32, 32, 64, 64, 128],
        neck_csp_e=float(0),
        neck_stage_block_type="Yolov6RepBlock",
        head_in_channels=[128, 256, 512],
        head_num_layers=3,
        head_anchors=3,
        head_strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type="siou",
        use_dfl=False,
        reg_max=0,  # if use_dfl is False, please set reg_max to 0
        reg_max_proj=16,
        class_loss_coefficient=1.0,
        iou_loss_coefficient=2.5,
        dfl_loss_coefficient=0.5,
        initializer_range=0.02,
        forward_fuse=False,
        export=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.block_type = block_type
        self.backbone_num_repeats = backbone_num_repeats
        self.backbone_out_channels = backbone_out_channels
        self.backbone_csp_e = backbone_csp_e
        self.backbone_fuse_P2 = backbone_fuse_P2
        self.backbone_cspsppf = backbone_cspsppf
        self.backbone_stage_block_type = backbone_stage_block_type
        self.neck_num_repeats = neck_num_repeats
        self.neck_out_channels = neck_out_channels
        self.neck_csp_e = neck_csp_e
        self.neck_stage_block_type = neck_stage_block_type
        self.head_in_channels = head_in_channels
        self.head_num_layers = head_num_layers
        self.head_anchors = head_anchors
        self.head_strides = head_strides
        self.atss_warmup_epoch = atss_warmup_epoch
        self.iou_type = iou_type
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.reg_max_proj = reg_max_proj
        self.class_loss_coefficient = class_loss_coefficient
        self.iou_loss_coefficient = iou_loss_coefficient
        self.dfl_loss_coefficient = dfl_loss_coefficient
        self.initializer_range = initializer_range
        self.forward_fuse = forward_fuse
        self.export = export


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
