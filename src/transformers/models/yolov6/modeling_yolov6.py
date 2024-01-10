# coding=utf-8
# Copyright 2022 School of EIC, Huazhong University of Science & Technology and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch YOLOV6 model."""


import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_yolov6 import Yolov6Config


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Yolov6Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "hustvl/yolos-small"
_EXPECTED_OUTPUT_SHAPE = [1, 3401, 384]


YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hustvl/yolos-small",
    # See all YOLOS models at https://huggingface.co/models?filter=yolos
]


@dataclass
class Yolov6ModelOutput(BackboneOutput):
    loss: Optional[torch.FloatTensor] = None


@dataclass
class Yolov6ObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Yolov6ForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~YolosImageProcessor.post_process`] to retrieve the unnormalized bounding
            boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def dist2bbox(distance, anchor_points, box_format="xyxy"):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == "xyxy":
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == "xywh":
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def generate_anchors(
    feats,
    fpn_strides,
    grid_cell_size=5.0,
    grid_cell_offset=0.5,
    device="cpu",
    is_eval=False,
    mode="af",
):
    """Generate anchors from features."""
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).to(torch.float)
            if mode == "af":  # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == "ab":  # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device).repeat(3, 1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor = (
                torch.stack(
                    [
                        shift_x - cell_half_size,
                        shift_y - cell_half_size,
                        shift_x + cell_half_size,
                        shift_y + cell_half_size,
                    ],
                    axis=-1,
                )
                .clone()
                .to(feats[0].dtype)
            )
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            if mode == "af":  # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == "ab":  # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3, 1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor


class Yolov6ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation_type: str,
        padding=None,
        groups=1,
        bias=False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.normalization = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activation_type = activation_type
        self.activation = ACT2FN[activation_type] if activation_type is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        if self.activation_type is not None:
            hidden_state = self.activation(hidden_state)
        return hidden_state

    def forward_fuse(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        if self.activation_type is not None:
            hidden_state = self.activation(hidden_state)
        return hidden_state


class Yolov6RepVGGBlock(nn.Module):
    """Yolov6RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
        use_se=False,
    ):
        super(Yolov6RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels, eps=0.001, momentum=0.03) if out_channels == in_channels and stride == 1 else None
            )
            self.rbr_dense = Yolov6ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation_type=None,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = Yolov6ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                activation_type=None,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs):
        """Forward process"""
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size**2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Yolov6ConvLayer):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class Yolov6BottleRep(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block=Yolov6RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class Yolov6RepBlock(nn.Module):
    """
    Yolov6RepBlock is a stage block with rep-style basic block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        block=Yolov6RepVGGBlock,
        basic_block=Yolov6RepVGGBlock,
        **kwargs,
    ):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == Yolov6BottleRep:
            self.conv1 = Yolov6BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = (
                nn.Sequential(
                    *(
                        Yolov6BottleRep(
                            out_channels,
                            out_channels,
                            basic_block=basic_block,
                            weight=True,
                        )
                        for _ in range(n - 1)
                    )
                )
                if n > 1
                else None
            )

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class Yolov6BepC3(nn.Module):
    """CSPStackRep Block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        e=0.5,
        block=Yolov6RepVGGBlock,
        activation_type="relu",
    ):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Yolov6ConvLayer(in_channels, c_, 1, 1, activation_type=activation_type)
        self.cv2 = Yolov6ConvLayer(in_channels, c_, 1, 1, activation_type=activation_type)
        self.cv3 = Yolov6ConvLayer(2 * c_, out_channels, 1, 1, activation_type=activation_type)

        self.m = Yolov6RepBlock(in_channels=c_, out_channels=c_, n=n, block=Yolov6BottleRep, basic_block=block)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPFModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, activation_type="silu"):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Yolov6ConvLayer(in_channels, c_, 1, 1, activation_type=activation_type)
        self.cv2 = Yolov6ConvLayer(c_ * 4, out_channels, 1, 1, activation_type=activation_type)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class CSPSPPFModule(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5, activation_type="silu"):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Yolov6ConvLayer(in_channels, c_, 1, 1, activation_type=activation_type)
        self.cv2 = Yolov6ConvLayer(in_channels, c_, 1, 1, activation_type=activation_type)
        self.cv3 = Yolov6ConvLayer(c_, c_, 3, 1, activation_type=activation_type)
        self.cv4 = Yolov6ConvLayer(c_, c_, 1, 1, activation_type=activation_type)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = Yolov6ConvLayer(4 * c_, c_, 1, 1, activation_type=activation_type)
        self.cv6 = Yolov6ConvLayer(c_, c_, 3, 1, activation_type=activation_type)
        self.cv7 = Yolov6ConvLayer(2 * c_, out_channels, 1, 1, activation_type=activation_type)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


class Transpose(nn.Module):
    """Normal Transpose, default for upsampling"""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class BiFusion(nn.Module):
    """BiFusion Block in PAN"""

    def __init__(self, in_channels, out_channels, activation_type="relu"):
        super().__init__()
        self.cv1 = Yolov6ConvLayer(in_channels[0], out_channels, 1, 1, activation_type=activation_type)
        self.cv2 = Yolov6ConvLayer(in_channels[1], out_channels, 1, 1, activation_type=activation_type)
        self.cv3 = Yolov6ConvLayer(out_channels * 3, out_channels, 1, 1, activation_type=activation_type)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = Yolov6ConvLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            activation_type=activation_type,
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


class RepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck + RepBiFPANNeck module.
    """

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=Yolov6RepVGGBlock,
        csp_e=float(1) / 2,
        stage_block_type="Yolov6BepC3",
        activation_type="relu",
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "Yolov6BepC3":
            stage_block = Yolov6BepC3
        elif stage_block_type == "Yolov6RepBlock":
            stage_block = Yolov6RepBlock
        else:
            raise NotImplementedError

        self.reduce_layer0 = Yolov6ConvLayer(
            in_channels=channels_list[4],  # 1024
            out_channels=channels_list[5],  # 256
            kernel_size=1,
            stride=1,
            activation_type=activation_type,
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]],  # 512, 256
            out_channels=channels_list[5],  # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[5],  # 256
            out_channels=channels_list[5],  # 256
            n=num_repeats[5],
            e=csp_e,
            block=block,
        )

        self.reduce_layer1 = Yolov6ConvLayer(
            in_channels=channels_list[5],  # 256
            out_channels=channels_list[6],  # 128
            kernel_size=1,
            stride=1,
            activation_type=activation_type,
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]],  # 256, 128
            out_channels=channels_list[6],  # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[6],  # 128
            out_channels=channels_list[6],  # 128
            n=num_repeats[6],
            e=csp_e,
            block=block,
        )

        self.downsample2 = Yolov6ConvLayer(
            in_channels=channels_list[6],  # 128
            out_channels=channels_list[7],  # 128
            kernel_size=3,
            stride=2,
            activation_type=activation_type,
        )

        self.Rep_n3 = stage_block(
            in_channels=channels_list[6] + channels_list[7],  # 128 + 128
            out_channels=channels_list[8],  # 256
            n=num_repeats[7],
            e=csp_e,
            block=block,
        )

        self.downsample1 = Yolov6ConvLayer(
            in_channels=channels_list[8],  # 256
            out_channels=channels_list[9],  # 256
            kernel_size=3,
            stride=2,
            activation_type=activation_type,
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[5] + channels_list[9],  # 256 + 256
            out_channels=channels_list[10],  # 512
            n=num_repeats[8],
            e=csp_e,
            block=block,
        )

    def forward(self, input):
        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]

        return outputs


class Yolov6Embeddings(nn.Module):
    def __init__(self, config: Yolov6Config) -> None:
        super().__init__()
        self.config = config

        if config.block_type == "Yolov6ConvBNSilu":
            block = Yolov6ConvLayer
        elif config.block_type == "Yolov6RepVGGBlock":
            block = Yolov6RepVGGBlock
        else:
            raise NotImplementedError
            
        self.embedder = block(
            in_channels=config.in_channels,
            out_channels=config.backbone_out_channels[0],
            kernel_size=3,
            stride=2,
        )

        self.in_channels = config.in_channels

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        num_channels = pixel_values.shape[1]
        if num_channels != self.in_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.embedder(pixel_values)
        return embedding


class Yolov6Encoder(nn.Module):
    def __init__(self, config: Yolov6Config) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList()

        assert config.backbone_out_channels is not None
        assert config.backbone_num_repeats is not None

        if config.block_type == "Yolov6ConvBNSilu":
            block = Yolov6ConvLayer
        elif config.block_type == "Yolov6RepVGGBlock":
            block = Yolov6RepVGGBlock
        else:
            raise NotImplementedError

        if config.backbone_stage_block_type == "Yolov6BepC3":
            stage_block = Yolov6BepC3
        elif config.backbone_stage_block_type == "Yolov6RepBlock":
            stage_block = Yolov6RepBlock
        else:
            raise NotImplementedError

        self.backbone_fuse_P2 = config.backbone_fuse_P2

        for i in range(0, len(config.backbone_out_channels) - 1):
            # ERBlock
            er_block = nn.Sequential(
                block(
                    in_channels=config.backbone_out_channels[i],
                    out_channels=config.backbone_out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                ),
                stage_block(
                    in_channels=config.backbone_out_channels[i + 1],
                    out_channels=config.backbone_out_channels[i + 1],
                    n=config.backbone_num_repeats[i + 1],
                    e=config.backbone_csp_e,
                    block=block,
                ),
            )

            # Channel merge layer
            if config.backbone_cspsppf:
                channel_merge_layer = CSPSPPFModule
            else:
                channel_merge_layer = SPPFModule

            if i == len(config.backbone_out_channels) - 2:
                if config.block_type == "Yolov6ConvBNSilu":
                    er_block.add_module(
                        "channel_merge_layer",
                        channel_merge_layer(
                            in_channels=config.backbone_out_channels[i + 1],
                            out_channels=config.backbone_out_channels[i + 1],
                            kernel_size=5,
                            activation_type="silu",
                        ),
                    )
                else:
                    er_block.add_module(
                        "channel_merge_layer",
                        channel_merge_layer(
                            in_channels=config.backbone_out_channels[i + 1],
                            out_channels=config.backbone_out_channels[i + 1],
                            kernel_size=5,
                            activation_type="relu",
                        ),
                    )
            self.blocks.add_module(f"ERBlock_{i+2}", er_block)

    def forward(
        self,
        hidden_states: Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for n, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if self.backbone_fuse_P2 and n == 0:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            else:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class Yolov6Neck(nn.Module):
    def __init__(self, config: Yolov6Config) -> None:
        super().__init__()
        self.config = config

        assert config.neck_out_channels is not None
        assert config.neck_num_repeats is not None

        channels_list = config.backbone_out_channels + config.neck_out_channels
        num_repeats = config.backbone_num_repeats + config.neck_num_repeats

        self.stage = RepBiFPANNeck(
            channels_list=channels_list,
            num_repeats=num_repeats,
            csp_e=config.neck_csp_e,
            stage_block_type=config.neck_stage_block_type,
        )

    def forward(
        self,
        hidden_state: Tensor,
    ):
        hidden_state = self.stage(hidden_state)
        return hidden_state


class Yolov6Head(nn.Module):
    export = False
    """Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    """

    def __init__(self, config: Yolov6Config):  # detection layer
        super().__init__()

        layers = []
        for i in range(config.head_num_layers):
            layers += [
                Yolov6ConvLayer(
                    config.neck_out_channels[2 * i + 1],
                    config.neck_out_channels[2 * i + 1],
                    kernel_size=1,
                    stride=1,
                    activation_type="silu",
                ),
                Yolov6ConvLayer(
                    config.neck_out_channels[2 * i + 1],
                    config.neck_out_channels[2 * i + 1],
                    kernel_size=3,
                    stride=1,
                    activation_type="silu",
                ),
                Yolov6ConvLayer(
                    config.neck_out_channels[2 * i + 1],
                    config.neck_out_channels[2 * i + 1],
                    kernel_size=3,
                    stride=1,
                    activation_type="silu",
                ),
                nn.Conv2d(config.neck_out_channels[2 * i + 1], config.num_labels, kernel_size=1),
                nn.Conv2d(config.neck_out_channels[2 * i + 1], 4 * (config.reg_max + 1), kernel_size=1),
            ]
        head_layers = nn.Sequential(*layers)

        self.config = config

        self.grid = [torch.zeros(1)] * config.head_num_layers
        self.prior_prob = 1e-2
        self.inplace = True
        self.stride = torch.tensor(config.head_strides)
        self.proj_conv = nn.Conv2d(config.reg_max_proj + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(config.head_num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])

        self.initialize_biases()

    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(
                -1,
            )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(
                -1,
            )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.config.reg_max_proj, self.config.reg_max_proj + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, self.config.reg_max_proj + 1, 1, 1]).clone().detach(),
            requires_grad=False,
        )

    def forward(self, x):
        cls_score_list = []
        reg_dist_list = []
        reg_distri_list = []
        if self.training:
            for i in range(self.config.head_num_layers):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            for i in range(self.config.head_num_layers):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.config.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.config.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(nn.functional.softmax(reg_output, dim=1))

                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.config.num_labels, l]))
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))

            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)

            anchor_points, stride_tensor = generate_anchors(
                x,
                self.stride,
                self.grid_cell_size,
                self.grid_cell_offset,
                device=x[0].device,
                is_eval=True,
                mode="af",
            )

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")
            pred_bboxes *= stride_tensor
            return None, cls_score_list, pred_bboxes


class Yolov6PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Yolov6Config
    base_model_prefix = "yolov6"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


YOLOV6_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Yolov6Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

YOLOV6_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`Yolov6ImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare YOLOV6 Model backbone with neck outputting raw hidden-states without any specific head on top.",
    YOLOV6_START_DOCSTRING,
)
class Yolov6Model(Yolov6PreTrainedModel):
    def __init__(self, config: Yolov6Config):
        super().__init__(config)
        self.config = config

        self.embedder = Yolov6Embeddings(config)
        self.encoder = Yolov6Encoder(config)
        self.neck = Yolov6Neck(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(YOLOV6_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embedder(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        outputs = self.neck(encoder_outputs.hidden_states)

        if not return_dict:
            return outputs

        return BaseModelOutputWithNoAttention(last_hidden_state=None, hidden_states=outputs)


@add_start_docstrings(
    """
    YOLOV6 Model with object detection heads on top, for tasks such as COCO detection.
    """,
    YOLOV6_START_DOCSTRING,
)
class Yolov6ForObjectDetection(Yolov6PreTrainedModel):
    def __init__(self, config: Yolov6Config):
        super().__init__(config)

        # Yolov6 backbone and neck model
        self.model = Yolov6Model(config)

        # Yolov6 object detection heads
        self.head = Yolov6Head(config)            

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(YOLOV6_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Yolov6ObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[List[Dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Yolov6ObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: `'class_labels'` and `'boxes'` (the class labels and bounding boxes of an image in the
            batch respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding
            boxes in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image,
            4)`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
        >>> model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
        ...     0
        ... ]

        >>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        Detected remote with confidence 0.994 at location [46.96, 72.61, 181.02, 119.73]
        Detected remote with confidence 0.975 at location [340.66, 79.19, 372.59, 192.65]
        Detected cat with confidence 0.984 at location [12.27, 54.25, 319.42, 470.99]
        Detected remote with confidence 0.922 at location [41.66, 71.96, 178.7, 120.33]
        Detected cat with confidence 0.914 at location [342.34, 21.48, 638.64, 372.46]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # First, sent images through YOLOS base model to obtain hidden states
        outputs = self.model(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        feats, logits, pred_boxes = self.head(outputs.hidden_states)

        loss, loss_dict = None, None
        if labels is not None:
            # First: create the matcher
            matcher = YolosHungarianMatcher(
                class_cost=self.config.class_cost,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
            )
            # Second: create the criterion
            losses = ["labels", "boxes", "cardinality"]
            criterion = Yolov6Loss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": 1, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient

            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if not return_dict:
            output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return Yolov6ObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


# Copied from transformers.models.detr.modeling_detr.dice_loss
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.DetrLoss with Detr->Yolos
class Yolov6Loss(nn.Module):
    """
    This class computes the losses for YolosForObjectDetection/YolosForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`YolosHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(source_boxes),
                center_to_corners_format(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
            "dfl": self.loss_dfl,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with Detr->Yolos
class YolosMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Copied from transformers.models.detr.modeling_detr.DetrHungarianMatcher with Detr->Yolos
class YolosHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# Copied from transformers.models.detr.modeling_detr.NestedTensor
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


# Copied from transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
