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
from functools import partial
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
)
from .configuration_yolov6 import Yolov6Config


if is_scipy_available():
    pass

if is_vision_available():
    pass


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Yolov6Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "superbai/yolov6n"
_EXPECTED_OUTPUT_SHAPE = [1, 3401, 384]


YOLOV6_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "superbai/yolov6n",
    # See all YOLOS models at https://huggingface.co/models?filter=yolov6
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
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).to(feats[0].dtype)
            if mode == "af":  # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=feats[0].dtype, device=device))
            elif mode == "ab":  # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=feats[0].dtype, device=device).repeat(3, 1))
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
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        bias=False,
        activation_type=None,
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
        **kwargs,
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
                nn.BatchNorm2d(num_features=in_channels, eps=0.001, momentum=0.03)
                if out_channels == in_channels and stride == 1
                else None
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
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

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
    def __init__(self, in_channels, out_channels, basic_block=Yolov6RepVGGBlock, activation_type=None, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels, activation_type=activation_type)
        self.conv2 = basic_block(out_channels, out_channels, activation_type=activation_type)
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
        activation_type=None,
        **kwargs,
    ):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == Yolov6BottleRep:
            self.conv1 = block(
                in_channels, out_channels, basic_block=basic_block, activation_type=activation_type, weight=True
            )
            n = n // 2
            self.block = (
                nn.Sequential(
                    *(
                        block(
                            out_channels,
                            out_channels,
                            basic_block=basic_block,
                            activation_type=activation_type,
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

        self.m = Yolov6RepBlock(
            in_channels=c_,
            out_channels=c_,
            n=n,
            block=Yolov6BottleRep,
            basic_block=block,
            activation_type=activation_type,
        )

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
            activation_type="relu",
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
            activation_type="relu",
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
            activation_type="relu",
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
            activation_type="relu",
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


class RepBiFPANNeck_P6(nn.Module):
    """
    CSPRepBiFPANNeck_P6 + RepBiFPANNeck_P6 module.
    """

    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]
    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=Yolov6BottleRep,
        csp_e=float(1) / 2,
        stage_block_type="Yolov6BepC3",
        activation_type="relu",
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "Yolov6BepC3":
            stage_block = partial(Yolov6BepC3, activation_type=activation_type)
        else:
            raise NotImplementedError

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = Yolov6ConvLayer(
            in_channels=channels_list[5],  # 1024
            out_channels=channels_list[6],  # 512
            kernel_size=1,
            stride=1,
            activation_type="relu",
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[4], channels_list[6]],  # 768, 512
            out_channels=channels_list[6],  # 512
        )

        self.Rep_p5 = stage_block(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[6],  # 512
            n=num_repeats[6],
            e=csp_e,
            block=block,
        )

        self.reduce_layer1 = Yolov6ConvLayer(
            in_channels=channels_list[6],  # 512
            out_channels=channels_list[7],  # 256
            kernel_size=1,
            stride=1,
            activation_type="relu",
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[3], channels_list[7]],  # 512, 256
            out_channels=channels_list[7],  # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[7],  # 256
            n=num_repeats[7],
            e=csp_e,
            block=block,
        )

        self.reduce_layer2 = Yolov6ConvLayer(
            in_channels=channels_list[7],  # 256
            out_channels=channels_list[8],  # 128
            kernel_size=1,
            stride=1,
            activation_type="relu",
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[2], channels_list[8]],  # 256, 128
            out_channels=channels_list[8],  # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8],  # 128
            n=num_repeats[8],
            e=csp_e,
            block=block,
        )

        self.downsample2 = Yolov6ConvLayer(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8],  # 128
            kernel_size=3,
            stride=2,
            activation_type="relu",
        )

        self.Rep_n4 = stage_block(
            in_channels=channels_list[8] + channels_list[8],  # 128 + 128
            out_channels=channels_list[9],  # 256
            n=num_repeats[9],
            e=csp_e,
            block=block,
        )

        self.downsample1 = Yolov6ConvLayer(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9],  # 256
            kernel_size=3,
            stride=2,
            activation_type="relu",
        )

        self.Rep_n5 = stage_block(
            in_channels=channels_list[7] + channels_list[9],  # 256 + 256
            out_channels=channels_list[10],  # 512
            n=num_repeats[10],
            e=csp_e,
            block=block,
        )

        self.downsample0 = Yolov6ConvLayer(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10],  # 512
            kernel_size=3,
            stride=2,
            activation_type="relu",
        )

        self.Rep_n6 = stage_block(
            in_channels=channels_list[6] + channels_list[10],  # 512 + 512
            out_channels=channels_list[11],  # 1024
            n=num_repeats[11],
            e=csp_e,
            block=block,
        )

    def forward(self, input):
        (x4, x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p5(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        f_out1 = self.Rep_p4(f_concat_layer1)

        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bifusion2([fpn_out2, x3, x4])
        pan_out3 = self.Rep_p3(f_concat_layer2)  # P3

        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2)  # P4

        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1)  # P5

        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0)  # P6

        outputs = [pan_out3, pan_out2, pan_out1, pan_out0]

        return outputs


class Yolov6Embeddings(nn.Module):
    def __init__(self, config: Yolov6Config) -> None:
        super().__init__()
        self.config = config

        if config.block_type == "Yolov6ConvBNSilu":
            block = Yolov6ConvLayer
            activation_type = "silu"
        elif config.block_type == "Yolov6RepVGGBlock":
            block = Yolov6RepVGGBlock
            activation_type = None
        else:
            raise NotImplementedError

        self.embedder = block(
            in_channels=config.in_channels,
            out_channels=config.backbone_out_channels[0],
            kernel_size=3,
            stride=2,
            activation_type=activation_type,
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
            activation_type = "silu"
            block = partial(Yolov6ConvLayer, activation_type=activation_type)
        elif config.block_type == "Yolov6RepVGGBlock":
            activation_type = "relu"
            block = partial(Yolov6RepVGGBlock, activation_type=None)
        else:
            raise NotImplementedError

        if config.backbone_stage_block_type == "Yolov6BepC3":
            stage_block = partial(Yolov6BepC3, activation_type=activation_type)
        elif config.backbone_stage_block_type == "Yolov6RepBlock":
            stage_block = partial(Yolov6RepBlock, activation_type=activation_type)
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

        if config.block_type == "Yolov6ConvBNSilu":
            activation_type = "silu"
            block = partial(Yolov6ConvLayer, activation_type=activation_type)
        elif config.block_type == "Yolov6RepVGGBlock":
            activation_type = "relu"
            block = Yolov6RepVGGBlock
        else:
            raise NotImplementedError

        if len(config.backbone_out_channels) == 5:
            self.stage = RepBiFPANNeck(
                channels_list=channels_list,
                num_repeats=num_repeats,
                block=block,
                csp_e=config.neck_csp_e,
                stage_block_type=config.neck_stage_block_type,
                activation_type=activation_type,
            )
        else:
            self.stage = RepBiFPANNeck_P6(
                channels_list=channels_list,
                num_repeats=num_repeats,
                block=block,
                csp_e=config.neck_csp_e,
                stage_block_type=config.neck_stage_block_type,
                activation_type=activation_type,
            )

    def forward(
        self,
        hidden_state: Tensor,
    ):
        hidden_state = self.stage(hidden_state)
        return hidden_state


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


class Yolov6Head(Yolov6PreTrainedModel):
    def __init__(self, config: Yolov6Config):  # detection layer
        super().__init__(config)

        index_function = (lambda x: 2 * x + 1) if len(config.backbone_out_channels) == 5 else (lambda x: x + 2)
        self.stems = nn.ModuleList(
            [
                Yolov6ConvLayer(
                    config.neck_out_channels[index_function(i)],
                    config.neck_out_channels[index_function(i)],
                    kernel_size=1,
                    activation_type="silu",
                )
                for i in range(config.head_num_layers)
            ]
        )

        self.cls_convs = nn.ModuleList(
            [
                Yolov6ConvLayer(
                    config.neck_out_channels[index_function(i)],
                    config.neck_out_channels[index_function(i)],
                    kernel_size=1,
                    activation_type="silu",
                )
                for i in range(config.head_num_layers)
            ]
        )
        self.reg_convs = nn.ModuleList(
            [
                Yolov6ConvLayer(
                    config.neck_out_channels[index_function(i)],
                    config.neck_out_channels[index_function(i)],
                    kernel_size=1,
                    activation_type="silu",
                )
                for i in range(config.head_num_layers)
            ]
        )
        self.cls_preds = nn.ModuleList(
            [
                Yolov6ConvLayer(
                    config.neck_out_channels[index_function(i)],
                    config.num_labels,
                    kernel_size=1,
                )
                for i in range(config.head_num_layers)
            ]
        )
        self.reg_preds = nn.ModuleList(
            [
                Yolov6ConvLayer(
                    config.neck_out_channels[index_function(i)],
                    4 * (config.reg_max + 1),
                    kernel_size=1,
                )
                for i in range(config.head_num_layers)
            ]
        )

        self.stride = torch.tensor(config.head_strides)

        self.post_init()

        self.proj = nn.Parameter(
            torch.linspace(0, config.reg_max_proj, config.reg_max_proj + 1),
            requires_grad=False,
        )
        self.proj_conv = nn.Conv2d(config.reg_max_proj + 1, 1, 1, bias=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, config.reg_max_proj + 1, 1, 1]).clone().detach(),
            requires_grad=False,
        )

    def forward(self, x):
        cls_score_list = []
        reg_distri_list = []

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

            if self.training:
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
            else:
                if self.config.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.config.reg_max_proj + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(nn.functional.softmax(reg_output, dim=1))

                if self.config.export:
                    cls_score_list.append(cls_output)
                    reg_distri_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.config.num_labels, l]))
                    reg_distri_list.append(reg_output.reshape([b, 4, l]))

        if self.config.export:
            return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_distri_list))

        if self.training:
            cls_score_list = torch.cat(cls_score_list, axis=1)
            pred_bboxes = torch.cat(reg_distri_list, axis=1)
        else:
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            pred_bboxes = torch.cat(reg_distri_list, axis=-1).permute(0, 2, 1)

            anchor_points, stride_tensor = generate_anchors(
                x,
                self.stride,
                device=x[0].device,
                is_eval=True,
            )
            pred_bboxes = dist2bbox(pred_bboxes, anchor_points, box_format="xywh")
            pred_bboxes *= stride_tensor

        return x, cls_score_list, pred_bboxes


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

        # Only maintain reg_max value to non-zero in training pipeline
        reg_max = self.config.reg_max if self.training else 0

        feats, logits, pred_boxes = self.head(outputs.hidden_states)
        loss, loss_dict = None, None

        if labels is not None:
            losses = ["classes", "boxes"]
            criterion = Yolov6Loss(
                num_classes=self.config.num_labels,
                warmup_epoch=self.config.atss_warmup_epoch,
                use_dfl=self.config.use_dfl,
                iou_type=self.config.iou_type,
                fpn_strides=self.config.head_strides,
                reg_max=reg_max,
                losses=losses,
                training=self.training,
            )
            criterion.to(self.device)
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["feats"] = feats
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes

            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_classes": self.config.class_loss_coefficient,
                "loss_iou": self.config.iou_loss_coefficient,
            }
            weight_dict["loss_dfl"] = self.config.dfl_loss_coefficient

            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # hacky solution for evaluation : normalize pred_boxes into 0~1 scale
        pred_boxes = pred_boxes / torch.tensor(
            [
                pixel_values.shape[-1],
                pixel_values.shape[-2],
                pixel_values.shape[-1],
                pixel_values.shape[-2],
            ]
            * (reg_max + 1)
        ).to(pred_boxes.device)

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


def iou_loss(box1, box2, iou_type, box_format="xyxy", eps=1e-10):
    """calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4]."""
    if box1.shape[0] != box2.shape[0]:
        box2 = box2.T
        if box_format == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        elif box_format == "xywh":
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        if box_format == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

        elif box_format == "xywh":
            b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
            b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
            b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
            b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
            b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
            b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    if iou_type == "giou":
        c_area = cw * ch + eps  # convex area
        iou = iou - (c_area - union) / c_area
    elif iou_type in ["diou", "ciou"]:
        c2 = cw**2 + ch**2 + eps  # convex diagonal squared
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
        ) / 4  # center distance squared
        if iou_type == "diou":
            iou = iou - rho2 / c2
        elif iou_type == "ciou":
            v = (4 / math.pi**2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            iou = iou - (rho2 / c2 + v * alpha)
    elif iou_type == "siou":
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        iou = iou - 0.5 * (distance_cost + shape_cost)
    loss = 1.0 - iou

    return loss


def df_loss(pred_dist, target, reg_max):
    # Calculate the left and right indices of the target values
    target_left = target.to(torch.long)
    target_right = target_left + 1

    # Calculate weights based on the distance from the target values
    weight_left = target_right.to(torch.float) - target
    weight_right = 1 - weight_left

    # Calculate loss for the left and right targets
    loss_left = (
        nn.functional.cross_entropy(pred_dist.view(-1, reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape
        )
        * weight_left
    )
    loss_right = (
        nn.functional.cross_entropy(pred_dist.view(-1, reg_max + 1), target_right.view(-1), reduction="none").view(
            target_left.shape
        )
        * weight_right
    )

    # Calculate the final loss by summing the left and right losses and maintaining the dimension
    return (loss_left + loss_right).mean(-1, keepdim=True)


# Copied from transformers.models.detr.modeling_detr.DetrLoss with Detr->Yolos
class Yolov6Loss(nn.Module):
    """
    This class computes the losses for Yolov6ForObjectDetection. The process happens in two steps: 1)
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

    def __init__(
        self,
        num_classes,
        warmup_epoch,
        use_dfl,
        iou_type,
        fpn_strides,
        reg_max,
        losses,
        training,
    ):
        super().__init__()
        self.fpn_strides = fpn_strides
        self.cached_feat_sizes = [torch.Size([0, 0]) for _ in fpn_strides]
        self.cached_anchors = None
        self.num_classes = num_classes
        self.losses = losses
        self.training = training

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type

        self.box_format = "xyxy"
        self.gamma = 2.0
        self.alpha = 0.75

    # removed logging parameter, which was part of the original implementation
    def loss_classes(self, outputs, targets, fg_mask, target_scores_sum):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs and "target_scores" not in targets and "target_labels" not in targets:
            raise KeyError("No logits were found in the outputs")

        target_scores = targets["target_scores"]
        target_labels = targets["target_labels"]

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = nn.functional.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]

        pred_scores = outputs["pred_scores"]

        weight = self.alpha * pred_scores.pow(self.gamma) * (1 - one_hot_label) + target_scores * one_hot_label
        with torch.cuda.amp.autocast(enabled=False):
            loss_classes = (
                nn.functional.binary_cross_entropy(pred_scores.float(), target_scores.float(), reduction="none")
                * weight
            ).sum()

        loss_classes /= target_scores_sum

        losses = {"loss_classes": loss_classes}

        return losses

    def loss_boxes(self, outputs, targets, fg_mask, target_scores_sum):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        target_scores = targets["target_scores"]
        target_bboxes = targets["target_bboxes"]

        pred_bboxes = outputs["pred_bboxes"]
        pred_dist = outputs["pred_boxes"]
        anchor_points = outputs["anchor_points_s"]

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = (
                iou_loss(
                    pred_bboxes_pos,
                    target_bboxes_pos,
                    iou_type=self.iou_type,
                    box_format="xyxy",
                )
                * bbox_weight
            )

            # dfl loss
            if self.use_dfl and self.training:
                dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = df_loss(pred_dist_pos, target_ltrb_pos, self.reg_max) * bbox_weight
            else:
                loss_dfl = pred_dist.sum() * 0.0

        else:
            loss_iou = pred_dist.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0

        losses = {}
        losses["loss_iou"] = loss_iou.sum() / target_scores_sum
        losses["loss_dfl"] = loss_dfl.sum() / target_scores_sum
        return losses

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = nn.functional.softmax(
                pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1
            ).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)

    def get_loss(self, loss, outputs, targets, fg_mask, target_scores_sum):
        loss_map = {
            "classes": self.loss_classes,
            "boxes": self.loss_boxes,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, fg_mask, target_scores_sum)

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
        feats, logits, pred_distri = (
            outputs["feats"],
            outputs["logits"],
            outputs["pred_boxes"],
        )
        pred_scores = logits.sigmoid()
        outputs["pred_scores"] = pred_scores

        if all(feat.shape[2:] == cfsize for feat, cfsize in zip(feats, self.cached_feat_sizes)):
            anchors, anchor_points, n_anchors_list, stride_tensor = self.cached_anchors
        else:
            self.cached_feat_sizes = [feat.shape[2:] for feat in feats]
            anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
                feats,
                self.fpn_strides,
                device=feats[0].device,
            )
            self.cached_anchors = anchors, anchor_points, n_anchors_list, stride_tensor

        assert pred_scores.type() == pred_distri.type()

        # targets
        max_size = max(v["class_labels"].size(0) for v in targets)

        gt_labels = torch.stack(
            [
                torch.cat(
                    [
                        v["class_labels"],
                        -torch.ones(
                            max_size - v["class_labels"].size(0),
                            dtype=v["class_labels"].dtype,
                        ).to(v["class_labels"].device),
                    ]
                )
                for v in targets
            ]
        )
        gt_bboxes = torch.stack(
            [
                torch.cat(
                    [
                        v["boxes"],
                        torch.zeros(max_size - v["boxes"].size(0), 4, dtype=v["boxes"].dtype).to(v["boxes"].device),
                    ]
                )
                for v in targets
            ]
        )

        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        outputs["anchor_points_s"] = anchor_points_s
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy
        outputs["pred_bboxes"] = pred_bboxes
        # in case of pred_scores might contain nan in train/validation step
        contains_nan = torch.isnan(pred_scores).any().item() or torch.isnan(pred_bboxes).any().item()
        if contains_nan:
            target_labels, target_bboxes, target_scores, fg_mask = self.warmup_assigner(
                anchors,
                n_anchors_list,
                gt_labels,
                gt_bboxes,
                mask_gt,
                None,  # pred_bboxes.detach() * stride_tensor
            )
        else:
            target_labels, target_bboxes, target_scores, fg_mask = self.formal_assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        # rescale bbox
        target_bboxes /= stride_tensor

        batch_targets = {}
        batch_targets["target_scores"] = target_scores
        batch_targets["target_labels"] = target_labels
        batch_targets["target_bboxes"] = target_bboxes

        target_scores_sum = target_scores.sum()
        if target_scores_sum < 1:
            target_scores_sum = 1

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, batch_targets, fg_mask, target_scores_sum))

        return losses


# Copied from https://github.com/meituan/YOLOv6/blob/87dd3d3963b6b373ccdc626b9bae5a2afec5639e/yolov6/assigners/atss_assigner.py#L7
class ATSSAssigner(nn.Module):
    """Adaptive Training Sample Selection Assigner"""

    def __init__(self, topk=9, num_classes=80):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes

    @torch.no_grad()
    def forward(self, anc_bboxes, n_level_bboxes, gt_labels, gt_bboxes, mask_gt, pd_bboxes):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anc_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List):len(3)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.n_anchors = anc_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full([self.bs, self.n_anchors], self.bg_idx).to(device),
                torch.zeros([self.bs, self.n_anchors, 4]).to(device),
                torch.zeros([self.bs, self.n_anchors, self.num_classes]).to(device),
                torch.zeros([self.bs, self.n_anchors]).to(device),
            )

        overlaps = iou2d_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        distances, ac_points = dist_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        is_in_candidate, candidate_idxs = self.select_topk_candidates(distances, n_level_bboxes, mask_gt)

        overlaps_thr_per_gt, iou_candidates = self.thres_calculator(is_in_candidate, candidate_idxs, overlaps)

        # select candidates iou >= threshold as positive
        is_pos = torch.where(
            iou_candidates > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_candidate,
            torch.zeros_like(is_in_candidate),
        )

        is_in_gts = select_candidates_in_gts(ac_points, gt_bboxes)
        mask_pos = is_pos * is_in_gts * mask_gt

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # soft label with iou
        if pd_bboxes is not None:
            ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            target_scores *= ious

        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool()

    def select_topk_candidates(self, distances, n_level_bboxes, mask_gt):
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):
            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_idxs.append(per_level_topk_idxs + start_idx)
            per_level_topk_idxs = torch.where(mask_gt, per_level_topk_idxs, torch.zeros_like(per_level_topk_idxs))
            is_in_candidate = nn.functional.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1, torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_idx = end_idx

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    def thres_calculator(self, is_in_candidate, candidate_idxs, overlaps):
        n_bs_max_boxes = self.bs * self.n_max_boxes
        _candidate_overlaps = torch.where(is_in_candidate > 0, overlaps, torch.zeros_like(overlaps))
        candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * torch.arange(n_bs_max_boxes, device=candidate_idxs.device)
        assist_idxs = assist_idxs[:, None]
        faltten_idxs = candidate_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[faltten_idxs]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        # assigned target labels
        batch_idx = torch.arange(self.bs, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_idx = batch_idx[..., None]
        target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
        target_labels = gt_labels.flatten()[target_gt_idx.flatten()]
        target_labels = target_labels.reshape([self.bs, self.n_anchors])
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.bg_idx))

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
        target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores
        target_scores = nn.functional.one_hot(target_labels.long(), self.num_classes + 1).float()
        target_scores = target_scores[:, :, : self.num_classes]

        return target_labels, target_bboxes, target_scores


# Copied from https://github.com/meituan/YOLOv6/blob/87dd3d3963b6b373ccdc626b9bae5a2afec5639e/yolov6/assigners/tal_assigner.py#L6
class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = (
            [],
            [],
            [],
            [],
        )
        # loop batch dim in case of numerous object box
        for i in range(cycle):
            start, end = i * step, (i + 1) * step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]

            mask_pos, align_metric, overlaps = self.get_pos_mask(
                pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_
            )

            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

            # assigned target
            target_labels, target_bboxes, target_scores = self.get_targets(
                gt_labels_, gt_bboxes_, target_gt_idx, fg_mask
            )

            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            target_scores = target_scores * norm_align_metric

            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)

        # concat
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts,
            topk_mask=mask_gt.repeat([1, 1, self.topk]).bool(),
        )
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        # ind[1] = gt_labels.squeeze(-1)
        ind[1] = gt_labels
        bbox_scores = pd_scores[ind[0], ind[1]]

        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = nn.functional.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]

        # assigned target scores
        target_labels[target_labels < 0] = 0
        target_scores = nn.functional.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))

        return target_labels, target_bboxes, target_scores


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


# Copied from https://github.com/meituan/YOLOv6/blob/87dd3d3963b6b373ccdc626b9bae5a2afec5639e/yolov6/assigners/iou2d_calculator.py#L22
def iou2d_calculator(bboxes1, bboxes2, mode="iou", is_aligned=False, scale=1.0, dtype=None):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    """Calculate IoU between 2D bboxes.

    Args:
        bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
            format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
        bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
            format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
            empty. If ``is_aligned `` is ``True``, then m and n must be
            equal.
        mode (str): "iou" (intersection over union), "iof" (intersection
            over foreground), or "giou" (generalized intersection over
            union).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert bboxes1.size(-1) in [0, 4, 5]
    assert bboxes2.size(-1) in [0, 4, 5]
    if bboxes2.size(-1) == 5:
        bboxes2 = bboxes2[..., :4]
    if bboxes1.size(-1) == 5:
        bboxes1 = bboxes1[..., :4]

    if dtype == "fp16":
        # change tensor type to save cpu and cuda memory and keep speed
        bboxes1 = cast_tensor_type(bboxes1, scale, dtype)
        bboxes2 = cast_tensor_type(bboxes2, scale, dtype)
        overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        if not overlaps.is_cuda and overlaps.dtype == torch.float16:
            # resume cpu float32
            overlaps = overlaps.float()
        return overlaps

    return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)


# Copied from https://github.com/meituan/YOLOv6/blob/87dd3d3963b6b373ccdc626b9bae5a2afec5639e/yolov6/assigners/iou2d_calculator.py#L7C1-L11C13
def cast_tensor_type(x, scale=1.0, dtype=None):
    if dtype == "fp16":
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


# Copied from https://github.com/meituan/YOLOv6/blob/87dd3d3963b6b373ccdc626b9bae5a2afec5639e/yolov6/assigners/iou2d_calculator.py#L14
def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = nn.functional.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)
    target_gt_idx = mask_pos.argmax(axis=-2)
    return target_gt_idx, fg_mask, mask_pos


def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union


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


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist


def xywh2xyxy(bboxes):
    """Transform bbox(xywh) to box(xyxy)."""
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes


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


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


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
