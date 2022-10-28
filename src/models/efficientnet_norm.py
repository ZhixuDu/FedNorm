import copy
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, List, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

import torch.nn.functional as F

from models.misc import *


__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
]

class FEATURES:
  content = []
  
class FedNorm(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    FEATURES.content.append(x)
    return x

class NoNorm(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x

class InstanceNorm2d(nn.Module):
  def __init__(self, planes, affine=True):
    super().__init__()
    self.norm = nn.InstanceNorm2d(planes, affine=True)
    self.norm.weight.data.fill_(1)
    self.norm.bias.data.zero_()
  def forward(self, x):
    return x if (x.shape[2]== 1 and x.shape[3] == 1) else self.norm(x)

def Norm2d(planes, norm='bn', momentum=0.1, eps=1e-5):
    """
    Separate C channels into C groups (equivalent with InstanceNorm), use InstanceNorm2d instead, to avoid the case where height=1 and width=1
    
    Put all C channels into 1 single group (equivalent with LayerNorm)
    """
    if   norm == 'gn':
        return nn.GroupNorm(planes//16, planes, eps=eps)
    elif norm == 'in': 
        return InstanceNorm2d(planes, affine=True, momentum=momentum, eps=eps)
    elif norm == 'ln':
        return nn.GroupNorm(1, planes, eps=eps)
    elif norm == 'bn':
        return nn.BatchNorm2d(planes, momentum=momentum, eps=eps)
    elif norm == 'sbn':
        return nn.BatchNorm2d(planes, momentum=None, eps=eps)
    elif norm == 'fbn':
        return FrozenBatchNorm2d(planes, eps=eps)
    elif norm == 'no':
        return NoNorm()
    elif norm == 'ours':
        return FedNorm()

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)


    def _forward_impl(self, x: Tensor) -> Tensor:
        
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        FEATURES.content = []
        return F.log_softmax(self._forward_impl(x), dim=1), FEATURES.content, 0


def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    **kwargs: Any,
) -> EfficientNet:

    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)

    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def efficientnet_b0(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B0_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(inverted_residual_setting, 0.2, last_channel, norm_layer=partial(Norm2d, norm=norm), **kwargs)


def efficientnet_b1(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B1_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return _efficientnet(inverted_residual_setting, 0.2, last_channel, norm_layer=partial(Norm2d, norm=norm), **kwargs)


def efficientnet_b2(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B2_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return _efficientnet(inverted_residual_setting, 0.3, last_channel, norm_layer=partial(Norm2d, norm=norm), **kwargs)


def efficientnet_b3(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B3_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _efficientnet(inverted_residual_setting, 0.3, last_channel, norm_layer=partial(Norm2d, norm=norm), **kwargs)


def efficientnet_b4(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B4_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return _efficientnet(inverted_residual_setting, 0.4, last_channel, norm_layer=partial(Norm2d, norm=norm), **kwargs)


def efficientnet_b5(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B5_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        norm_layer=partial(Norm2d, norm=norm, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B6_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        norm_layer=partial(Norm2d, norm=norm, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        weights (EfficientNet_B7_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(
        inverted_residual_setting,
        0.5,
        last_channel,
        norm_layer=partial(Norm2d, norm=norm, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_v2_s(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (EfficientNet_V2_S_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        norm_layer=partial(Norm2d, norm=norm, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_m(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (EfficientNet_V2_M_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        norm_layer=partial(Norm2d, norm=norm, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_l(
    *, norm, **kwargs: Any
) -> EfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `"EfficientNetV2: Smaller Models and Faster Training" <https://arxiv.org/abs/2104.00298>`_.

    Args:
        weights (EfficientNet_V2_L_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        0.4,
        last_channel,
        norm_layer=partial(Norm2d, norm=norm, eps=1e-03),
        **kwargs,
    )