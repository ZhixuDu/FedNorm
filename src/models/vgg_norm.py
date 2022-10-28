from functools import partial
from typing import Union, List, Dict, Any, Optional, cast

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "VGG",
    "vgg11",
    "vgg11_nm",
    "vgg13",
    "vgg13_nm",
    "vgg16",
    "vgg16_nm",
    "vgg19",
    "vgg19_nm",
]

class FEATURES:
  content = []
  
class FedNorm(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    FEATURES.content.append(x)
    return x

class ConbineNorm(nn.Module):
  def __init__(self, planes, norm='cnbn'):
    super().__init__()
    if   'gn' in norm:
        self.norm = nn.GroupNorm(planes//16, planes)
    elif 'in' in norm: 
        self.norm = InstanceNorm2d(planes, planes)
    elif 'ln' in norm:
        self.norm = nn.GroupNorm(1, planes)
    elif 'bn' in norm:
        self.norm = nn.BatchNorm2d(planes)
    elif 'sbn' in norm:
        self.norm = nn.BatchNorm2d(planes, momentum=None)
    elif 'fbn' in norm:
        self.norm = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
    elif 'no' in norm:
        self.norm = NoNorm()
  def forward(self, x):
    x = self.norm(x)
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

def Norm2d(planes, norm='bn'):
    """
    Separate C channels into C groups (equivalent with InstanceNorm), use InstanceNorm2d instead, to avoid the case where height=1 and width=1
    
    Put all C channels into 1 single group (equivalent with LayerNorm)
    """
    if   norm == 'gn':
        return nn.GroupNorm(planes//16, planes)
    elif norm == 'in': 
        return InstanceNorm2d(planes, planes)
    elif norm == 'ln':
        return nn.GroupNorm(1, planes)
    elif norm == 'bn':
        return nn.BatchNorm2d(planes)
    elif norm == 'sbn':
        return nn.BatchNorm2d(planes, momentum=None)
    elif norm == 'fbn':
        return nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
    elif norm == 'no':
        return NoNorm()
    elif norm == 'ours':
        return FedNorm()
    elif 'cn' in norm:
        return ConbineNorm(planes)
        

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, norm: str = 'bn', restriction: str = 'mean+std', num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features    = features
        self.restriction = restriction
        '''
        The original implementation in the pytorch vgg model, however, this makes cifar10 ustable
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # makes model unstable on cifar10
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        '''
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self.conv_list = []
        self.mse = nn.MSELoss()
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    #m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    self.conv_list.append(m)
                elif isinstance(m, nn.BatchNorm2d) and norm != 'fbn':
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        FEATURES.content = []
        x = self.features(x)
        #x = self.avgpool(x) # comment this line to reduce possible additional problems (and for cifar10, this should be commented out)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x), FEATURES.content, 0
      
    def distri_norm(self, feature_list, global_stat):
        mean_list = []
        var_list  = []
        for (idx, feature), conv in zip(enumerate(feature_list), self.conv_list):
            mean = feature.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var  = ((feature - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            if global_stat is None:
                if self.restriction == 'mean+std':
                  loss = self.mse(mean, torch.zeros(mean.shape).to('cuda')) + self.mse(var, torch.ones(var.shape).to('cuda'))
                elif self.restriction == 'mean':
                  loss = self.mse(mean, torch.zeros(mean.shape).to('cuda'))
                elif self.restriction == 'std':
                  loss = self.mse(var, torch.ones(var.shape).to('cuda'))
                grad = torch.autograd.grad(loss, [conv.weight, conv.bias], retain_graph=True)
            else:
                if self.restriction == 'mean+std':
                  loss = self.mse(mean, global_stat[0][idx]) + self.mse(var, global_stat[1][idx])
                elif self.restriction == 'mean':
                  loss = self.mse(mean, global_stat[0][idx])
                elif self.restriction == 'std':
                  loss = self.mse(var, global_stat[1][idx])
                grad = torch.autograd.grad(loss, [conv.weight, conv.bias], retain_graph=True)
  
            if conv.weight.grad is None:
                conv.weight.grad = grad[0]
                conv.bias.grad = grad[1]
            else:
                conv.weight.grad += grad[0]
                conv.bias.grad += grad[1]
            
            mean_list.append(mean.detach())
            var_list.append(var.detach())

        return mean_list, var_list


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, norm: str = 'bn') -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, Norm2d(v, norm), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
    


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, norm: str, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, norm=norm), norm=norm, **kwargs)
    return model


def vgg11(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("A", False, norm, **kwargs)


def vgg11_nm(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("A", True, norm, **kwargs)


def vgg13(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("B", False, norm, **kwargs)


def vgg13_nm(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("B", True, norm, **kwargs)


def vgg16(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("D", False, norm, **kwargs)


def vgg16_nm(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("D", True, norm, **kwargs)


def vgg19(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("E", False, norm, **kwargs)


def vgg19_nm(*, norm='bn', **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    """
    return _vgg("E", True, norm, **kwargs)

if __name__ == '__main__':
    vgg = vgg11_nm(norm = 'bn', num_classes=10, dropout=0.5)
    state_dict = vgg.state_dict()
    for m in vgg.modules():
      if isinstance(m, nn.GroupNorm):
        print(m.weight.shape)
        print(m.bias.shape)
        print(m)
    for key in state_dict:
        print(key, state_dict[key].shape)
