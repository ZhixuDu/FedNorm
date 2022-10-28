#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def Norm2d(planes, norm='bn', num_groups=None):
    """
    Separate C channels into C groups (equivalent with InstanceNorm), use InstanceNorm2d instead, to avoid the case where height=1 and width=1
    
    Put all C channels into 1 single group (equivalent with LayerNorm)
    """
    if   norm == 'gn':
        return nn.GroupNorm(planes//16, planes) if num_groups is None else nn.GroupNorm(num_groups, planes)
    elif norm == 'in': 
        return nn.InstanceNorm2d(planes, affine=True)
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

def Norm1d(planes, norm='bn', num_groups=None):
    """
    Separate C channels into C groups (equivalent with InstanceNorm), use InstanceNorm2d instead, to avoid the case where height=1 and width=1
    
    Put all C channels into 1 single group (equivalent with LayerNorm)
    """
    if   norm == 'gn':
        return nn.GroupNorm(planes//16, planes) if num_groups is None else nn.GroupNorm(num_groups, planes)
    elif norm == 'in': 
        return nn.InstanceNorm1d(planes, affine=True)
    elif norm == 'ln':
        return nn.GroupNorm(1, planes)
    elif norm == 'bn':
        return nn.BatchNorm1d(planes)
    elif norm == 'sbn':
        return nn.BatchNorm1d(planes, momentum=None)
    elif norm == 'fbn':
        return nn.BatchNorm1d(planes, affine=False, track_running_stats=False)
    elif norm == 'no':
        return NoNorm()
    elif norm == 'ours':
        return FedNorm()


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class MLP_general(nn.Module):
    def __init__(self, args):
        super(MLP_general, self).__init__()
        if(args.dataset == 'cifar10'):
            self.fc1 = nn.Linear(3072, 300)
        elif(args.dataset == 'mnist'):
            self.fc1 = nn.Linear(784, 300)
        elif(args.dataset == 'HAR'):
            self.fc1 = nn.Linear(561, 300)
        elif(args.dataset == 'HAD'):
            self.fc1 = nn.Linear(633, 300)
        self.fc2 = nn.Linear(300, 100)
        if(args.dataset == 'HAR'):
            self.fc3 = nn.Linear(100, 6)
        elif(args.dataset == 'HAD'):
            self.fc3 = nn.Linear(100, 12)
        else:
            self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        #return x

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        self.mse = nn.MSELoss()


    def forward(self, x):
        feature1 = self.conv1(x)
        x = F.max_pool2d(F.relu(feature1), 2)
        feature2 = self.conv2(x)
        x = F.max_pool2d(F.relu(feature2), 2)
        x = x.view(x.size(0), -1)
        feature3 = self.fc1(x)
        x = F.relu(feature3)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), [feature1, feature2, feature3], 0

    def distri_norm(self, feature_list, global_stat):
        feature1 = feature_list[0]
        mean1 = feature1.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var1 = ((feature1 - mean1) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        if global_stat is None:
            grad_weight1 = torch.autograd.grad(self.mse(mean1, torch.zeros(mean1.shape).to('cuda')) + self.mse(var1, torch.ones(var1.shape).to('cuda')), self.conv1.weight, retain_graph=True)
            grad_bias1 = torch.autograd.grad(self.mse(mean1, torch.zeros(mean1.shape).to('cuda')) + self.mse(var1, torch.ones(var1.shape).to('cuda')), self.conv1.bias, retain_graph=True)
        else:
            grad_weight1 = torch.autograd.grad(self.mse(mean1, global_stat[0][0]) + self.mse(var1, global_stat[1][0]), self.conv1.weight, retain_graph=True)
            grad_bias1 = torch.autograd.grad(self.mse(mean1, global_stat[0][0]) + self.mse(var1, global_stat[1][0]), self.conv1.bias, retain_graph=True)

        #grad_weight1 = torch.autograd.grad(self.mse(mean1, torch.zeros(mean1.shape).to('cuda')), self.conv1.weight, retain_graph=True)
        #grad_bias1 = torch.autograd.grad(self.mse(mean1, torch.zeros(mean1.shape).to('cuda')), self.conv1.bias, retain_graph=True)



        if self.conv1.weight.grad is None:
            self.conv1.weight.grad = grad_weight1[0]
            self.conv1.bias.grad = grad_bias1[0]
        else:
            self.conv1.weight.grad += grad_weight1[0]
            self.conv1.bias.grad += grad_bias1[0]

        feature2 = feature_list[1]
        mean2 = feature2.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var2 = ((feature2 - mean2) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        if global_stat is None:
            grad_weight2 = torch.autograd.grad(self.mse(mean2, torch.zeros(mean2.shape).to('cuda')) + self.mse(var2, torch.ones(var2.shape).to('cuda')), self.conv2.weight, retain_graph=True)
            grad_bias2 = torch.autograd.grad(self.mse(mean2, torch.zeros(mean2.shape).to('cuda')) + self.mse(var2, torch.ones(var2.shape).to('cuda')), self.conv2.bias, retain_graph=True)
        else:
            grad_weight2 = torch.autograd.grad(self.mse(mean2, global_stat[0][1]) + self.mse(var2, global_stat[1][1]), self.conv2.weight, retain_graph=True)
            grad_bias2 = torch.autograd.grad(self.mse(mean2, global_stat[0][1]) + self.mse(var2, global_stat[1][1]), self.conv2.bias, retain_graph=True)
        #grad_weight2 = torch.autograd.grad(self.mse(mean2, torch.zeros(mean2.shape).to('cuda')), self.conv2.weight, retain_graph=True)
        #grad_bias2 = torch.autograd.grad(self.mse(mean2, torch.zeros(mean2.shape).to('cuda')), self.conv2.bias, retain_graph=True)


        if self.conv2.weight.grad is None:
            self.conv2.weight.grad = grad_weight2[0]
            self.conv2.bias.grad = grad_bias2[0]
        else:
            self.conv2.weight.grad += grad_weight2[0]
            self.conv2.bias.grad += grad_bias2[0]

        return [mean1.detach(), mean2.detach()], [var1.detach(), var2.detach()]


    def print_grad(self):
        print(self.conv1.weight.grad)

    def print_distri(self, feature_list):
        feature1 = feature_list[0]
        mean1 = feature1.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var1 = ((feature1 - mean1) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
 
        feature2 = feature_list[1]
        mean2 = feature2.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var2 = ((feature2 - mean2) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        print(mean1)
        print(var1)
        print(mean2)
        print(var2)


class CNNMnist_norm(nn.Module):
    def __init__(self, args):
        super(CNNMnist_norm, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        if args.norm == 'bn':
            self.bn1 = nn.BatchNorm2d(10)
            self.bn2 = nn.BatchNorm2d(20)
        elif args.norm == 'gn':
            self.bn1 = nn.GroupNorm(2, 10)
            self.bn2 = nn.GroupNorm(4, 20)
        elif args.norm == 'sbn':
            self.bn1 = nn.BatchNorm2d(10, momentum=None)
            self.bn2 = nn.BatchNorm2d(20, momentum=None)
        self.mse = nn.MSELoss()


    def forward(self, x):
        feature1 = self.conv1(x)
        x = F.max_pool2d(F.relu(self.bn1(feature1)), 2)
        feature2 = self.conv2(x)
        x = F.max_pool2d(F.relu(self.bn2(feature2)), 2)
        x = x.view(x.size(0), -1)
        feature3 = self.fc1(x)
        x = F.relu(feature3)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), [feature1, feature2, feature3], 0


class CNNMnist_deep(nn.Module):
    def __init__(self, args):
        super(CNNMnist_deep, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5)
        #self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2048, 50)
        #self.bn4 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, args.num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        #print(x.shape)
        feature_1 = self.conv1(x)
        #feature_1_bn = self.bn1(feature_1)
        x = F.relu(feature_1)
        feature_2 = self.conv2(x)
        #feature_2_bn = self.bn2(feature_2)
        x = F.relu(feature_2)
        feature_3 = self.conv3(x)
        #feature_3_bn = self.bn3(feature_3)
        x = self.pool(F.relu(feature_3))

        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        feature_4 = self.fc1(x)
        #feature_4_bn = self.bn4(feature_4)
        x = F.relu(feature_4)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), [feature_1, feature_2, feature_3, feature_4], 0


class MLPMnist_BN(nn.Module):
    def __init__(self, args):
        super(MLPMnist_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.bn = nn.BatchNorm2d(1)
        self.fc2 = nn.Linear(576, args.num_classes)

    def forward(self, x):
        feature = self.conv1(x)
        feature_bn = self.bn(feature)
        x = F.relu(feature_bn)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), feature, feature_bn

    def print_bn(self):
        print(self.bn.running_mean)
        print(self.bn.running_var)
        print(self.bn.weight)
        print(self.bn.bias)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

class CNNFeMnist_sim(nn.Module):
    def __init__(self, args):
        super(CNNFeMnist_sim, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*20, 512)
        self.fc2 = nn.Linear(512, 62)

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)
        return F.log_softmax(out, dim=1)

class CNNFeMnist(nn.Module):
    def __init__(self, args):
        super(CNNFeMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(7*7*64, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x):
        #print(x.shape)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.fc2(out)
        #print(out.shape)
        return F.log_softmax(out, dim=1)
        

class CNNCifar_norm(nn.Module):
    def __init__(self, args):
        super(CNNCifar_norm, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.mse = nn.MSELoss()
        self.bn1 = Norm2d(6, args.norm, 2)
        self.bn2 = Norm2d(16, args.norm, 4)
        self.bn3 = Norm1d(120, args.norm, 4)
        self.norm= args.norm
        self.conv_list = [self.conv1, self.conv2]

    def forward(self, x):
        FEATURES.content = []
        feature1 = self.conv1(x)
        x = self.pool(F.relu(self.bn1(feature1)))
        feature2 = self.conv2(x)
        x = self.pool(F.relu(self.bn2(feature2)))
        x = x.reshape(x.size(0), -1)
        feature3 = self.fc1(x)
        if self.norm == 'in':
          x = F.relu(feature3)
        else:
          x = F.relu(self.bn3(feature3))
        feature4 = self.fc2(x)
        x = F.relu(feature4)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), [feature1, feature2, feature3, feature4], 0
    
    def distri_norm(self, feature_list, global_stat):
        mean_list = []
        var_list  = []
        for (idx, feature), conv in zip(enumerate(feature_list), self.conv_list):
            mean = feature.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var  = ((feature - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            if global_stat is None:
                grad_weight = torch.autograd.grad(self.mse(mean, torch.zeros(mean.shape).to('cuda')) + self.mse(var, torch.ones(var.shape).to('cuda')), conv.weight, retain_graph=True)
                grad_bias = torch.autograd.grad(self.mse(mean, torch.zeros(mean.shape).to('cuda')) + self.mse(var, torch.ones(var.shape).to('cuda')), conv.bias, retain_graph=True)
            else:
                grad_weight = torch.autograd.grad(self.mse(mean, global_stat[0][idx]) + self.mse(var, global_stat[1][idx]), conv.weight, retain_graph=True)
                grad_bias = torch.autograd.grad(self.mse(mean, global_stat[0][idx]) + self.mse(var, global_stat[1][idx]), conv.bias, retain_graph=True)
  
            if conv.weight.grad is None:
                conv.weight.grad = grad_weight[0]
                conv.bias.grad = grad_bias[0]
            else:
                conv.weight.grad += grad_weight[0]
                conv.bias.grad += grad_bias[0]
            
            mean_list.append(mean.detach())
            var_list.append(var.detach())

        return mean_list, var_list

    def print_grad(self):
        print(self.conv1.weight.grad)

    def print_distri(self, feature_list):
        feature1 = feature_list[0]
        mean1 = feature1.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var1 = ((feature1 - mean1) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
 
        feature2 = feature_list[1]
        mean2 = feature2.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var2 = ((feature2 - mean2) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        print(mean1)
        print(var1)
        print(mean2)
        print(var2)


class CNNMiniImagenet(nn.Module):
    def __init__(self, args):
        super(CNNMiniImagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(10, 20, 3)
        #self.conv4 = nn.Conv2d(32, 32, 3)
        #self.fc1 = nn.Linear(10368, 4096)
        self.fc1 = nn.Linear(7220, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 100)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class modelC(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1) 
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)

        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
