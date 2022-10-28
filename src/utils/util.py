#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import sys

import copy
import heapq
import torch
import numpy as np
from dataset import FemnistDataset, read_data_json, MiniImagenetDataset, HARDataset, read_data_HAR, HADDataset, read_data_HAD
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_extr_noniid, miniimagenet_extr_noniid, mnist_extr_noniid, cifar_Dirichlet_noniid, cifar_unbalanced_noniid

import math
import torch.linalg as linalg
from torch.utils.data import DataLoader, Dataset

from options import args_parser


def get_mal_dataset(dataset, num_mal, num_classes):
    X_list = np.random.choice(len(dataset), num_mal)
    print(X_list)
    Y_true = []
    for i in X_list:
        _, Y = dataset[i]
        Y_true.append(Y)
    Y_mal = []
    for i in range(num_mal):
        allowed_targets = list(range(num_classes))
        allowed_targets.remove(Y_true[i])
        Y_mal.append(np.random.choice(allowed_targets))
    return X_list, Y_mal, Y_true

def get_dataset_femnist(data_dir):
    data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test = read_data_json(data_dir)
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = FemnistDataset(data_x_train, data_y_train, apply_transform)
    dataset_test = FemnistDataset(data_x_test, data_y_test, apply_transform)
    return dataset_train, dataset_test, user_group_train, user_group_test

def get_dataset_HAR(data_dir, num_samples):
    data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test = read_data_HAR(data_dir, num_samples)
    dataset_train = HARDataset(data_x_train, data_y_train)
    dataset_test = HARDataset(data_x_test, data_y_test)
    return dataset_train, dataset_test, user_group_train, user_group_test

def get_dataset_HAD(data_dir, num_samples):
    data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test = read_data_HAD(data_dir, num_samples)
    dataset_train = HADDataset(data_x_train, data_y_train)
    dataset_test = HADDataset(data_x_test, data_y_test)
    return dataset_train, dataset_test, user_group_train, user_group_test

def get_dataset_cifar10_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = '../data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def get_dataset_miniimagenet_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = '../dataset/mini-imagenet/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    train_dataset = MiniImagenetDataset(mode = 'train', root = data_dir,
                                   transform=apply_transform)

    test_dataset = MiniImagenetDataset(mode = 'test', root = data_dir,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = miniimagenet_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def get_dataset(args, shift = False):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if 'cifar' in args.dataset:
        data_dir = '../data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if '100' in args.dataset:
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                        transform=apply_transform)
            '''
            transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
            '''

            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                        transform=apply_transform)
            
            '''
            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
            '''            
        else:
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                        transform=apply_transform)
            '''
            transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
            '''

            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
            
            '''
            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
            '''

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal == 1:
                # Chose uneuqal splits for every user
                partition_sizes = [1.0 / args.num_users for _ in range(args.num_users)]
                user_groups = cifar_Dirichlet_noniid(train_dataset.targets, partition_sizes, alpha=0.5)
            elif args.unequal == 2:
                partition_sizes = [1.0 / _ for _ in range(1, args.num_users)]
                user_groups = cifar_noniid(train_dataset, args.num_users)
                num_per_client = len(user_groups[0]) - 50
                for i in range(0, len(user_groups), 2):
                    size = partition_sizes[i//2]
                    tmp = user_groups[i][:int(num_per_client*size)]
                    user_groups[i] = user_groups[i][int(num_per_client*size):]
                    user_groups[i+1] = np.concatenate([user_groups[i+1], tmp], axis=0)
            elif args.unequal == 3:
                #globally imbalanced partition
                user_groups = cifar_unbalanced_noniid(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fashion_mnist/'

        if shift == False:
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
        else:
            apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4307,), (0.3081,))])
        
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)

        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)

            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups


def handle_boosting(w, ns):
    w_new = []
    for i in range(len(ns)):
        for j in range(int(ns[i])):
            w_new.append(copy.deepcopy(w[i]))
    return w_new


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_median(w):
    """
    Returns the median of the weights
    """
    w_med = copy.deepcopy(w[0])
    for key in w_med.keys():
        w_stack = torch.stack([w[i][key] for i in range(len(w))], dim=0)
        w_med[key] = torch.median(w_stack, dim=0).values
    return w_med

def average_weights_trim_mean(w, belta):
    """
    Returns the trimmed mean of the weights
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_stack = torch.stack([w[i][key] for i in range(len(w))], dim=-1)
        w_list_stack = w_stack.view(-1, w_stack.shape[-1])
        w_avg_list = []
        for w_list in w_list_stack:
            w_list = torch.sort(w_list).values
            w_tm = torch.mean(w_list[int(len(w_list)*belta) : int(len(w_list)*(1-belta)+1)]).item()
            w_avg_list.append(w_tm)
        w_avg_list = torch.Tensor(w_avg_list)
        w_avg[key] = w_avg_list.reshape(w_avg[key].shape)
    return w_avg


def average_weights_ns(w, ns):
    """
    Returns the weighted average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * ns[0]
        for i in range(1, len(w)):
            w_avg[key] += ns[i] * w[i][key]
        w_avg[key] = torch.div(w_avg[key], sum(ns))
    return w_avg

def average_weights_with_masks(w, masks, device):
    '''
    Returns the average of the weights computed with masks.
    '''
    step = 0
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'weight' in key:
            mask = masks[0][step]
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
                mask += masks[i][step]
            w_avg[key] = torch.from_numpy(np.where(mask<1, 0, w_avg[key].cpu().numpy()/mask)).to(device)
            step += 1
        else:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def difference_weights(w_new, w):
    w_diff = copy.deepcopy(w)
    for key in w.keys():
        w_diff[key] = w_new[key] - w[key]
    return w_diff

def cosine_weights(w1, w2):
    cosines = []
    for key in w1.keys():
        cosines.append(torch.cosine_similarity(w1[key].view(-1), w2[key].view(-1), 0))
    return sum(cosines)/len(cosines)

def average_stat(stat_list):
    avg_stat = stat_list[0]
    N = len(stat_list)
    D = len(avg_stat[0])
    for i in range(D):
        for j in range(1, N):
            avg_stat[0][i] += stat_list[j][0][i]
            avg_stat[1][i] += stat_list[j][1][i]
        avg_stat[0][i] /= N
        avg_stat[1][i] /= N
    return avg_stat


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def local_similarities_update(indx_users, local_weights, last_global_weights, global_weights, grad_cosine_similarities):
    diff_global = difference_weights(global_weights, last_global_weights)
    for i in range(len(indx_users)):
        grad_cosine_similarities[indx_users[i]] = cosine_weights(diff_global, difference_weights(local_weights[i], last_global_weights))
    return

def choose_devices(grad_cosine_similarities, m):
    sim = copy.deepcopy(grad_cosine_similarities)
    return list(map(sim.index, heapq.nsmallest(m, sim)))

def choose_devices_greedy(grad_cosine_similarities, m):
    sim = copy.deepcopy(grad_cosine_similarities)
    return list(map(sim.index, heapq.nlargest(m, sim)))     

def choose_devices_less_lazy(grad_cosine_similarities, m, alpha):
    sim = copy.deepcopy(grad_cosine_similarities)
    min_half = list(map(sim.index, heapq.nsmallest(int(len(sim)*alpha), sim)))
    lower_index = np.random.choice(min_half, m, replace=False)
    return lower_index

def choose_devices_less_greedy(grad_cosine_similarities, m, alpha):
    sim = copy.deepcopy(grad_cosine_similarities)
    max_half = list(map(sim.index, heapq.nlargest(int(len(sim)*alpha), sim)))
    higher_index = np.random.choice(max_half, m, replace=False)
    return higher_index


def choose_devices_balance(grad_cosine_similarities, m, alpha):
    sim = copy.deepcopy(grad_cosine_similarities)
    max_half = list(map(sim.index, heapq.nlargest(int(len(sim)/2), sim)))
    min_half = list(map(sim.index, heapq.nsmallest(len(sim) - int(len(sim)/2), sim)))
    higher_index = list(np.random.choice(max_half, int(m * alpha), replace=False))
    lower_index = list(np.random.choice(min_half, m - int(m * alpha), replace=False))
    higher_index.extend(lower_index)
    return higher_index


def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    return mask

def make_mask_ratio(model, ratio):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            temp_tensor = np.ones_like(tensor)
            temp_array = temp_tensor.reshape(-1)
            zero_index = np.random.choice(range(temp_array.size), size = int(temp_array.size * ratio), replace = False)
            temp_array[zero_index] = 0
            temp_tensor = temp_array.reshape(temp_tensor.shape)
            mask[step] = temp_tensor
            step = step + 1
    return mask

# Prune by Percentile module
def prune_by_percentile(model, mask, percent, resample=False, reinit=False,**kwargs):

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1

# Mask the model
def mask_model(model, mask, initial_state_dict):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]

# mix the global_weight_epoch and global_weight
def mix_global_weights(global_weights_last, global_weights_epoch, masks, device):
    step = 0
    global_weights = copy.deepcopy(global_weights_epoch)
    for key in global_weights.keys():
        if 'weight' in key:
            mask = masks[0][step]
            for i in range(1, len(masks)):
                mask += masks[i][step]
            global_weights[key] = torch.from_numpy(np.where(mask<1, global_weights_last[key].cpu(), global_weights_epoch[key].cpu())).to(device)
            step += 1
    return global_weights


def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''
    X, Y = X.cuda(), Y.cuda()
    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)

    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).cpu().float()

def measure_external_covariate(weights, model, datasets):
    def get_features(name):
        def hook(model, input, output):
            if type(input) is tuple:
                activations[name] = copy.deepcopy(input[0].cpu().detach())
            else:
                activations[name] = copy.deepcopy(input.cpu().detach())
        return hook

    stats_list = []
    #features for vgg11 networks
    #names = ["classifier.5"]

    #features for resnet18 networks
    names = ["layer2.0.conv1", "fc"]
    
    #load data

    W_dists = {}
    with torch.no_grad():
        for i in range(len(weights)):
            model.load_state_dict(weights[i])
            
            dataloader = DataLoader(datasets[i], batch_size=500, shuffle=False)
            batch_data, _ = next(iter(dataloader))
            batch_data = batch_data.cuda()

            hooks = []
            activations = {}

            for name, module in model.named_modules():
                if name in names:
                    hook = module.register_forward_hook(get_features(name))
                    hooks.append(hook)
            
            model(batch_data)

            for hook in hooks:
                hook.remove()
            
            stats_list.append(copy.deepcopy(activations))

    for name in names:
        W_dists[name] = []
        for i in range(len(stats_list)):
            for j in range(i+1, len(stats_list)):
                X = stats_list[i][name]
                Y = stats_list[j][name]
                X, Y = X.view(X.shape[0], -1), Y.view(Y.shape[0], -1)
                dist = calculate_2_wasserstein_dist(X, Y)
                W_dists[name].append(dist)
        W_dists[name] = np.mean(W_dists[name])

    return W_dists


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor


if __name__ == '__main__':
    args = args_parser()
    print("Dataset: {}, iid: {}, unequal: {}".format(args.dataset, args.iid, args.unequal))
    train_dataset, test_dataset, user_groups = get_dataset(args)
