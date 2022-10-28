import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference

from models.models_new import MLP, MLP_general, CNNMnist, CNNMnist_norm, CNNMnist_deep, CNNFashion_Mnist, CNNCifar_norm, CNNFeMnist, CNNFeMnist_sim, CNNMiniImagenet
from models.resnet_norm import *
from models.vgg_norm import *
from models.efficientnet_norm import *

from utils.util import get_dataset, get_dataset_femnist, get_dataset_cifar10_extr_noniid, get_dataset_HAR, get_dataset_HAD, get_dataset_mnist_extr_noniid, average_weights, average_weights_with_masks, average_stat, exp_details, make_mask, prune_by_percentile, mask_model, mix_global_weights, get_dataset_miniimagenet_extr_noniid
from utils.util import measure_external_covariate
from utils.helper import Helper as helper

from concurrent.futures.thread import ThreadPoolExecutor

def local_update_for_multithreads(args, user_groups, train_dataset, logger, global_model, global_stat, epoch, local_weights, local_losses, device, idx, i):
 
    if args.dataset == 'HAR' or args.dataset == 'shakespeare' or 'extr_noniid' in args.dataset:
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups_train[idx], logger=logger, dataset_test=test_dataset, idxs_test=user_groups_test[idx])
    else:
        local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger)
                     
    w, loss, stat = local_model.update_weights(
        model=copy.deepcopy(global_model), global_round=epoch, device=device, stat=global_stat)
    # get new model
    #new_model = copy.deepcopy(global_model)
    #new_model.load_state_dict(w)
   
    print('user {}, loss {}'.format(idx, loss))
    local_weights[i] = copy.deepcopy(w)
    local_losses[i]  = copy.deepcopy(loss)
    #stat_list.append(stat)
    

def fedavg_main(args, logger, wandb):
    
    exp_details(args)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    print('loading dataset: {}...\n'.format(args.dataset))
    if args.dataset == 'femnist':
        #data_dir = '/home/js905/code/femnist_data'
        data_dir = '/home/js905/code/leaf/data/femnist/data/'
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_femnist(data_dir)
    elif args.dataset == 'cifar10_extr_noniid':
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_cifar10_extr_noniid(args.num_users, args.nclass, args.nsamples, args.rate_unbalance)
    elif args.dataset == 'miniimagenet_extr_noniid':
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_miniimagenet_extr_noniid(args.num_users, args.nclass, args.nsamples, args.rate_unbalance)
    elif args.dataset == 'mnist_extr_noniid':
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_mnist_extr_noniid(args.num_users, args.nclass, args.nsamples, args.rate_unbalance)
    elif args.dataset == 'HAR':
        data_dir = '../data/UCI HAR Dataset'
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_HAR(data_dir, args.num_samples)
    elif args.dataset == 'HAD':
        data_dir = '../data/USC_HAD'
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_HAD(data_dir, args.num_samples)
    else:
        train_dataset, test_dataset, user_groups = get_dataset(args)
    print('data loaded\n')

    print('building model...\n')

    num_classes = 100 if "100" in args.dataset else 10
    
    # BUILD MODEL
    if args.model == 'cnn':
       # Convolutional neural netork
        if args.dataset == 'mnist' or args.dataset == 'mnist_extr_noniid':
            if args.norm == 'ours' or args.norm == 'no':
                global_model = CNNMnist(args=args)
            else:
                global_model = CNNMnist_norm(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar' or args.dataset == 'cifar10_extr_noniid':
                global_model = CNNCifar_norm(args=args)
        elif args.dataset == 'femnist':
            global_model = CNNFeMnist_sim(args=args)
        elif args.dataset == 'miniimagenet_extr_noniid':
            global_model = CNNMiniImagenet(args=args)

    elif args.model == 'mlp_general':
        global_model = MLP_general(args)
    
    elif args.model == 'efficientnet_b0':
        global_model = efficientnet_b0(num_classes=num_classes, norm=args.norm)
        
    elif args.model == 'efficientnet_v2_s':
        global_model = efficientnet_v2_s(num_classes=num_classes, norm=args.norm)

    elif args.model == 'resnet18':
        global_model = resnet18(pretrained=False, num_classes=num_classes, norm=args.norm, restriction=args.restriction)
    
    elif args.model == 'vgg11':
      if args.norm == 'no':
        global_model = vgg11(norm=args.norm, restriction=args.restriction, num_classes=num_classes, dropout=0.5)
      else:
        global_model = vgg11_nm(norm=args.norm, restriction=args.restriction, num_classes=num_classes, dropout=0.5)
        
    else:
        exit('Error: unrecognized model')
    print('model built\n')

    
    # Set the model to train and send it to device.
    global_model.to(device)
    activations= {}
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    
    test_acc, test_loss, class_pcs, class_rcl = test_inference(args, global_model, test_dataset)    
    wandb.log({"loss": test_loss, "Global model Accuracy": 100*test_acc}, step=0)
    #wandb.log({"Class-{} Precision".format(i): 100*class_pcs[i] for i in range(len(class_pcs))}, step=0)
    #wandb.log({"Class-{} Recall".format(i): 100*class_rcl[i] for i in range(len(class_rcl))}, step=0)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        activations[epoch] = {}
        global_stat = None
        stat_list = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)
        local_weights = [0 for i in idxs_users]
        local_losses  = [0 for i in idxs_users]
        
        #with ThreadPoolExecutor(max_workers=20) as executor:
        for i, idx in enumerate(idxs_users):
            if args.dataset == 'HAR' or args.dataset == 'shakespeare' or 'extr_noniid' in args.dataset:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups_train[idx], c=idx, logger=logger, dataset_test=test_dataset, idxs_test=user_groups_test[idx])
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], c=idx, logger=logger)
                             
            w, loss, stat = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, device=device, stat=global_stat)
            # get new model
            #new_model = copy.deepcopy(global_model)
            #new_model.load_state_dict(w)
           
            print('user {}, loss {}'.format(idx, loss))
            local_weights[i] = copy.deepcopy(w)
            local_losses[i]  = copy.deepcopy(loss)
            activations[epoch][idx] = stat
            #stat_list.append(stat)
        
        global_weights = average_weights(local_weights)
        #global_stat = average_stat(stat_list)

        # update global weights
        global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # print global training loss after every 'i' rounds 
        if (epoch+1) % print_every == 0:
            norm = helper.network_norm(global_model)
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            test_acc, test_loss, class_pcs, class_rcl = test_inference(args, global_model, test_dataset)
            print('Global model Accuracy: {:.2f}% \n'.format(100*test_acc))
            #print('Class Precision: {}% \n'.format([100*acc for acc in class_pcs]))
            #print('Class Recall: {}% \n'.format([100*acc for acc in class_rcl]))
            wandb_norm = norm*np.ones_like(np.mean(np.array(train_loss)))
            wandb.log({"loss": np.mean(np.array(train_loss)), "Global model Accuracy": 100*test_acc, "Norm of weights": wandb_norm}, step=epoch+1)

            if (epoch+1) % 100 == 0 or epoch==0:
                W_dists = measure_external_covariate(local_weights, copy.deepcopy(global_model), train_datasets)
                wandb.log(W_dists, step=epoch+1)

            #wandb.log({"Class-{} Precision".format(i): 100*class_pcs[i] for i in range(len(class_pcs))}, step=epoch+1)
            #wandb.log({"Class-{} Recall".format(i): 100*class_rcl[i] for i in range(len(class_rcl))}, step=epoch+1)
            if (epoch+1) % 500 == 0:
                torch.save(global_model, args.save_dirs['checkpoints']+'model-{}.pth'.format(epoch+1))

    #f_rs_new.close()
    #f_rs_old.close()
    # Test inference after completion of training
    test_acc, test_loss, class_pcs, class_rcl= test_inference(args, global_model, test_dataset)
    torch.save(activations, args.save_dirs['results']+"all_output.dict")

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    wandb.log({"Global model Accuracy": "{:.2f}%".format(100*test_acc)}, step=args.epochs+1)
    torch.save(global_model, args.save_dirs['checkpoints']+'model-{}.pth'.format(args.epochs+1))
