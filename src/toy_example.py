#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset, average_weights_ns
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, MLPMnist_BN, CNNMnist_BN
from sampling import mnist_noniid
from update import DatasetSplit


if __name__ == '__main__':
    args = args_parser()
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args, False)
    train_dataset_shift, test_dataset_shift, _ = get_dataset(args, False)

    user_groups = mnist_noniid(train_dataset, args.num_users)

    train_dataset1 = DatasetSplit(train_dataset, user_groups[0])
    train_dataset2 = DatasetSplit(train_dataset, user_groups[1])

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'cnn_bn':
        if args.dataset == 'mnist':
            global_model = CNNMnist_BN(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    init_global_model = copy.deepcopy(global_model)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader1 = DataLoader(train_dataset1, batch_size=64, shuffle=True)
    trainloader2 = DataLoader(train_dataset2, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)
    epoch_loss = []

    # no shift training
    print("Client1 training")
    for epoch in tqdm(range(args.epochs)):
        #global_model.print_bn()
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _, _ = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader1.dataset),
                    100. * batch_idx / len(trainloader1), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
    
    global_model.print_bn()


    global_model.eval()
    for batch_idx, (images, labels) in enumerate(trainloader1):
        images, labels = images.to(device), labels.to(device)
        outputs, feature, feature_bn = global_model(images)
        for i in range(len(feature)):
            feature[i] = feature[i].cpu().detach().numpy().reshape(-1,1)
            feature_bn[i] = feature_bn[i].cpu().detach().numpy().reshape(-1,1)


        #feature = feature.cpu().detach().numpy().reshape(-1)
        #np.save("local_no_shift_feature.npy", feature)
        #plt.figure()
        #plt.hist(feature, bins=20)
        #plt.title("local no shift feature")
        #plt.savefig("local_no_shift_feature.png")
        #feature_bn = feature_bn.cpu().detach().numpy().reshape(-1)
        #np.save("local_no_shift_feature_bn.npy", feature_bn)
        #plt.figure()
        #plt.hist(feature_bn, bins=20)
        #plt.title("local no shift feature bn")
        #plt.savefig("local_no_shift_feature_bn.png")
        break

    # test
    global_model.eval()
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Loss: {:.2f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(100*test_acc))

    # shift training


    print("Client2 training")
    global_model_shift = copy.deepcopy(init_global_model)
    global_model_shift.train()
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model_shift.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model_shift.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader2):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _, _ = global_model_shift(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader2.dataset),
                    100. * batch_idx / len(trainloader2), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    global_model_shift.print_bn()


    global_model_shift.eval()
    for batch_idx, (images, labels) in enumerate(trainloader2):
        images, labels = images.to(device), labels.to(device)
        outputs, feature, feature_bn = global_model_shift(images)
        #feature = feature.cpu().detach().numpy().reshape(-1)
        #plt.figure()
        #plt.hist(feature, bins=20)
        #plt.title("local shift feature")
        #plt.savefig("local_shift_feature.png")
        #feature_bn = feature_bn.cpu().detach().numpy().reshape(-1)
        #plt.figure()
        #plt.hist(feature_bn, bins=20)
        #plt.title("local shift feature bn")
        #plt.savefig("local_shift_feature_bn.png")
        for i in range(len(feature)):
            feature[i] = feature[i].cpu().detach().numpy().reshape(-1,1)

            feature_bn[i] = feature_bn[i].cpu().detach().numpy().reshape(-1,1)



        #np.save("local_shift_feature.npy", feature)
        #np.save("local_shift_feature_bn.npy", feature_bn)
        break
    '''
   
    # test
    global_model_shift.eval()
    test_acc, test_loss = test_inference(args, global_model, test_dataset_shift)
    print('Test on', len(test_dataset_shift), 'samples')
    print("Test Loss: {:.2f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(100*test_acc))


    print("aggregating....")
    print("###################")
    global_model.print_bn()
    print("###################")
    global_model_shift.print_bn()
    print("###################")
    agg_model_weight = average_weights_ns([global_model.state_dict(), global_model_shift.state_dict()], [9, 1])
    agg_model = copy.deepcopy(global_model)
    agg_model.load_state_dict(agg_model_weight)
    print("###################")
    agg_model.print_bn()
    print("###################")
    # test on unshift data
    agg_model.eval()
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        outputs, feature, feature_bn = agg_model(images)
        #feature = feature.cpu().detach().numpy().reshape(-1)
        #plt.figure()
        #plt.hist(feature, bins=20)
        #plt.title("global no shift feature")
        #plt.savefig("global_no_shift_feature.png")
        #feature_bn = feature_bn.cpu().detach().numpy().reshape(-1)
        #plt.figure()
        #plt.hist(feature_bn, bins=20)
        #plt.title("global no shift feature bn")
        #plt.savefig("global_no_shift_feature_bn.png")
        for i in range(len(feature)):
            feature[i] = feature[i].cpu().detach().numpy().reshape(-1,1)

            feature_bn[i] = feature_bn[i].cpu().detach().numpy().reshape(-1,1)



        np.save("global_no_shift_feature.npy", feature)
        np.save("global_no_shift_feature_bn.npy", feature_bn)

        break

    agg_model.eval()
    test_acc, test_loss = test_inference(args, agg_model, test_dataset)
    print('Test on', len(test_dataset), 'unshift samples')
    print("Test Loss: {:.2f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(100*test_acc))

    # test on shift data
    agg_model.eval()
    for batch_idx, (images, labels) in enumerate(trainloader_shift):
        images, labels = images.to(device), labels.to(device)
        outputs, feature, feature_bn = agg_model(images)
        #feature = feature.cpu().detach().numpy().reshape(-1)
        #plt.figure()
        #plt.hist(feature, bins=20)
        #plt.title("global shift feature")
        #plt.savefig("global_shift_feature.png")
        #feature_bn = feature_bn.cpu().detach().numpy().reshape(-1)
        #plt.figure()
        #plt.hist(feature_bn, bins=20)
        #plt.title("global shift feature bn")
        #plt.savefig("global_shift_feature_bn.png")
        for i in range(len(feature)):
            feature[i] = feature[i].cpu().detach().numpy().reshape(-1,1)

            feature_bn[i] = feature_bn[i].cpu().detach().numpy().reshape(-1,1)



        np.save("global_shift_feature.npy", feature)
        np.save("global_shift_feature_bn.npy", feature_bn)

        break

    agg_model.eval()
    test_acc, test_loss = test_inference(args, agg_model, test_dataset_shift)
    print('Test on', len(test_dataset), 'shift samples')
    print("Test Loss: {:.2f}".format(test_loss))
    print("Test Accuracy: {:.2f}%".format(100*test_acc))

    '''





    '''
    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))
    '''
    '''
    # testing
    global_model.eval()
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
    '''
