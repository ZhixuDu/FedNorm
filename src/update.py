#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.helper import Helper as helper

from utils.helper import Helper as helper 
 

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, c, logger, dataset_test = None, idxs_test = None):
        self.args = args
        self.logger = logger
        if args.dataset == 'femnist' or args.dataset == 'cifar10_extr_noniid' or args.dataset == 'miniimagenet_extr_noniid' or args.dataset == 'mnist_extr_noniid' or args.dataset == 'HAR' or args.dataset == 'HAD':
            if dataset_test is None or idxs_test is None:
                print('error: femnist and cifar10_extr_noniid need dataset_test and idx_test in LocalUpdate!\n')
            self.trainloader, self.validloader, self.testloader = self.train_val_test_femnist(dataset, list(idxs), dataset_test, list(idxs_test))
        else:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        
        if args.cvt_shift:
        
            #idx is in [0, 99], each client is handling in different regions
            _min = (c%10)*0.1
            _max = min(_min + 0.2, 1.0)
            
            brightness = (_min, _max)         if c%2 == 0 else (0.0, 0.0)
            contrast   = (_min, _max)         if c%3 == 0 else (0.0, 0.0)
            saturation = (_min, _max)         if c%5 == 0 else (0.0, 0.0) 
            hue        = (_min-0.5, _max-0.5) if c%3 == 1 else (0.0, 0.0)
            
            self.covariate_transform = transforms.Compose([transforms.ColorJitter(brightness, contrast, saturation, hue), 
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.covariate_transform = None


    def train_val_test_femnist(self, dataset, idxs, dataset_test, idxs_test):
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):]
        idxs_test = idxs_test

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset_test, idxs_test),
                                batch_size=40, shuffle=True)
        return trainloader, validloader, testloader    


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, device, stat, ):
        EPS = 1e-6
        # Set mode to train model
        model.train()
        epoch_loss = []
        new_stat_mean = None
        new_stat_var = None

        activations = {}

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
            optimizer_norm = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)

        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=3e-4)
            optimizer_norm = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=3e-4)
        
        if 'cnn' not in self.args.model:
            helper.adjust_learning_rate(optimizer, global_round, self.args.lr, decay_every=1000)
        
         
        for iter in range(self.args.local_ep):
            batch_loss = []
            avg_mean_list = None
            avg_var_list = None
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.covariate_transform is not None:
                    images     = self.covariate_transform(images.detach())
                
                model.zero_grad()
                log_probs, feature_list, _ = model(images)
                #log_probs = model(images)
                #if batch_idx is 0:
                #    model.print_distri(feature_list)
                if self.args.norm == 'ours':
                    if iter%5 == 0:
                        #print("################norm next round##################")
                        mean_list, var_list = model.distri_norm(feature_list, stat)
                        '''
                        if avg_mean_list is None or avg_var_list is None:
                            avg_mean_list = mean_list
                            avg_var_list = var_list
                        else:
                            for i in range(len(mean_list)):
                                avg_mean_list[i] = 0.9*avg_mean_list[i] + 0.1*mean_list[i]
                                avg_var_list[i] = 0.9*avg_var_list[i] + 0.1*var_list[i]
                        '''
                        optimizer_norm.step()
                        optimizer_norm.zero_grad()
                        log_probs, feature_list, _ = model(images)
                        #log_probs = model(images)
                
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                '''
                # Freezing Pruned weights by making their gradients Zero
                for name, p in model.named_parameters():
                    if 'weight' in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        grad_tensor = np.where(abs(tensor) < EPS, 0, grad_tensor)
                        p.grad.data = torch.from_numpy(grad_tensor).to(device)
                
                optimizer.step()
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                '''
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            #print("epoch loss: {} \n".format(epoch_loss[-1]))



        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), activations

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            #each client test 100 images at most for running time
            if batch_idx > 1:
                break
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs, _, _ = model(images)
            #outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

           
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            
        accuracy = correct/total 
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.train()
    loss, total, correct = [], 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    class_correct = {i:0.0 for i in range(args.num_classes)}
    class_total = {i:0.0 for i in range(args.num_classes)}
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs, _, _ = model(images)
        #outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss.append(batch_loss.item())
        
        result = helper.get_accuracy(outputs, labels)[0]     
        
        correct += result[0]
        total += result[1]
        
        for i in range(args.num_classes):
            result = helper.get_accuracy(outputs, labels, label=i)[0]
            class_correct[i] += result[0]
            class_total[i]   += result[1].item()

    accuracy = correct/total
    loss = sum(loss)/len(loss)

    return accuracy, loss, [class_correct[i]/class_total[i] for i in range(args.num_classes)], [class_correct[i]/total*10 for i in range(args.num_classes)]
