import os
import logging 
import argparse
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models import LeNet, AlexNet
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch.optim as optim
from distbelief.optim import DownpourSGD
from distbelief.server import ParameterServer


def get_dataset(args, transform):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the dataset
    """
    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader

def main(args):

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    trainloader, testloader = get_dataset(args, transform)
    net = LeNet()

    if args.distributed:
        optimizer = DownpourSGD(net.parameters(), lr=args.lr, freq=args.freq, model=net)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-3)

    # train
    net.train()
    if args.cuda:
        net = net.cuda()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.log_interval == 0 and i > 0:    # print every n mini-batches
                print('Epoch: %d, Iteration: %5d loss: %.3f' % (epoch, i, running_loss / args.log_interval))
                running_loss = 0.0

        val_loss = evaluate(net, testloader, args)
        scheduler.step(val_loss)

    print('Finished Training')


def evaluate(net, testloader, args):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            running_loss += F.cross_entropy(outputs, labels).item()

    print('Loss: {:.3f}'.format(running_loss / len(testloader)))
    print('Accuracy: {:.3f}'.format(accuracy_score(predicted, labels)))
    print(classification_report(predicted, labels, target_names=classes))
    
    return running_loss

def init_server(args):
    model = LeNet()
    server = ParameterServer(model=model)
    server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--freq', type=int, default=10, metavar='N', help='how often to send/pull grads (default: 10)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--distributed', action='store_true', default=False, help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--rank', type=int, metavar='N', help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=3, metavar='N', help='size of the world')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    args = parser.parse_args()
    print(args)

    if args.distributed:
        """ Initialize the distributed environment.
        Server and clients must call this as an entry point.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('tcp', rank=args.rank, world_size=args.world_size)
        if args.server:
            init_server(args)
    main(args)
