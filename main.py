"""
gevent actor test
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

from downpour_sgd import DownpourSGD, init_sgd
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from utils import ravel_model_params, unravel_model_params, init_processes, send_message, DEFAULT_LEARNING_RATE, MessageCode
from experimental import evaluate

from model import Net

import threading
import pandas as pd

log_dataframe = []

def train(args, model, device, train_loader, test_loader, epoch):
    model.train() # pytorch indication that the model is in training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        # send gradient request every 10 iterations
        if batch_idx % 10 == 0:
            send_message(MessageCode.ParameterRequest, torch.zeros(ravel_model_params(model).size()))

        data, target = data.to(device), target.to(device)
        output = model(data)
        model.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        gradients = ravel_model_params(model, grads=True)
        send_message(MessageCode.GradientUpdate, gradients) # send gradients to the server

        # and this is our internal gradient update
        unravel_model_params(model, ravel_model_params(model) - DEFAULT_LEARNING_RATE * gradients)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        test(model, device, test_loader, log_dataframe)

def test(model, device, test_loader, dataframe):
    model.eval() # pytorch indication that the model is in testing mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

    df_line = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
    dataframe.append(df_line)
    

def main(*args, **kwargs):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)


    model.share_memory()
    # this sets the initial model parameters
    send_message(MessageCode.ParameterUpdate, ravel_model_params(model))
    # start the  training thread
    update_thread = threading.Thread(target=init_sgd, args=(model,))
    update_thread.start()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, test_loader, epoch)

    if dist.get_rank() == 1:
        evaluate(log_dataframe)

if __name__ == "__main__":
    init_processes(1, 3, main)
