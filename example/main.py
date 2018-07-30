import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from distbelief.optim import DownpourSGD
import threading
import argparse


class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # hack for MNIST vs CIFAR
        self.fc1 = nn.Linear(16*5*5, 120) if num_channels == 3 else nn.Linear(16*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # hack for CIFAR vs MNIST
        x = x.view(-1, 16* 5* 5) if self.num_channels == 3 else x.view(-1, 16*16)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dataset(dataset_name, transform, batch_size=64):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the dataset
    """
    if dataset_name == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to train with (CIFAR or MNIST)')

    args = parser.parse_args()

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    trainloader, testloader = get_dataset(args.dataset, transform)

    net = Net(num_channels=1 if args.dataset=='MNIST' else 3)

    criterion = nn.CrossEntropyLoss()
    print('initializing downpour listener')
    downpour_optim = optim.SGD(net.parameters(), lr=args.lr)
    optimizer = DownpourSGD(net.parameters(), lr=args.lr, freq=10, model=net, internal_optim=downpour_optim)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.0)

    net.train()
    num_print = 20
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % num_print == 0:    # print every n mini-batches
                print('Epoch: %d, Iteration: %5d loss: %.3f' % (epoch, i, running_loss / num_print))
                running_loss = 0.0
        evaluate(net, testloader, args.dataset)

    # join the listener
    # optimizer.stop_listening()
    print('Finished Training')


def evaluate(net, testloader, dataset_name):
    if dataset_name == 'MNIST':
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    else:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = [0. for _ in range(10)]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
