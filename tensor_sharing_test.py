import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

def send_tensor(rank, size):
    print('Rank {}: sleeping for 5 seconds'.format(dist.get_rank()))
    time.sleep(5)
    test = torch.ones(3, 3)
    print('Rank {}:'.format(dist.get_rank()), test)
    dist.send(test, dst=0)

def receive_tensor(rank, size):
    test = torch.zeros(3, 3)
    print('Rank {}:'.format(dist.get_rank()), test)
    dist.recv(test, src=1)
    print('Rank {}:'.format(dist.get_rank()), test)

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '192.168.1.16'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



if __name__ == "__main__":
    size = 2
    processes = []
    if sys.argv[1] == 'SERVER':
        p = Process(target=init_processes, args=(0, size, receive_tensor))
    else:
        p = Process(target=init_processes, args=(1, size, send_tensor))

    p.start()
    processes.append(p)

    for p in processes:
        p.join()

