"""
gevent actor test
"""
import gevent
from gevent.queue import Queue
from gevent import Greenlet
from models.mnist import Net
from parameter_server import ParameterServer
from train import SGDClient
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

DEFAULT_LEARNING_RATE = 0.005

def train_dist(rank, size):
    """ Distributed function to be implemented later. """
    model=Net()
    if rank == 0:
        p = ParameterServer(learning_rate=DEFAULT_LEARNING_RATE, model=model, rank=rank, size=size)

    else:
        p = SGDClient(learning_rate=DEFAULT_LEARNING_RATE, model=model, rank=rank, size=size)
    p.run()

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, train_dist))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

