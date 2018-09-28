import os
import socket
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size, hostname):
    print("I am {} of {} in {}".format(rank, size, hostname))
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def init_processes(rank, size, hostname, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, hostname)


if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    hostname = socket.gethostname()

init_processes(world_rank, world_size, hostname, run, backend='mpi')
