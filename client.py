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
from shared import init_processes




class Client:
	def __init__(self, id, world_size):
		self.process = Process(target=init_processes, args=(id, world_size, self.run_client))

	def start_process(self):
		self.process.start()

	def join_process(self):
		self.process.join()

	def run_client(self, client_rank, world_size):
		print('Hey! Im running')
		pass

if __name__ == '__main__':
	# sys.argv[1] = total number of running processes (include server)
	# sysa.argv[2] - id of this client (should be between 1 and sys.argv[1] -1)

	if len(sys.argv) != 3:
		print('USAGE: python3 client.py NUM_PROCESSES PROCESS_ID')
		exit(1)

	size = int(sys.argv[1])
	num = int(sys.argv[2])
	# init this client
	client = Client(num, size)
	client.start_process()
	# p = Process(target=init_processes, args=(num, size, run_client))
	# p.start()
	# p.join()