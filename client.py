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
import threading
from mnist import Net



class Client:
	def __init__(self, id, world_size, model_class):
		print(model_class)
		self.model = model_class()
		self.process = Process(target=init_processes, args=(id, world_size, self.run_client))


	def start_process(self):
		self.process.start()

	def join_process(self):
		self.process.join()

	def make_requests(self):
		print('make requests function {}'.format(threading.get_ident()))

	def training_thread(self):
		print('training thread: {}'.format(threading.get_ident()))
		print("model is {}".format(self.model))

	def run_client(self, client_rank, world_size):
		print('Hey! Im running')
		requests_thread = threading.Thread(target=self.make_requests)
		train_thread = threading.Thread(target=self.training_thread)
		threads = [requests_thread, train_thread]
		for thread in threads:
			thread.start()

		for thread in threads:
			thread.join()

		print('All threads completed.')
		print('Main thread {}'.format(threading.get_ident()))



if __name__ == '__main__':
	# sys.argv[1] = total number of running processes (include server)
	# sys.argv[2] - id of this client (should be between 1 and sys.argv[1] -1)

	if len(sys.argv) != 3:
		print('USAGE: python3 client.py NUM_PROCESSES PROCESS_ID')
		exit(1)

	size = int(sys.argv[1])
	num = int(sys.argv[2])
	# init this client
	client = Client(num, size, Net)
	client.start_process()
	# p = Process(target=init_processes, args=(num, size, run_client))
	# p.start()
	# p.join()