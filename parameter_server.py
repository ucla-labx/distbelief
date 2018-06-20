import gevent
import os
import torch
from utils import Messages, squash_model, ACTION_CODES, CODE_ACTIONS, send_message, init_processes 
import torch.distributed as dist
from models.mnist import Net
from torch.multiprocessing import Process

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)
DEFAULT_LEARNING_RATE = 0.005

class ParameterServer():

    def __init__(self, learning_rate, model, random_seed=42):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        _LOGGER.info("Creating ParameterServer with LR: {}".format(learning_rate))
        self.learning_rate = learning_rate
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(squash_model(model).numel() + 1)

    def receive(self, message, parameter):
        print("Processing message: {}".format(message))
        if message == 'ParameterRequest':
            send_message('ParameterUpdate', self.parameters, dst=1)    

        elif message == 'GradientUpdate':
            self.parameters -= self.learning_rate * gradient

    def run(self):
        _LOGGER.info("Parameter Server Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for data")
            req = dist.recv(tensor=self.m_parameter)
            _LOGGER.info("Got message")
            req.wait()
            self.receive(ACTION_CODES[self.m_parameter[0].item()], self.m_parameter[1:])

def init_server(rank, size):
    model = Net()
    server = ParameterServer(learning_rate=DEFAULT_LEARNING_RATE, model=model)
    server.run()

if __name__ == "__main__":
     p = Process(target = init_processes, args=(0, 2, init_server))
     p.start()
     p.join()
