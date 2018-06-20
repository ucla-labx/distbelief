import gevent
from utils import Messages, send_message, squash_model, set_params
import torch
import time
import logging
from models.mnist import Net
import torch.distributed as dist

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

DEFAULT_LEARNING_RATE = 0.005



class DownpourSGD():

    def __init__(self, learning_rate, model):
        self.learning_rate = learning_rate
        self.request_frequency = 5 # request every 5 self.runs (including 0)
        self.model = model
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(squash_model(model).numel() + 1)

    def receive(self, message, parameter):
        print("Processing message: {}".format(message))
        if message == 'ParamaterUpdate':
            set_params(self.model, parameter)

    def run(self):
        _LOGGER.info("DownpourSGD Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for data")
            dist.recv(tensor=self.m_parameter)
            _LOGGER.info("Got message")
            self.receive(ACTION_CODES[self.m_parameter[0].item()], self.m_parameter[1:])

def init_sgd():
    model = Net()
    server = DownpourSGD(learning_rate=DEFAULT_LEARNING_RATE, model=model)
    server.run()

