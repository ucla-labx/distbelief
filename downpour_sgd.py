"""
This class listens for a ParameterUpdate from the parameter server and then updates the model accordingly
"""

from utils import send_message, ravel_model_params, unravel_model_params, MessageCode
import torch
import logging
from models.mnist import Net
import torch.distributed as dist

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(process)d %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

class DownpourSGD():
    def __init__(self, model):
        self.model = model
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)

    def receive(self, message_code, parameter):
        _LOGGER.info("Processing message: {}".format(message_code.name))
        if message_code == MessageCode.ParameterUpdate:
            unravel_model_params(self.model, parameter)

    def run(self):
        _LOGGER.info("DownpourSGD Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for data")
            dist.recv(tensor=self.m_parameter)
            _LOGGER.info("Got message")
            self.receive(MessageCode(self.m_parameter[1].item()), self.m_parameter[2:])

def init_sgd(model):
    server = DownpourSGD(model=model)
    server.run()

