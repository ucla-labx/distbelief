"""
This class listens for a ParameterUpdate from the parameter server and then updates the model accordingly
"""
import logging
import torch
import torch.distributed as dist
from utils import send_message, ravel_model_params, unravel_model_params, MessageCode, MessageListener
from model import Net

_LOGGER = logging.getLogger(__name__)

class DownpourSGD(MessageListener):

    def receive(self,sender, message_code, parameter):
        _LOGGER.info("Processing message: {}".format(message_code.name))
        if message_code == MessageCode.ParameterUpdate:
            unravel_model_params(self.model, parameter)

def init_sgd(model):
    server = DownpourSGD(model=model)
    server.run()

