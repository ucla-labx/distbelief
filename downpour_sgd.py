"""
This class listens for a ParameterUpdate from the parameter server and then updates the model accordingly
"""
from utils import send_message, ravel_model_params, unravel_model_params, MessageCode, MessageListener
import torch
import logging
from models.mnist import Net
import torch.distributed as dist

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(process)d %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

class DownpourSGD(MessageListener):

    def receive(self,sender, message_code, parameter):
        """receive

        :param message_code:
        :param parameter:
        """
        _LOGGER.info("Processing message: {}".format(message_code.name))
        if message_code == MessageCode.ParameterUpdate:
            unravel_model_params(self.model, parameter)

def init_sgd(model):
    server = DownpourSGD(model=model)
    server.run()

