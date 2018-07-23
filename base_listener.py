from utils import MessageCode, ravel_model_params
import torch
import torch.distributed as dist
import logging 

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

# Base class that runs and listens for messages. Classes should inherit from this and implement the receive method.
class MessageListener():
    def __init__(self, model):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        self.model = model
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)

    def receive(self, sender, message_code, parameter):
        raise NotImplementedError('Classes should inherit from MessageListener and implement this method.')

    def run(self):
        _LOGGER.info("Started Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for message...")
            dist.recv(tensor=self.m_parameter)
            _LOGGER.info("Sucessfully received message")
            self.receive(int(self.m_parameter[0].item()),
                         MessageCode(self.m_parameter[1].item()),
                         self.m_parameter[2:])
