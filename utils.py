from enum import Enum
import logging
import os
import torch
import torch.distributed as dist

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment.
    Server and clients must call this as an entry point.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn()

class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3

def ravel_model_params(model, grads=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]

def unravel_model_params(model, parameter_update):
    """
    Assigns parameter_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in parameter_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0 # keep track of where to read from parameter_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data = parameter_update[current_index:current_index+numel].view(size)
        current_index += numel

def send_message(message_code, payload, dst=0):
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
    """
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
    m_parameter = torch.cat((m_parameter, payload))
    dist.isend(tensor=m_parameter, dst=dst)


class MessageListener():
    def __init__(self, model):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        self.model = model
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)

    def receive(self, sender, message_code, parameter):
        raise NotImplementedError()

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
