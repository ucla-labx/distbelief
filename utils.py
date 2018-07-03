from enum import Enum
import logging
import os
import torch
import torch.distributed as dist

DEFAULT_LEARNING_RATE = 0.3

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

class MessageCode(Enum):
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2

def ravel_model_params(model, grads=False):
    m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]

def unravel_model_params(model, parameter_update):
    current_index = 0
    for parameter in list(model.parameters()):
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data = parameter_update[current_index:current_index+numel].view(size)
        current_index += numel

def send_message(message_code, payload, dst=0):
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
    m_parameter = torch.cat((m_parameter, payload))
    dist.send(tensor=m_parameter, dst=dst)


class MessageListener()

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
