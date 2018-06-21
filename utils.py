from enum import Enum
import logging
import os
import torch
import torch.distributed as dist

ACTION_CODES = {
    0: 'ParameterRequest',
    1: 'GradientUpdate',
    2: 'ParameterUpdate',
    3: 'Train',
}
CODE_ACTIONS = {
    'ParameterRequest': 0,
    'GradientUpdate': 1,
    'ParameterUpdate': 2,
    'Train': 3,
}

DEFAULT_LEARNING_RATE = 0.01

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


class Messages(Enum):
    ParameterRequest = 'parameter_request'
    GradientUpdate = 'gradient_update'
    ParameterUpdate = 'paramater_update'
    Train = 'train'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

def squash_model(model, grads=False):
    m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]

def set_params(model, parameter_update):
    current_index = 0
    for parameter in list(model.parameters()):
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data = parameter_update[current_index:current_index+numel].view(size)
        current_index += numel

def send_message(message, payload, dst=0):
    m_parameter = torch.Tensor([CODE_ACTIONS[message]])
    m_parameter = torch.cat((m_parameter, payload))
    dist.send(tensor=m_parameter, dst=dst)
