import gevent
import torch
from utils import Messages

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

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



class ParameterServer():

    def __init__(self, learning_rate, model, random_seed=42, rank=0, size=0):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        self.learning_rate = learning_rate
        self.parameters = list(model.parameters())
        print("Creating param server")
        super().__init__(model)

    def receive(self, message=None, gradient=None):
        message = self.inbox.get()
        message_type = message['message_type']
        if message_type == Messages.ParameterRequest:
            print('in server: got parameter request message, need to send parameters')
            self.send_message('ParameterUpdate', self.parameters, dst=1)    

        elif message_type == 'GradientUpdate':
            self.parameters -= self.learning_rate * gradient

    def run(self):
        _LOGGER.info("Running!")
        self.running = True
        if not self.model:
            raise ValueError("Need a model!")
        else:
            _LOGGER.info("Setting m_parameter")
            m_parameter = torch.zeros(self.squash_model().size())

            while self.running:
                dist.recv(tensor=m_parameter)
                self.receive(ACTION_CODES[m_parameter[0]], m_parameter[1:])

    def squash_model(self, grads=False):
        m_parameter = torch.Tensor([0])
        for parameter in list(self.model.parameters()):
            if grads:
                m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
            else:
                m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
        return m_parameter[1:]

    def set_params(self, parameter_update):
        current_index = 0
        for parameter in list(model.parameters()):
            numel = parameter.data.numel()
            size = parameter.data.size()
            paramater.data = paramater_update[current_index:current_index+numel].view(size)
            current_index += numel

    def send_message(self, message, payload, dst=0):
        m_parameter = torch.Tensor([CODE_ACTIONS[message]])
        m_parameter = torch.cat((m_parameter, payload))
        dist.isend(tensor=m_parameter, dst=dst)


