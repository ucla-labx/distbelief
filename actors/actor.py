"""
Actor class for python written in gevent.
"""

import gevent
import torch
from gevent.queue import Queue
import torch.distributed as dist
from enum import Enum

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


class ModelActor(gevent.Greenlet):
    def __init__(self, model):
        self.model = model
        self.inbox = Queue()
        self.running = False
        gevent.Greenlet.__init__(self)

    def receive(self, message, payload):
        raise NotImplementedError('Classes should inheret this method and override.')

    def _run_demo_path(self):
        # just demo code
            run_count = 0
            while self.running:
                message = self.inbox.get()
                self.receive(message)
                gevent.sleep(0)
                run_count +=1 
                if run_count == 10:
                    return True # exit

    def _run(self):
        self.running = True
        if not self.model:
            self._run_demo_path()
        # else:
        #     m_parameter = torch.zeros(self.squash_model().size())
        #     while self.running:
        #         dist.recv(tensor=m_parameter)
        #         self.receive(ACTION_CODES[m_parameter[0]], m_parameter[1:])

    def squash_model(self, grads=False):
        m_parameter = torch.Tensor([0])
        for parameter in list(self.model.parameters()):
            if grads:
                m_parameter = torch.cat(m_parameter, parameter.grad.view(-1))
            else:
                m_parameter = torch.cat(m_parameter, parameter.data.view(-1))
        return m_parameter[1:]

    def set_params(self, parameter_update):
        current_index = 0
        for parameter in list(model.parameters()):
            numel = parameter.data.numel()
            size = parameter.data.size()
            paramater.data = paramater_update[current_index:current_index+numel].view(size)
            current_index += numel

    def send_message(self, message, payload):
        m_parameter = torch.Tensor([CODE_ACTIONS[message]])
        m_parameter = torch.cat(m_parameter, payload)
        dist.isend(tensor=m_parameter)

