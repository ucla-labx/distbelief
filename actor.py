"""
Actor class for python written in gevent.
"""

import gevent
import torch
from gevent.queue import Queue
from util import serialize_tensor, deserialize_tensor
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

class ModelActor(gevent.Greenlet):

    def __init__(self, model):
        self.model = model
        gevent.Greenlet.__init__(self)

    def receive(self, message, payload):
        raise NotImplemented()

    def _run(self):
        self.running = True
        m_parameter = torch.zeros(self.squash_model().size())
        while self.running:
            dist.recv(tensor=m_parameter)
            self.receive(ACTION_CODES[m_parameter[0]], m_parameter[1:])

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
        dist.send(tensor=m_parameter)

class ParameterShardActor(Actor):

    def __init__(self, learning_rate, model, random_seed=42):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        
        self.parameters = torch.zeros(self.squash_model().size())
        self.learning_rate = learning_rate


    def receive(self, message, gradient):
        message_code = m_parameter[0]

        if message == 'ParmaterRequest':
            self.send_m_parameter('ParameterUpdate', self.parameters)
        
        if message == 'GradientUpdate':
            #get the gradients
            self.parameters -= self.learning_rate * gradient

class SGDClientActor(Actor):

    def __init__(self, learning_rate, model):
        self.model = model
        self.learning_rate = learning_rate
        
    def receive(self, message, parameter):

        if message == 'ParmaterUpdate':
            self.set_params(parameter)
            gevent.sleep(0)
        
        if message == 'Train':
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                # pull params
                self.send_message('ParameterRequest', torch.zeros(self.squash_mode().size()))

                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                gradients = self.squash_model(grads=True)
                self.send_message('GraidentUpdate', gradients)

                # and this is our internal gradient update
                self.set_params(self.squash_model() - self.learning_rete * gradients)
