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

class Actor(gevent.Greenlet):

    def __init__(self, model):
        self.model = model
        gevent.Greenlet.__init__(self)

    def receive(self, m_parameter):
        raise NotImplemented()

    def _run(self):
        self.running = True
        while self.running:
            dist.recv(tensor=self.m_parameter)
            self.receive(m_parameter)

    def squash_params(self,):
        m_parameter = torch.Tensor([0])

        def join_tensors(x, y):
            pass
            
        for parameter in list(model.parameters()):
            m_prameter = torch.cat(m_parameter, parameter.data.view(-1))




        
    def set_params(self, ):

class ParameterShardActor(Actor):

    def __init__(self, learning_rate, model, random_seed=42):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        
        self.m_parameters = {name, torch.nn.init.xavier_uniform(tensor) for name, tensor in params}
        self.learning_rate = learning_rate


    def receive(self, m_parameter):
        m_parameter = torch.zeros(self.parameters)
        dist.recv(tensor=m_parameter)
        message_code = m_parameter[0].item()

        if ACTION_CODES[message_code]== 'ParmaterRequest':
            self.m_parameters[0] = CODE_ACTIONS['ParameterUpdate']
            dist.send(self.parameters)
        
        if ACTION_CODES[message_code] == 'GradientUpdate':
            #get the gradients
            self.m_parameters -= self.learning_rate *m_parameter

class SGDClientActor(Actor):

    def __init__(self, model):
        self.model = model
        
        self.m_parameters = {name, torch.nn.init.xavier_uniform(tensor) for name, tensor in params}
        self.learning_rate = learning_rate


    def receive(self, m_parameter):
        m_parameter = torch.zeros(self.squash_params())
        dist.recv(tensor=m_parameter)
        message_code = m_parameter[0].item()

        if ACTION_CODES[message_code]== 'ParmaterUpdate':
            self.set_params(m_parameter)
            gevent.sleep(0)
        
        if ACTION_CODES[message_code] == 'Train':
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                #get gradients programatically
                gradients = squash_params()
                dist.send(gradients)
