"""
Actor class for python written in gevent.
"""

import gevent
import torch
from gevent.queue import Queue
from util import serialize_tensor, deserialize_tensor

class Actor(gevent.Greenlet):

    def __init__(self):
        self.inbox = Queue()
        gevent.Greenlet.__init__(self)

    def receive(self, message):
        raise NotImplemented()

    def _run(self):
        self.running = True
        while self.running:
            message = self.inbox.get()
            self.receive(message)

class ParameterShardActor(Actor):

    def __init__(self, learning_rate, params, random_seed=42):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        
        self.parameters = {name, torch.nn.init.xavier_uniform(tensor) for name, tensor in params}
        self.learning_rate = learning_rate


    def receive(self, message):

        if message.get('request') == 'ParmaterRequest':
            message = {
                    'target': 'client',
                    'sender': 'server',
                    'request': 'ParamaterUpdate'
                    'parmeters': parameters
                    }
            self.inbox.put(message)
            gevent.sleep(0)
        
        if message.get('request') == 'GradientUpdate':
            #get the gradients
            gradients = message.get('gradients')
            for name, param in self.parameters:
                self.parameters[name] -= self.learning_rate * gradients[name]
    
    def generate_parameter_message():
        pass


class SGDClientActor(Actor):

    def __init__(self, model):
        self.model = model


    def receive(self, message):
        if message.get('request') == 'ParameterUpdate':
            self.model.load_state_dict(message.,get('parameters')
            gevent.sleep(0)

        if message.get('request') == 'Train':
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

                message = {
                        'request': 'GradientUpdate'
                        'gradients': gradients
                        }

                self.inbox.put(message)
