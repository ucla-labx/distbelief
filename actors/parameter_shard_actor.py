import gevent
import torch
from .actor import ModelActor
from utils import Messages

class ParameterShardActor(ModelActor):

    def __init__(self, learning_rate, model, clients = [], random_seed=42):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        self.learning_rate = learning_rate
        self.parameters = list(model.parameters())
        self.clients = clients
        super().__init__(model)

    def receive(self, message=None, gradient=None):
        message = self.inbox.get()
        message_type = message['message_type']
        if message_type == Messages.ParameterRequest:
            print('in server: got parameter request message, need to send parameters')
            # self.send_message('ParameterUpdate', self.parameters)    
        elif message_type == 'GradientUpdate':
            #get the gradients
            # self.parameters -= self.learning_rate * gradient
            pass

    def _run(self):
        assert not self.running
        super()._run()
        assert self.running
        print('in parameter shard running')
        # poll the inbox for a parameter request or gradient computation request
        while self.inbox.empty():
            print('server is waiting for parameter request')
            gevent.sleep(0)
        self.receive()


    def add_client(self, client):
        self.clients.append(client)

