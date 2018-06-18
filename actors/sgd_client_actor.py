import gevent
from .actor import ModelActor
from utils import Messages, ModelMessage


class SGDClientActor(ModelActor):

    def __init__(self, learning_rate, model, server_actor):
        self.learning_rate = learning_rate
        super().__init__(model)
        self.parameters = None # need to request parameters from parameter_shard actor
        self.request_frequency = 5 # request every 5 self.runs (including 0)
        self.run_count = 0
        # for now, we'll initialize each client with the server they should communicate with (and similarly for servers)
        # we probably don't want to do this long-term, and should switch to some sort of polling-based approach. This will require monkeypatching.
        self.server_actor = server_actor

    def _run(self):
        assert not self.running
        super()._run()
        assert self.running
        print('in sgd client run')
        if self.run_count % self.request_frequency == 0:
            # request parameters from client
            print('requesting parameters from client')
            self.send_message(message_type=Messages.ParameterRequest)
            while self.inbox.empty():
                # waiting for parameters
                print('client waiting for parameters...')
                gevent.sleep(3)
            assert not self.inbox.empty()
            print('sgd client actor inbox is not empty')
            # TODO - we have the parameters now - process the message and set self.parameters to not None.
        else:
            # this code path assumes that we don't need to request parameters
            # hence we have the parameters, so we should assume that as
            pass
        gevent.sleep(0)

    def send_message(self, message_type):
        if message_type == Messages.ParameterRequest:
            # put parameter request into server's inbox
            message = ModelMessage(message_type=message_type)
            self.server_actor.inbox.put(message.message_dict)
        else:
            pass # not implemented yet.



    def receive(self, message, parameter):

        if message == 'ParamaterUpdate':
            self.set_params(parameter)
            gevent.sleep(0)
        
        if message == 'Train':
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                # pull params synchronously for now (TODO: figure out how to express SGD client with async as an actor)
                self.send_message('ParameterRequest', torch.zeros(self.squash_mode().size()))
                gevent.sleep(0) #give up until we get our parameter update

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
