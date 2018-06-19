import gevent
from .actor import ModelActor
from utils import Messages


class SGDClientActor(ModelActor):

    def __init__(self, learning_rate, model, rank=0, size=0):
        self.learning_rate = learning_rate
        super().__init__(model)
        self.parameters = None # need to request parameters from parameter_shard actor
        self.request_frequency = 5 # request every 5 self.runs (including 0)
        self.run_count = 0

    def receive(self, message, parameter):

        if message == 'ParamaterUpdate':
            print("Got message {}".format(parameter))
            self.set_params(parameter)
            print("working")
            gevent.sleep(0)
        

    def run():
            self.model.train()
            while True:
                # pull params synchronously for now (TODO: figure out how to express SGD client with async as an actor) 
                self.send_message('ParameterRequest', torch.zeros(self.squash_mode().size()))
                print("sent param request method")


                # if args.cuda:
                    # data, target = data.cuda(), target.cuda()
                # data, target = Variable(data), Variable(target)
                # optimizer.zero_grad()
                # output = model(data)
                # loss = F.nll_loss(output, target)
                # loss.backward()

                gradients = self.squash_model(grads=True)
                self.send_message('GraidentUpdate', gradients)

                # and this is our internal gradient update
                self.set_params(self.squash_model() - self.learning_rete * gradients)
