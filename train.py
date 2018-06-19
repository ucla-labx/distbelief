import gevent
from utils import Messages, send_message, squash_model, set_params
import torch
import time


class SGDClient():

    def __init__(self, learning_rate, model, rank=0, size=0):
        self.learning_rate = learning_rate
        self.request_frequency = 5 # request every 5 self.runs (including 0)
        self.run_count = 0
        self.model = model

    def receive(self, message, parameter):

        if message == 'ParamaterUpdate':
            print("Got message {}".format(parameter))
            self.set_params(parameter)
            print("working")
            gevent.sleep(0)

    def run(self):
        _LOGGER.info("Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for data")
            dist.recv(tensor=self.m_parameter)
            _LOGGER.info("Got message")
            self.receive(ACTION_CODES[self.m_parameter[0].item()], self.m_parameter[1:])
        

    def run(self):
            self.model.train()
            self.model.shared_memory()
            num_processes = 2
            # NOTE: this is required for the ``fork`` method to work
            self.model.share_memory()
            processes = []
            for rank in range(num_processes):
                p = mp.Process(target=train, args=(model,))
                p.start()
                processes.append(p)
            for p in processes:
              p.join()







            while True:
                # pull params synchronously for now (TODO: figure out how to express SGD client with async as an actor) 
                print("sent param request method")
                send_message('ParameterRequest', torch.zeros(squash_model(self.model).size()))
                time.sleep(5)

                # if args.cuda:
                    # data, target = data.cuda(), target.cuda()
                # data, target = Variable(data), Variable(target)
                # optimizer.zero_grad()
                # output = model(data)
                # loss = F.nll_loss(output, target)
                # loss.backward()

                # self.send_message('GraidentUpdate', gradients)

                # and this is our internal gradient update
                # set_params(self.model, self.squash_model() - self.learning_rate * gradients)
