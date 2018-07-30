import logging
import torch
from torch.optim.optimizer import Optimizer, required
import torch.optim
from threading import Thread
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.messaging import MessageCode, MessageListener, send_message

_LOGGER = logging.getLogger(__name__)


class DownpourListener(MessageListener):
    """DownpourListener"""
    def __init__(self, model):
        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        """receive parameter updates from the server and reflect them into the client's model."""
        _LOGGER.info("Processing message: {}".format(message_code.name))
        if message_code == MessageCode.ParameterUpdate:
            unravel_model_params(self.model, parameter)


class DownpourSGD(Optimizer):
    """DownpourSGD"""

    def __init__(self, params, lr=required, freq=required, model=required, internal_optim=None):
        """__init__

        :param params:
        :param lr:
        :param freq:
        :param model:
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,)
        self.lr = lr
        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size())
        self.freq = freq

        self.model = model
        # this sets the initial model parameters
        send_message(MessageCode.ParameterUpdate, ravel_model_params(self.model))
        self.idx = 0

        # create a listener that runs in a separate thread, checking for info from servers.
        listener = DownpourListener(self.model)
        print('starting thread')
        t = Thread(target=listener.run)
        t.start()

        # initializing internal optimizer
        self.internal_optim = internal_optim if internal_optim else optim.SGD(params, defaults)
        super(DownpourSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # send parameter request every N iterations
        if self.idx % self.freq == 0:
            send_message(MessageCode.ParameterRequest, self.accumulated_gradients) # dummy val

        # keep track of accumulated gradients so that we can send 
        gradients = ravel_model_params(self.model, grads=True)
        # this does self.accumulated_gradients+=-self.lr * gradients
        self.accumulated_gradients.add_(-self.lr, gradients)

        # send gradient update every N iterations
        if self.idx % self.freq == 0:
            send_message(MessageCode.GradientUpdate, self.accumulated_gradients) # send gradients to the server
            self.accumulated_gradients.zero_()

        # internal sgd update

        self.internal_optim.step(closure)
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad.data
        #         p.data.add_(-group['lr'], d_p)
        
        self.idx += 1
        return loss

    # def start_listening(self):
    #     self.listener.start()
    #
    # def stop_listening(self):
    #     self.listener.join()


