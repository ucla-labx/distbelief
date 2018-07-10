"""
This class listens for a ParameterUpdate from the parameter server and then updates the model accordingly
"""
import logging
import threading 
from utils import ravel_model_params, unravel_model_params, MessageCode, MessageListener, send_message, init_processes

from torch.optim.optimizer import Optimizer, required

_LOGGER = logging.getLogger(__name__)

class DownpourListener(MessageListener):
    """Client code for interacting with server. Training clients should run an instance of this class in a separate thread."""
    def __init__(self, model):
            super().__init__(model)

    def receive(self, sender, message_code, parameter):
            """receive parameter updates from the server and reflect them into the client's model."""
            _LOGGER.info("Processing message: {}".format(message_code.name))
            if message_code == MessageCode.ParameterUpdate:
                    unravel_model_params(self.model, parameter)


class DownpourSGD(Optimizer):

    def __init__(self, params, lr=required, freq=15, model=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,)
        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size())
        self.freq = freq

        model.share_memory()
        # this sets the initial model parameters
        send_message(MessageCode.ParameterUpdate, ravel_model_params(model))
        # start the  training thread
        update_thread = threading.Thread(target=listen, args=(model,))
        update_thread.start()

        super(DownpourSGD, self).__init__(params, defaults)

    @staticmethod
    def listen(model):
            """Init and run the sgd client"""
            sgd_client = DownpourListener(model=model)
            sgd_client.run()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # send gradient request every 10 iterations
        if batch_idx % self.freq == 0:
            send_message(MessageCode.ParameterRequest, torch.zeros(ravel_model_params(model).size()))

        gradients = ravel_model_params(model, grads=True)
        accumulated_gradients += gradients

        if batch_idx % self.freq == 0:
            send_message(MessageCode.GradientUpdate, accumulated_gradients) # send gradients to the server
            accumulated_gradients = torch.zeros(ravel_model_params(model).size())

        # internal sgd update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss

