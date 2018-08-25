import logging
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
# 
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.messaging import MessageCode, MessageListener, send_message

_LOGGER = logging.getLogger(__name__)

class AsyncListener(MessageListener):
    """AsyncListener"""
    def __init__(self, model):
        self.model = model

        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        """receive parameter updates from the server and reflect them into the client's model."""
        _LOGGER.info("Processing message: {}".format(message_code.name))
        if message_code == MessageCode.ParameterUpdate:
            unravel_model_params(self.model, parameter)

        elif message_code == MessageCode.GradientUpdate:
            param_shard = ravel_model_params(self.model).add_(parameter)
            unravel_model_params(self.model, param_shard)
            send_message(MessageCode.ParameterUpdate, param_shard, dst=sender)    

class AsyncDecentralizedSGD(Optimizer):
    """AsyncDecentralizedSGD"""

    def __init__(self, params, lr=required, n_sync=required, model=required):
        """__init__

        :param params:
        :param lr:
        :param freq:
        :param model:
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,)
        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size())
        self.n_sync = n_sync

        self.model = model
        self.idx = 0

        listener = AsyncListener(self.model)
        listener.start()

        super(AsyncDecentralizedSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        #get the lr
        lr = self.param_groups[0]['lr']
        # keep track of accumulated gradients so that we can send 
        gradients = ravel_model_params(self.model, grads=True)
        self.accumulated_gradients.add_(-lr, gradients)

        # we only send one message
        if self.idx % self.n_sync == 0:
            send_message(MessageCode.GradientUpdate, self.accumulated_gradients, dist.get_rank()+1 % dist.get_world_size()) # send gradients to the next thing in the chain
            self.accumulated_gradients.zero_()

        # internal sgd update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        
        self.idx += 1
        return loss
