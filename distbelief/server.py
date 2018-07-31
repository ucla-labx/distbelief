"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
from distbelief.utils.messaging import MessageCode, MessageListener, send_message
from distbelief.utils.serialization import ravel_model_params, unravel_model_params

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

class ParameterServer(MessageListener):
    """ParameterServer"""
    def __init__(self, model, lr=0.01):
        _LOGGER.info("Creating ParameterServer")
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.lr = lr
        self.model = model
        # init an optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
        self.optimizer.zero_grad()
        #init superclass
        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        print("Processing message: {} from sender {}".format(message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            #be sure to clone here
            cloned_params = parameter.clone()
            self.parameter_shard = cloned_params
            # also unravel model params into the model, keep self.model params and parameter_shard the same
            unravel_model_params(self.model, cloned_params)


        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)    

        elif message_code == MessageCode.GradientUpdate:
            # we get only gradients from client, so scale by LR for SGD
            self.parameter_shard.add_(-self.lr, parameter)
            # keep parameter shard and model.params the same
            self.optimizer.step() # updates model.params
            # training breaks if we comment in the next line
            # self.parameter_shard = ravel_model_params(self.model)
            self.optimizer.zero_grad()
