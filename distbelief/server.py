# 
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
    def __init__(self, model):
        _LOGGER.info("Creating ParameterServer")
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model = model
        #init superclass
        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        print("Processing message: {} from sender {}".format(message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            #be sure to clone here
            self.parameter_shard = parameter.clone()

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)    

        elif message_code == MessageCode.GradientUpdate:
            self.parameter_shard.add_(parameter)
