"""
Parameter server for distbelief
"""
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from utils import ravel_model_params, send_message, init_processes, DEFAULT_LEARNING_RATE, MessageCode, MessageListener, unravel_model_params

from torchvision import datasets, transforms
from model import Net
from experimental import parameter_server_test

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)

class ParameterServer(MessageListener):

    def __init__(self, learning_rate, model, random_seed=42):
        _LOGGER.info("Creating ParameterServer with LR: {}".format(learning_rate))
        self.learning_rate = learning_rate
        self.model = model
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.log_dataframe = []
        self.idx = 0
        #init superclass
        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        _LOGGER.info("Processing message: {} from sender {}".format(message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            #be sure to clone here
            self.parameter_shard = parameter.clone()

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)    

        elif message_code == MessageCode.GradientUpdate:
            self.idx += 1
            self.parameter_shard -= self.learning_rate * parameter
            if self.idx % 10 == 0:
                unravel_model_params(self.model, self.parameter_shard)
                parameter_server_test(self.model, self.log_dataframe)

        elif message_code == MessageCode.EvaluateParams:
            evaluate(self.log_dataframe)
    

def init_server(rank, size):
    model = Net()
    server = ParameterServer(learning_rate=DEFAULT_LEARNING_RATE, model=model)
    server.run()

if __name__ == "__main__":
     p = Process(target=init_processes, args=(0, 3, init_server))
     p.start()
     p.join()
