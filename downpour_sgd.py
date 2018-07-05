"""
This class listens for a ParameterUpdate from the parameter server and then updates the model accordingly
"""
import logging
from utils import unravel_model_params, MessageCode, MessageListener

_LOGGER = logging.getLogger(__name__)

class DownpourSGD(MessageListener):
	"""Client code for interacting with server. Training clients should run an instance of this class in a separate thread."""
	def receive(self, sender, message_code, parameter):
		"""receive parameter updates from the server and reflect them into the client's model."""
		_LOGGER.info("Processing message: {}".format(message_code.name))
		if message_code == MessageCode.ParameterUpdate:
			unravel_model_params(self.model, parameter)

def init_sgd(model):
	"""Init and run the sgd client"""
	sgd_client = DownpourSGD(model=model)
	sgd_client.run()

