from enum import Enum

class Messages(Enum):
    ParameterRequest = 'parameter_request'
    GradientUpdate = 'gradient_update'
    ParameterUpdate = 'paramater_update'
    Train = 'train'


class ModelMessage(object):
	def __init__(self, message_type, parameters=None, gradients=None):
		self.message_dict = {
		'message_type': message_type,
		'parameters': parameters,
		'gradients': gradients
		}

	