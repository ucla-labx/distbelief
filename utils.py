from enum import Enum
import logging

class Messages(Enum):
    ParameterRequest = 'parameter_request'
    GradientUpdate = 'gradient_update'
    ParameterUpdate = 'paramater_update'
    Train = 'train'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)
