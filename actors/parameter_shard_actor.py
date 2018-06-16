class ParameterShardActor(Actor):

    def __init__(self, learning_rate, model, random_seed=42):
        """__init__

        :param learning_rate:
        :param params: a dict of str -> pytorch tensor that represents the gradients
        :param random_seed:
        """
        
        self.parameters = torch.zeros(self.squash_model().size())
        self.learning_rate = learning_rate


    def receive(self, message, gradient):
        message_code = m_parameter[0]

        if message == 'ParmaterRequest':
            self.send_message('ParameterUpdate', self.parameters)
        
        if message == 'GradientUpdate':
            #get the gradients
            self.parameters -= self.learning_rate * gradient

