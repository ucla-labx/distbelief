
class SGDClientActor(Actor):

    def __init__(self, learning_rate, model):
        self.model = model
        self.learning_rate = learning_rate
        
    def receive(self, message, parameter):

        if message == 'ParmaterUpdate':
            self.set_params(parameter)
            gevent.sleep(0)
        
        if message == 'Train':
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):

                # pull params synchronously for now (TODO: figure out how to express SGD client with async as an actor)
                self.send_message('ParameterRequest', torch.zeros(self.squash_mode().size()))
                gevent.sleep(0) #give up until we get our parameter update

                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                gradients = self.squash_model(grads=True)
                self.send_message('GraidentUpdate', gradients)

                # and this is our internal gradient update
                self.set_params(self.squash_model() - self.learning_rete * gradients)
