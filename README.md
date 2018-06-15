# distbelief
Implementing Google's DistBelief paper

## DownpourSGD for PyTorch

DownpourSGD is pretty simple, there are two core concepts - a parameter server and a training node.

The parameter server is just a copy of the model parameters, it can get a gradient update or send parameters.

The training node asynchronously pulls the parameters, and then does your usual train step (compute loss, the backprop).
Once we've gotten the gradients, we send a copy to the parameter server, apply the update, and then continue training. 

## Actor Model
First we implemented a simple actor model that communicates via pytorch's `send` and `recv`. 

Core concept of the actor model is that we have **actors**, each of which has a mailbox. An actor can respond to a message in it's mailbox in one of three ways.
- send messages to other actors
- create new actors
- specify what behavior to be used for the next message
-
Everything the actors do will be single threaded.

### Paramater Shard Actor

This actor is extremely simple. 
There are two messages that it will take - `ParameterRequest` and `GradientUpdate`. 
- ParameterRequest corresponds to a request for parameters. In this case, we issue a message back of type ParameterUpdate, which contains the parameters to update
- GradientUpdate means we have a gradient update to apply to our parameters, so we apply them. 

That's all there is to implement. This is perfect because since only one greenlet wil run at once we won't have any weird race conditions when updating parameters.

### SGD Client Actor

This is a bit less clear - see the sgd client is supposed to do two things, pull gradeints asynchronously and train. 

TODO: figure out how to do SGDClinet async as an actor
TODO: figure out pytorch distributed to start the actors
TODO: update message with a sender value


### References
- [Pytorch distributed tutorial](http://pytorch.org/tutorials/intermediate/dist_tuto.html)

