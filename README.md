# distbelief
Implementing Google's DistBelief paper

### Usage Instructions

## DownpourSGD for PyTorch

DownpourSGD is pretty simple, there are two core concepts - a parameter server and a training node.

The parameter server is just a copy of the model parameters, it can get a gradient update or send parameters.

The training node asynchronously pulls the parameters, and then does your usual train step (compute loss, the backprop).
Once we've gotten the gradients, we send a copy to the parameter server, apply the update, and then continue training. 


### Sending messages

We're using `dist.send` and `dist.recv` to send and receive tensors via PyTorch.



### Parameter Server

The parameter server simply polls, 

### Training

Here our training process uses two threads



### References
- [Pytorch distributed tutorial](http://pytorch.org/tutorials/intermediate/dist_tuto.html)

