# distbelief
Implementing Google's DistBelief paper

## DownpourSGD for PyTorch

## Actor Model
First we implemented a simple actor model that communicates via pytorch's `send` and `recv`. This can be done 

### Paramater Shard Actor

This actor is extremely simple. 
There are two messages that it will take - `ParameterRequest` and `GradientUpdate`. 
- ParameterRequest corresponds to a request for parameters. In this case, we issue a message back of type ParameterUpdate, which contains the parameters to update
- GradientUpdate




### Server class
- API Routes:
	- Post data and MLModelObject that is user defined to the client
	- GET endpoint that satisfies client request to get parameters

- Functions
	- __init__ - get data from the user, get pytorch model from the user, initialize MLModelObject, post that to all of the clients
	- 2 above API routes
	- updateWeights() function


### References
- [Pytorch distributed tutorial](http://pytorch.org/tutorials/intermediate/dist_tuto.html)


## Actor model
