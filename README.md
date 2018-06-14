# distbelief
Implementing Google's DistBelief paper


#### MLModelObject Class

- Attributes:
	- library_type (only pytorch for now)
	- list of weights
	- list of gradients
	- reference to forward pass
	reference to backward pass


#### Client Class

- API Routes:
	- Post to server:
		- Send gradients of the MLModelObject
			- Done by calling MLModelObject to get list of gradients
			- MLModelObject should abstract the type of library we are using and the way the parameters are actually obtained
	- GET from server:
		- Get parameters from server
		- Probably a really simple request, as there will only be one MLModelObject for class

- Functions:
	- __init__ - get data and MLModelObject from the server
	- run(n_iterations, n_get, n_send):
		- run iterations of gradient descent
		- calls getParameters() and sendGradients() 
		- might have to lock parameters 
	- getParameters(blocking=False) get parameters from the server
	- sendGradientsToServer() push grads to the server

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
