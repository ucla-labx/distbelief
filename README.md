# distbelief
Implementing Google's DistBelief paper.

Check out the (https://jcaip.github.io/Distbelief/)[blog post]!
## Installation/Development instructions

To install the latest stable version (pytorch-distbelief 0.1.0), run `pip install pytorch-distbelief`

Otherwise, you can build and run the latest master with the instructions below.

You'll want to create a python3 virtualenv first by running `make setup`, after which, you should run `make install`. 

You'll then be able to use distbelief by importing distbelief
```python 

from distbelief.optim import DownpourSGD

optimizer = DownpourSGD(net.parameters(), lr=0.1, n_push=5, n_pull=5, model=net)

```

As an example, you can see our implementation running by using the script provided in `example/main.py`.

To run a 2-training node setup locally, open up three terminal windows, source the `venv` and then run `make first`, `make second`, and `make server`.
This will begin training AlexNet on CIFAR10 locally with all default params.

## Benchmarking

**NOTE:** we graph the train/test accuracy of each node, hence node1, node2, node3. A better comparison would be to evaluate the parameter server's params and use that value.
However we can see that the accuracy between the three nodes is fairly consistent, and adding an evaluator might put too much stress on our server. 

We scale the learning rate of the nodes to be learning_rate/freq (.03) .

![train](/docs/train_time.png)

![test](/docs/test_time.png)

We used AWS c4.xlarge instances to compare the CPU runs, and a GTX 1060 for the GPU run.

## DownpourSGD for PyTorch

### Diagram

<img src="./docs/diagram.jpg" width="500">

Here **2** and **3** happen concurrently. 

You can read more about our implementation [here](https://jcaip.github.io/Distbelief/).

### References
- [Pytorch distributed tutorial](http://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Akka implementation of distbelief](http://alexminnaar.com/implementing-the-distbelief-deep-neural-network-training-framework-with-akka.html)
- [gevent actor tutorial](http://sdiehl.github.io/gevent-tutorial/#actors)
- [DistBelief paper](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
- [Analysis of delayed grad problem](https://openreview.net/pdf?id=BJLSGcywG)
