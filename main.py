"""
gevent actor test
"""
import gevent
from gevent.queue import Queue
from gevent import Greenlet
from actors.actor import ModelActor
from actors.parameter_shard_actor import ParameterShardActor
from actors.sgd_client_actor import SGDClientActor
from models.mnist import Net

DEFAULT_LEARNING_RATE = 0.005

class Pinger(ModelActor):
    def receive(self, message):
        print("in pinger receive: {}".format(message))
        pong.inbox.put('ping')
        gevent.sleep(0)

class Ponger(ModelActor):
    def receive(self, message):
        print('in ponger receive: {}'.format(message))
        ping.inbox.put('pong')
        gevent.sleep(0)



if __name__ == '__main__':
    model = Net()
    parameter_shard = ParameterShardActor(learning_rate=DEFAULT_LEARNING_RATE, model=model)
    sgd_client = SGDClientActor(learning_rate=DEFAULT_LEARNING_RATE, model=model, server_actor=parameter_shard)
    parameter_shard.add_client(sgd_client)
    actors = [parameter_shard, sgd_client]
    for actor in actors:
        actor.start()
    gevent.joinall(actors)

	# ping = Pinger(model=None)
	# pong = Ponger(model=None)
	# ping.start()
	# pong.start()
	# ping.inbox.put('start')
	# gevent.joinall([ping, pong])
