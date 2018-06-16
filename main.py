"""
gevent actor test
"""
import gevent
from gevent.queue import Queue
from gevent import Greenlet
from actors.actor import ModelActor

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
	ping = Pinger(model=None)
	pong = Ponger(model=None)

	ping.start()
	pong.start()
	ping.inbox.put('start')
	gevent.joinall([ping, pong])
