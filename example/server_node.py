from distbelief.server import ParameterServer
from distbelief.utils.distributed import init_processes
from main import Net


def init_server():
    model = Net()
    print('before parameter server constructed')
    server = ParameterServer(model=model, lr=0.01)
    print('parameter server constructed')
    print('fuck fuck')
    server.run()
    print('called server.run')


if __name__ == "__main__":
     init_processes(0, 3, init_server)
