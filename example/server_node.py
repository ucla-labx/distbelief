from distbelief.server import ParameterServer
from distbelief.utils import init_processes
from main import Net

def init_server():
    model = Net()
    server = ParameterServer(model=model)
    server.start()

if __name__ == "__main__":
     init_processes(0, 3, init_server)
