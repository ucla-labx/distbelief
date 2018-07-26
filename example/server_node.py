from distbelief.server import ParameterServer
from distbelief.utils.distributed import init_processes
from main import AlexNet

def init_server():
    model = AlexNet()
    server = ParameterServer(model=model)
    server.start()

if __name__ == "__main__":
     init_processes(0, 3, init_server)
