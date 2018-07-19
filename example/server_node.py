from distbelief.server import ParameterServer
from main import Net

def init_server():
    model = Net()
    server = ParameterServer(learning_rate=0.001, model=model)
    server.start()

if __name__ == "__main__":
     init_processes(0, 3, init_server)
