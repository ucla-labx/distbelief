from main import main
from distbelief.utils.distributed import init_processes

from torch.multiprocessing import Process
from distbelief.server import ParameterServer
from distbelief.utils.distributed import init_processes
from main import Net

def init_server():
    model = Net()
    server = ParameterServer(model=model)
    server.start()

if __name__ == "__main__":
    processes = []
    server = Process(target=init_processes, args=(0, 3, init_server))
    server.start()
    processes.append(server)
    first = Process(target=init_processes, args=(1, 3, main))
    first.start()
    processes.append(first)
    second = Process(target=init_processes, args=(2, 3, main))
    second.start()
    processes.append(second)

    for p in processes:
        p.join()
