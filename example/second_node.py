from main import main
from distbelief.utils import init_processes

if __name__ == "__main__":
    init_processes(2, 3, main)
