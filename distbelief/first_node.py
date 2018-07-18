from main import main
from utils import init_processes

if __name__ == "__main__":
    init_processes(1, 3, main)
