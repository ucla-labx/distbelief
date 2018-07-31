first:
	python example/main.py --rank 1 --world-size 3 --distributed

second:
	python example/main.py --rank 2 --world-size 3 --distributed

server:
	python example/main.py --rank 0 --world-size 3 --server --distributed

install:
	pip install .

single:
	python example/main.py 
