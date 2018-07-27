first:
	python -u example/first_node.py | tee first.log

second:
	python -u example/second_node.py | tee second.log

server:
	python -u example/server_node.py | tee server.log

install:
	pip install .
