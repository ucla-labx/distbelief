server:
	python parameter_server.py

train:
	python -u main.py | tee train.log
