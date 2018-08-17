setup:
	-sudo apt-get -y virtualenv
	virtualenv -p python3 venv
	. venv/bin/activate && pip install -r requirements.txt && pip install .

install:
	pip install .

graph:
	python example/graph.py
	mv train_time.png test_time.png docs

local:

