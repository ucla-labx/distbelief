setup:
	-sudo apt-get -y virtualenv
	virtualenv -p python3 venv
	. venv/bin/activate && pip install -r requirements.txt && pip install .

install:
	pip install .

dist:
	python3 setup.py sdist bdist_wheel

upload: dist
	twine upload dist/*

upload-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

install-test:
	python3 -m pip install --index-url https://test.pypi.org/simple/ pytorch-distbelief	

graph:
	python example/graph.py
	mv train_time.png test_time.png docs

first:
	python example/main.py --rank 1 --world-size 3

second:
	python example/main.py --rank 2 --world-size 3

server:
	python example/main.py --rank 0 --world-size 3 --server

single:
	python example/main.py --no-distributed 

gpu:
	python example/main.py --no-distributed --cuda


