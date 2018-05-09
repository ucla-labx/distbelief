import os
import torch
from flask import Flask, url_for, request



app = Flask(__name__)


parameters = []
@app.route('/')
def api_root():
	return 'Welcome'

def get_parameters():
	parameters.append(1)
	return 'params len: {}'.format(len(parameters))

def receive_parameters():
	return 'You sent some parameters'

@app.route('/api/parameters', methods = ['GET', 'POST'])
def handle_parameters():
	if request.method == 'GET':
		return get_parameters()
	elif request.method == 'POST':
		return receive_parameters()


@app.route('/api/gradients',  methods = ['POST'])
def handle_gradient_post():
	return 'You posted some gradients!'

@app.route('/echo', methods = ['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def api_echo():
    if request.method == 'GET':
        return "ECHO: GET\n"

    elif request.method == 'POST':
        return "ECHO: POST\n"

    elif request.method == 'PATCH':
        return "ECHO: PACTH\n"

    elif request.method == 'PUT':
        return "ECHO: PUT\n"

    elif request.method == 'DELETE':
        return "ECHO: DELETE"

if __name__ == '__main__':
	app.run()
