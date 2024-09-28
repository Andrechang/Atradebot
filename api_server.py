'''
model server for api calls
'''
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename
import base64
from atradebot import init_model, run_model, collect_input

pretrained_path = '' # select model path
model = init_model(pretrained_path)

app = Flask(__name__)

# create your own secret api key
API_KEYS = [""] 

@app.before_request
def check_api_key():
    api_key = request.headers.get('x-api-key')
    if api_key not in API_KEYS:
        abort(401)


@app.route('/advise', methods=['POST'])
def run_advise():
    try:
        # collect input data to feed model
        userconfig = request.json['config'] #user config
        data = collect_input()

        #run model
        response = run_model(data, model)
    except:
        return jsonify({'error': 'Something went wrong'}), 400

    return jsonify({'message': response,}), 200

if __name__ == '__main__':
    app.run()

