# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import StringIO

import flask
from flask import g

import numpy as np
import pandas as pd

import mnist_model
import model_data

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model', 'model.ckpt')

def get_model ():
    model = getattr (g, '_mnist_model', None)
    if model is None:
        model = mnist_model.Model ({})
        model.restore (model_path)
        g._mnist_model = model

    return model

# The flask app for serving predictions
app = flask.Flask(__name__)

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == labels) /
        predictions.shape[0])

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    model = get_model ()
    (test_data, test_labels) = model_data.test_data('/opt/ml/input/data/eval')
    test_error = error_rate(model.predict(test_data), test_labels)

    status = 200 if test_error < 10 else 404
    return flask.Response(response='{"test_error": '+ str(test_error) +'}\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from JSON to numpy
    if flask.request.content_type == 'text/json':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = np.array(json.load(s), dtype=np.float32)
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    (test_data, test_labels) = model_data.test_data('/opt/ml/input/data/eval')
    print('Input data has shape {}'.format(data.shape))

    model = get_model ()
    # Do the prediction
    predictions = []
    for img in data:
        predictions += model.predict(np.array ([img])).tolist ()

    return flask.Response(response=json.dumps (predictions), status=200, mimetype='text/json')
