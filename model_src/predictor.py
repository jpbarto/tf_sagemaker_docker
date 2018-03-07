# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import StringIO

import flask

import numpy as np
import pandas as pd

import model
import model_data

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

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
    (test_data, test_labels) = model_data.test_data('/opt/ml/input/data')
    test_error = error_rate(model.predict(test_data, {'model_path': model_path}), test_labels)

    status = 200 if test_error < 10 else 404
    return flask.Response(response='{"test_error": '+ str(test_error) +'}\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/json':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = np.array(json.loads(s), dtype=np.float32)
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = model.predict(data, {'model_path': model_path})

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    pd.DataFrame({'results': predictions}).to_csv(
        out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
