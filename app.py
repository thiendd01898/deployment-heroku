from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import re
from werkzeug.debug import DebuggedApplication

app = Flask(__name__)

model = load_model('deployment_28042020')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']


@app.route('/')
def home():
    # x = 1 + 5
    # y = x + 10
    # raise //uncomment for live debugging
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


@app.route('/hello')
def hello():
    return 'Hello, World'


@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content


@app.route('/crash')
def crash():
    raise Exception()


def create_app():
    # Insert whatever else you do in your Flask app factory.

    if app.debug:
        app.wsgi_app = DebuggedApplication(app.wsgi_app, evalex=True)

    return app


# Comment below lines for VSC debugger
if __name__ == '__main__':
    app.run(debug=True)