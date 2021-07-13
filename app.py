import joblib
from flask import Flask, render_template, request
from helpers.dummy import *
import numpy as np


app = Flask(__name__)
model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    data1 = [
        float(request.args['age']),
        float(request.args['trtbps']),
        float(request.args['chol']),
        float(request.args['thalachh']),
        float(request.args['oldpeak'])
    ]

    data1 = np.reshape(data1, (1, -1))
    data1 = scaler.transform(data1)
    data2 = []

    for v in data1:
        for x in v:
            data2.append(float(x))

    data3 =[
        float(request.args['Gender']),
        float(request.args['exng'])
    ]
    data = data2+data3

    data += major_vessels[request.args['caa']]
    data += Chest_pain[request.args['cp']]
    data4 = [
        float(request.args['fbs'])

    ]
    data += data4
    data += Resting_electrocardiographic[request.args['restecg']]
    data += Slope[request.args['slp']]
    data += Thalium_Stress[request.args['thall']]


    prediction = round(model.predict([data])[0])
    x=''
    if prediction ==0:
        x='less chance of heart attack'

    else:
        x='more chance of heart attack'

    return render_template('result.html', the_predicted_type_is=x)


if __name__ == "__main__":
    app.run(debug=True)
    

