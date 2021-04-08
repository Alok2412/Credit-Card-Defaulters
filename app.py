import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    LIMIT_BAL = [float(request.form.get("LIMIT_BAL"))]
    SEX = [float(request.form.get("SEX"))]
    EDUCATION = [float(request.form.get("EDUCATION"))]
    MARRIAGE = [float(request.form.get("MARRIAGE"))]
    AGE = [float(request.form.get("AGE"))]
    PAY_0 = [float(request.form.get("PAY_0"))]
    PAY_2 = [float(request.form.get("PAY_2"))]
    PAY_3 = [float(request.form.get("PAY_3"))]
    PAY_4 = [float(request.form.get("PAY_4"))]
    PAY_5 = [float(request.form.get("PAY_5"))]
    PAY_6 = [float(request.form.get("PAY_6"))]
    BILL_AMT1 = [float(request.form.get("BILL_AMT1"))]
    BILL_AMT2 = [float(request.form.get("BILL_AMT2"))]
    BILL_AMT3 = [float(request.form.get("BILL_AMT3"))]
    BILL_AMT4 = [float(request.form.get("BILL_AMT4"))]
    BILL_AMT5 = [float(request.form.get("BILL_AMT5"))]
    BILL_AMT6 = [float(request.form.get("BILL_AMT6"))]
    PAY_AMT1 = [float(request.form.get("PAY_AMT1"))]
    PAY_AMT2 = [float(request.form.get("PAY_AMT2"))]
    PAY_AMT3 = [float(request.form.get("PAY_AMT3"))]
    PAY_AMT4 = [float(request.form.get("PAY_AMT4"))]
    PAY_AMT5 = [float(request.form.get("PAY_AMT5"))]
    PAY_AMT6 = [float(request.form.get("PAY_AMT6"))]
    
    
    data = {
    'LIMIT_BAL':LIMIT_BAL,
    'SEX':SEX,
    'EDUCATION':EDUCATION,
    'MARRIAGE':MARRIAGE,
    'AGE':AGE,
    'PAY_0':PAY_0,
    'PAY_2':PAY_2,
    'PAY_3':PAY_3,
    'PAY_4':PAY_4,
    'PAY_5':PAY_5,
    'PAY_6':PAY_6,
    'BILL_AMT1':BILL_AMT1,
    'BILL_AMT2':BILL_AMT2,
    'BILL_AMT3':BILL_AMT3,
    'BILL_AMT4':BILL_AMT4,
    'BILL_AMT5':BILL_AMT5,
    'BILL_AMT6':BILL_AMT6,
    'PAY_AMT1':PAY_AMT1,
    'PAY_AMT2':PAY_AMT2,
    'PAY_AMT3':PAY_AMT3,
    'PAY_AMT4':PAY_AMT4,
    'PAY_AMT5':PAY_AMT5,
    'PAY_AMT6':PAY_AMT6
    }
    df = pd.DataFrame(data)

    prediction = model.predict(df)
    
    output = round(prediction[0], 8)
    
    probability = model.predict_proba(df)

    return render_template('index.html', prediction_text='Predicted status of customer is: {} with the probability of being non defaulter is: {} and with the probability of being defaulter is: {}'.format(output,probability[0][0],probability[0][1]))


if __name__ == "__main__":
    app.run(debug=True)