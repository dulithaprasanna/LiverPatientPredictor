from flask import Flask,render_template,url_for,request
app = Flask(__name__)

import pandas as pd
import numpy as np

from sklearn.externals import joblib

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view')
def view():
    data = pd.read_csv("data/liver_patient.csv")
    return render_template('data_view.html', data_view = data)

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        Total_Bilirubin = request.form['Total_Bilirubin']
        Direct_Bilirubin = request.form['Direct_Bilirubin']
        Alkaline_Phosphotase = request.form['Alkaline_Phosphotase']
        Alamine_Aminotransferase = request.form['Alamine_Aminotransferase']
        Aspartate_Aminotransferase = request.form['Aspartate_Aminotransferase']
        Total_Protiens = request.form['Total_Protiens']
        Albumin = request.form['Albumin']
        Albumin_and_Globulin_Ratio = request.form['Albumin_and_Globulin_Ratio']
        model = request.form['model']

        data = [age, gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]
        ex1= np.array(data).reshape(1,-1)
        ex1 = ex1.astype(np.float)

        if model == 'lgmodel':
            logit_model = joblib.load("data/logit_model.pkl")
            result_prediction = logit_model.predict(ex1)
        elif model == 'knnmodel':
            knn_model = joblib.load("data/knn_model.pkl")
            result_prediction = knn_model.predict(ex1)
        elif model == 'gboostmodel':
            gboost_model = joblib.load("data/gboost_model.pkl")
            result_prediction = gboost_model.predict(ex1)
        else:
            logit_model = joblib.load("data/logit_model.pkl")
            result_prediction = logit_model.predict(ex1)


    return render_template("index.html", age = age, gender = gender,
                     Total_Bilirubin = Total_Bilirubin,
                    Direct_Bilirubin = Direct_Bilirubin,
                    Alkaline_Phosphotase = Alkaline_Phosphotase,
                    Alamine_Aminotransferase = Alamine_Aminotransferase,
                    Aspartate_Aminotransferase = Aspartate_Aminotransferase,
                    Total_Protiens = Total_Protiens,
                    Albumin = Albumin,
                    Albumin_and_Globulin_Ratio = Albumin_and_Globulin_Ratio,
                    result_prediction = result_prediction )

if __name__ == '__main__':
    app.run(debug=True)