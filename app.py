from flask import Flask, render_template, request, jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomClass
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import sys

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def prediction_data():
    try:
        if request.method == 'GET':
            return render_template('home.html')
    
        else:
            data = CustomClass(
                age = int(request.form.get('age')),
                workclass = int(request.form.get('workclass')),
                education_num = int(request.form.get('education_num')),
                marital_status = int(request.form.get('marital_status')),
                occupation = int(request.form.get('occupation')),
                relationship = int(request.form.get('relationship')),
                race = int(request.form.get('race')),
                sex = int(request.form.get('sex')),
                capital_gain = int(request.form.get('capital_gain')),
                capital_loss = int(request.form.get('capital_loss')),
                hours_per_week = int(request.form.get('hours_per_week')),
                native_country = int(request.form.get('native_country')))

        df = data.get_dataframe()
        pipeline_prediction = PredictionPipeline()
        pred = pipeline_prediction.predict(df)

        result = pred

        if result == 0:
            return render_template("results.html", final_result = "Your Yearly Income is Less than or Equal to 50k:{0}".format(result))

        elif result == 1:
            return render_template("results.html", final_result = "Your Yearly Income is More than 50k:{0}".format(result))
            
    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
     app.run(host = "0.0.0.0", debug = True)


