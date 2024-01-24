import sys
from src.logger import logging
from src.exception import UserException

from flask import Flask, request, render_template
from src.components.pipeline.predict_pipeline import CustomData, PredictPipeline

logging.info('creating flask application...!')
app = Flask(__name__)
logging.info('finished creating flask application...!')

@app.route('/heartdiseaseclassifier', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    
    else:
        logging.info('fetching data from the home.html...!')
        data = CustomData(
            age = request.form.get('age'),
            sex = request.form.get('sex'),
            cp = request.form.get('cp'),
            trestbps = request.form.get('trestbps'),
            chol = request.form.get('chol'),
            fbs = request.form.get('fbs'),
            restecg = request.form.get('restecg'),
            thalach = request.form.get('thalach'),
            exang = request.form.get('exang'),
            oldpeak = request.form.get('oldpeak'),
            slope = request.form.get('slope'),
            ca = request.form.get('ca'),
            thal = request.form.get('thal')
        )

        mydf = data.get_data_as_data_frame()
        print(mydf)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(mydf)
        return render_template('home.html', results = results)
    

if __name__ =='__main__':
    app.run(host = '0.0.0.0', debug = True)
