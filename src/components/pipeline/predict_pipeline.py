import sys
import os
import pandas as pd

from src.logger import logging

from src.exception import UserException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info('Obtating model and preprocessor objects...!')
            model_path = 'models\\model.pkl'
            preprocessor_path = 'preprocessor_pkl_file\\preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            # print(features)
            logging.info('model predicting the target value...!')
            # print(type(model))
            preds = model.predict(data_scaled)
            # output = pd.DataFrame(preds, columns=['target'])
            # output[['target']] = output[['target']].replace({0:'No_heart_disease', 1:'heart_disease'})
            if preds == 0:
                return 'No_heart_disease'
                logging.info('model successfully predicted the target value and returned the result to the user...!')

            else:
                return 'Heart_disease'
                logging.info('model successfully predicted the target value and returned the result to the user...!')


            # return output

        except Exception as error:
            raise UserException(error, sys)
        

class CustomData:
    def __init__(self,age,sex,cp,trestbps, chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'sex':[self.sex],
                'cp' :[self.cp],
                'trestbps' : [self.trestbps],
                'chol' : [self.chol],
                'fbs' : [self.fbs],
                'restecg' : [self.restecg],
                'thalach' : [self.thalach],
                'exang' : [self.exang],
                'oldpeak' : [self.oldpeak],
                'slope' : [self.slope],
                'ca' : [self.ca],
                'thal' : [self.thal]
            }

            mydf =  pd.DataFrame(custom_data_input_dict)
            mydf[['sex']] = mydf[['sex']].replace({'Male':0, 'Female':1})
            mydf[['cp']] = mydf[['cp']].replace({'Typical angina':0, 'Atypical angina':1, 'Non-anginal pain':2, 'Asymptomatic':3})
            mydf[['fbs']] = mydf[['fbs']].replace({'Yes':1, 'No':0})
            mydf[['restecg']] = mydf[['restecg']].replace({'Normal':0, 'Having ST-T wave abnormality':1, 'Showing probable or definite left ventricular hypertrophy':2})
            mydf[['exang']] = mydf[['exang']].replace({'Yes':1, 'No':0})
            mydf[['slope']] = mydf[['slope']].replace({'Upsloping':0, 'Flat':1, 'Downsloping':2})
            mydf[['thal']] = mydf[['thal']].replace({'Normal':0, 'Fixed defect':1, 'Reversible defect':2, 'Not described':3})
            logging.info('data successfully fetched and returned as dataframe...!')
            # print(mydf)
            return mydf
        except Exception as error:
            raise UserException(error, sys)


        

