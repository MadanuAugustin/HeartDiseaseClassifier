import os
import sys
import numpy as np

from src.exception import UserException
from src.logger import logging

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
## from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from dataclasses import dataclass


from src.utils import save_object, model_evaluation


@dataclass
class ModelTrainerConfig:
        trained_model_path = os.path.join('models', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,  train_arr, test_arr):
        try:
            logging.info('started splitting train and test data...!')

            X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])


            models = [
               {
                   'classifier' : AdaBoostClassifier(),
                   'params':{
                        'n_estimators': [100,200,300],
                        'learning_rate':[0.0001, 0.001, 0.01, 0.1, 1.0]
                   }
              },

              {
                   'classifier':RandomForestClassifier(),
                   'params' :{
                        'n_estimators': [25, 50, 100, 150], 
                        'max_features': ['sqrt', 'log2', None], 
                        'max_depth': [3, 6, 9], 
                        'max_leaf_nodes': [3, 6, 9], 
                   }    
              },

              {
                   'classifier' : LogisticRegression(),
                   'params' : {
                        'C':[0.001, 0.01, 0.1, 1, 10, 100],
                        'max_iter' : [100, 1000,2500, 5000],
                        'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                   }
              },

              {
                  'classifier' : SVC(),
                  'params' : {
                       'C': [0.1, 1, 10],
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                        'kernel': ['linear', 'rbf', 'poly']

                  }
              },

              {
                   'classifier':KNeighborsClassifier(),
                   'params':{
                        'n_neighbors': np.arange(1, 11),
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'p': [1, 2]
                   }
              },

              {
                   'classifier' : DecisionTreeClassifier(),
                   'params':{
                        'criterion':['gini','entropy'],
                        'max_depth':np.arange(1,21).tolist()[0::2],
                        'min_samples_split':np.arange(2,11).tolist()[0::2],
                        'max_leaf_nodes':np.arange(3,26).tolist()[0::2],
                   }
              },

            ]

            best_model_classifier =  model_evaluation(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model = models)

           
            
            logging.info('best model found...!')

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model_classifier
            )

            return best_model_classifier
        
        except Exception as error:
            raise UserException(error , sys)


