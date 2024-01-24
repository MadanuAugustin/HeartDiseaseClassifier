
import os
import sys
import dill

from src.logger import logging
from src.exception import UserException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            
            dill.dump(obj, file_obj)


    except Exception as error:
        raise UserException(error, sys)
    


def model_evaluation(X_train, y_train, X_test, y_test, model):
    try:
        best_model = None
        best_score = 0.0

        for classifier_info in model:
            classifier = classifier_info["classifier"]
            param_grid = classifier_info["params"]

            grid_search = GridSearchCV(classifier, param_grid, cv = 5, scoring = 'accuracy')
            grid_search.fit(X_train, y_train)

            
            # model.fit(X_train,y_train)

            best_classifier = grid_search.best_estimator_
            best_classifier.set_params(**grid_search.best_params_)
            best_classifier.fit(X_train, y_train)
            test_model_accuracy = accuracy_score(y_test, best_classifier.predict(X_test))

            if test_model_accuracy > best_score:
                best_score = test_model_accuracy
                best_model = best_classifier
                # print(best_score, best_model)


            return  best_model, best_score #best_model.get_params()]
            

    except Exception as error:
        raise UserException(error, sys)
    



def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj: # opening the file_path in readbyte mode and loading the pkl file
            return dill.load(file_obj)
        
    except Exception as error:
        raise UserException(error, sys)