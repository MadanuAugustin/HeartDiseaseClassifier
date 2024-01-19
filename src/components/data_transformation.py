import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from src.utils import save_object

from src.exception import UserException
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:

    preprocessor_path = os.path.join('preprocessor_pkl_file', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def preprocessor_fun(self):
        try:
            categorical_columns = ['cp', 'restecg', 'thal']

            numerical_columns = ["age","sex","trestbps","chol","fbs","thalach","exang","oldpeak","slope","ca"]

            logging.info('creating numeric and categoric pipelines...!')

            numeric_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('robust_scaler', RobustScaler(with_centering=False))
                ]
            )

            categoric_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('robustscaler', RobustScaler(with_centering=False))
                ]
            )

            logging.info('Finished creating numeric and categoric pipelines...!')

            logging.info('creating preprocessor pipelines...!')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', numeric_pipeline, numerical_columns),
                    ('cat_pipeline', categoric_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        

        except Exception as error:
            raise UserException(error, sys)
        

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info('reading train and test data...!')

            train_df = pd.read_csv(train_data_path)
            
            test_df = pd.read_csv(test_data_path)

            logging.info('completed reading train and test data...!')

            logging.info('obtaining preprocessor object...!')

            preprocessor_obj = self.preprocessor_fun()

            target_column_name = 'target'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('applying preprocessing object on train and testing dataframe...!')

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] 

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            train_arr_df  = pd.DataFrame(train_arr)

            test_arr_df = pd.DataFrame(test_arr)

            train_arr_df.to_csv('artifacts\\transformed_train_data.csv', header=True, index = False)

            test_arr_df.to_csv('artifacts\\transformed_test_data.csv', header=True, index = False)

            logging.info('completed preprocessing on train and testing dataframes...!')

            logging.info('saving the preprocessor object as a pickle file...!')

            save_object(
                file_path = self.data_transformation_config.preprocessor_path,
                obj = preprocessor_obj
            )
            
            logging.info('successfully saved the preprocessor object as pickle file...!')

            return(
                train_arr, test_arr
            )

        except Exception as error:
            raise UserException(error, sys)
