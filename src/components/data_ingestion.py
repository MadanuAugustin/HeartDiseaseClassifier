
import os
import sys
from src.exception import UserException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.path.join('artifacts', 'raw_data.csv')

    train_data_path : str = os.path.join('artifacts', 'train_data.csv')
    
    test_data_path : str = os.path.join('artifacts', 'test_data.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info('started data ingestion...!')

        try:
            mydf = pd.read_csv('Data\\raw_data.csv')

            logging.info('Finished reading the raw data...!')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            mydf.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('splitting the data into training and testing...!')

            train_data, test_data = train_test_split(mydf, test_size = 0.2)

            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)

            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('train and test split completed...!')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as error:
            raise UserException(error, sys)


if __name__ == '__main__':
    dataingestion_obj = DataIngestion()
    train_data_path, test_data_path = dataingestion_obj.initiate_data_ingestion()

    datatransformation_obj = DataTransformation()
    train_arr, test_arr = datatransformation_obj.initiate_data_transformation(train_data_path, test_data_path)

    model_train_obj = ModelTrainer()
    print(model_train_obj.initiate_model_trainer(train_arr, test_arr))
    