
import os
import sys
import dill

from src.logger import logging
from src.exception import UserException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            
            dill.dump(obj, file_obj)


    except Exception as error:
        raise UserException(error, sys)