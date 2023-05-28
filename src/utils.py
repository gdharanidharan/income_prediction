import os, sys
from src.logger import logging
from src.exception import CustomException
import pickle


def save_subject(file_path, object):
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_object:
            pickle.dump(object, file_object)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_subject(file_path):
    try: 
        with open(file_path, 'rb') as file_object:
            pickle.load(file_object)
    
    except Exception as e:
        raise CustomException(e, sys)