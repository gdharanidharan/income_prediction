from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
import sys

def main():

    try:
        logging.info('Pipeline Started')
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
        logging.info('Pipeline Completed')

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()