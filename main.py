from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import sys

def main():

    try:
        print('Pipeline Started')
        logging.info('Pipeline Started')
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info('Pipeline Completed')
        print('Pipeline Completed')

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()