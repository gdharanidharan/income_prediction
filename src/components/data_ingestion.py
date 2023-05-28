from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os, sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifact', 'data_ingestion', 'train.csv')
    test_data_path : str = os.path.join('artifact', 'data_ingestion', 'test.csv')
    raw_data_path : str = os.path.join('artifact', 'data_ingestion', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion phase started')

            logging.info('Reading data from original path')
            data = pd.read_csv(r'data\income_cleandata.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train and Test Split')
            train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

            logging.info('Preparing train data')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            logging.info('Preparing test data')
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion phase completed')

            return (self.ingestion_config.train_data_path, 
                    self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)