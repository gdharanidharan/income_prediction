# Handle Missing value
# Outliers treatment
# Handle Imblanced data
# Convert categorical columns into numerical columns

from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src import utils

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'data_transformation', 'preprocessor.pkl')
    

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        
        try:
            features = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
                        'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                        'hours_per_week', 'native_country']
            
            pipeline = Pipeline(steps=[
                ('simple_imputer', SimpleImputer(strategy='median')),
                ('standard_scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('pipeline', pipeline, features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException
        

    def remove_outliers_IQR(self, column, df):
        try:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            IQR = q3 - q1
            lower_limit = q1 - 1.5 * IQR
            upper_limit = q1 + 1.5 * IQR
            
            df.loc[(df[column] > upper_limit), column] = upper_limit
            df.loc[(df[column] < lower_limit), column] = lower_limit

            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            
            logging.info('Data Transformation phase started')
            logging.info('Reading train and test data')
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            features = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
                        'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                        'hours_per_week', 'native_country']
            
            logging.info('Handling Outliers')
            for column in features:
                self.remove_outliers_IQR(column, train_data)

            logging.info('Handling Outliers completed')

            target_column = 'income'
            drop_columns = [target_column]

            input_features_train_data = train_data.drop(drop_columns, axis=1)
            target_feature_train_data = train_data[target_column]

            input_features_test_data = test_data.drop(drop_columns, axis=1)
            target_feature_test_data = test_data[target_column]

            logging.info('Handling missing values and scaling data')
            preprocessor_object = self.get_data_transformation_object()
            input_train_arr = preprocessor_object.fit_transform(input_features_train_data)
            input_test_arr = preprocessor_object.transform(input_features_test_data)

            train_arr = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_arr = np.c_[input_test_arr, np.array(target_feature_test_data)]

            logging.info('saving preprocessor object')
            utils.save_subject(self.data_transformation_config.preprocessor_obj_file_path, preprocessor_object)

            logging.info('Data Transformation phase completed')
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)
