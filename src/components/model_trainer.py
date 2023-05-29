from src.logger import logging
from src.exception import CustomException
from src import utils
import pandas as pd
import numpy as np
import os, sys
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifact', 'model_trainer', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Model Training Phase Started')

            logging.info('Spliting independent and dependent features')
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            models = {
                'LogisticRegression' : LogisticRegression(), 'DecisionTreeClassifier' : DecisionTreeClassifier(), 'RandomForestClassifier' : RandomForestClassifier()
            }

            params = {
                "LogisticRegression":{
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                "DecisionTreeClassifier":{
                    "class_weight":["balanced"],
                    "criterion":['gini',"entropy","log_loss"],
                    "splitter":['best','random'],
                    "max_depth":[3,4,5,6],
                    "min_samples_split":[2,3,4,5],
                    "min_samples_leaf":[1,2,3],
                    "max_features":["auto","sqrt","log2"]
                },
                "RandomForestClassifier":{
                    "class_weight":["balanced"],
                    'n_estimators': [20, 50, 30],
                    'max_depth': [10, 8, 5],
                    'min_samples_split': [2, 5, 10],
                }
            }

            logging.info('Evaluating the model')
            model_report:dict = utils.evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f'Best Model found, Model Name is {best_model_name}, accuracy_score:{round(best_model_score,2)}')

            logging.info('Saving the best model in pickle format')
            utils.save_subject(self.model_trainer_config.trained_model_path, best_model)

            logging.info('Model Training Phase Completed')
            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)