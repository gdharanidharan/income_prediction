import os, sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def save_subject(file_path, object):
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_object:
            pickle.dump(object, file_object)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try: 
        with open(file_path, 'rb') as file_object:
            return pickle.load(file_object)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            GS = GridSearchCV(model, param, cv=5)
            GS.fit(X_train, y_train)

            model.set_params(**GS.best_params_)
            model.fit(X_train, y_train)

            # make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred)
            # recall = recall_score(y_test, y_pred)

            report[list(models.values())[i]] = accuracy

            return report

    except Exception as e:
        raise CustomException(e, sys)