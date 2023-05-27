from flask import Flask
from src.logger import logging
from src.exception import CustomException
import sys

app = Flask(__name__)

@app.route('/', methods=['get', 'post'])
def index():
    try:
        logging.info('testing')
        return 'hello world'
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(debug=True)