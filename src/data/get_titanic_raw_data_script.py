
## write the script to the file
##Import
import os
from dotenv import load_dotenv, find_dotenv
import requests
from requests import session
import logging


###Pay load
payload ={
    "action":"login",
    "username": os.environ.get("KAGGLE_USERNAME"),
    "password": os.environ.get("KAGGLE_PASSWORD")
}

def extract_data(url, file_path):
    """
    Method to extract data from kaggle site
    """
    with session() as c:
        c.post("https://www.kaggle.com/account/login", data=payload)
        with open(file_path, 'w') as f:
            response = c.get(url, stream = True)
            for data in response.iter_content(1024):
                f.write(data.decode("utf-8"))


                
### Main method

def main(project_dir):
    """
     Main method
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading the raw data ..............")
    
    #urls
    ## URLS
    train_url = "https://www.kaggle.com/c/titanic/download/train.csv"
    test_url = "https://www.kaggle.com/c/titanic/download/test.csv"
    
    ## Csv file paths    
    raw_data_path = os.path.join(project_dir, "data", "raw")
    train_csv = os.path.join(raw_data_path, "train.csv")
    test_csv = os.path.join(raw_data_path, "test.csv")
    
    ## extract data
    extract_data(train_url, train_csv)
    extract_data(test_url, test_csv)
    logger.info("Downloaded raw data files ")

    
## Cal the main

if __name__ == '__main__':
    ##get the root directory    
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    ## setup logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    ###load the env variable
    dot_env_path = find_dotenv()
    load_dotenv(dot_env_path)
    
    ##call main
    main(project_dir)
    
