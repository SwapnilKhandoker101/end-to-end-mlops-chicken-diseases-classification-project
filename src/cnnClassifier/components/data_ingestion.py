import os
import kaggle
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from kaggle.api.kaggle_api_extended import KaggleApi
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config

    def download_file(self):
        os.environ['KAGGLE_USERNAME']=os.environ.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY']=os.environ.get('KAGGLE_KEY')

        if not os.path.exists(self.config.local_data_file):
            try:
                api=KaggleApi()
                api.authenticate()

                api.dataset_download_files(
                    self.config.kaggle_dataset,
                    path=self.config.root_dir,
                    unzip=True
                )

                logger.info(f"Dataset downloaded to {self.config.root_dir}")
            except Exception as e:
                logger.exception(f"An error occurred while downloading the dataset: {e}")

        else:
           logger.info(f"File already exists at {self.config.local_data_file}") 

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Dataset extracted to {unzip_path}")

