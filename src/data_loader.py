import roboflow
import dotenv
import os
from pathlib import Path

class DataLoader:
    def __init__(self, config):
        dotenv.load_dotenv()
        self.config = config
        self.rf = roboflow.Roboflow(api_key=os.getenv("API_KEY"))
        self.project = self.rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])
        self.version = config['roboflow']['version']
        self.model_format = config['roboflow']['model_format']
        self.data_path = config['data']['path']

    def download_dataset(self, location=None):
        # if no location given, default to repo-root/data

        version = self.project.version(self.version)
        dataset = version.download(self.model_format)
        return dataset

