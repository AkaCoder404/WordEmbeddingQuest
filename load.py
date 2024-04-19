"""
Title: Helper functions for downloading embeddings and datasets
"""

import json

               
class Loader():
    """
    Load a corpus from a path
    """
    def __init__(self):
        """
        Initialize the loader
        """
    
    def load_txt(self, path):
        with open(path, 'r') as file:
            data = file.readlines()
        
        return data
    
    def load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        
        return data    
        
        

class Downloader():
    """
    Download Datasets
    """

