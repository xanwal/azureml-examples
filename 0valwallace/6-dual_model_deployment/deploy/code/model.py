import abc
from multiprocessing.sharedctypes import Value
from PIL import Image
import numpy as np
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as k
import json

class Model(object):
    def __init__(self, models_directory):
        self.models_directory = models_directory
        self.load_model(models_directory)
    
    @abc.abstractmethod
    def load_model(self, models_directory):
        pass 

    @abc.abstractmethod
    def score(self, payload): 
        return

class ImgModel(Model):
    def payload_to_numpy(self, payload, resize_to=(32,32)):
        #return np.asarray(Image.open(payload))[None, :]
        payload = np.asarray(payload)
        if payload.shape != resize_to:
            img = Image.fromarray(payload.astype(np.uint8))
            payload = np.asarray(img.resize(size=resize_to))
        return payload[None, :]

    def load_model(self, models_directory):
        self.model = k.models.load_model(Path(models_directory) / 'model' / self.category)

    def score(self, payload):
        payload = self.payload_to_numpy(payload)
        return self.model.predict(payload)

class Horses(ImgModel):
    def __init__(self, models_directory):
        self.category = 'horses'
        super().__init__(models_directory)


class Cats(ImgModel):
    def __init__(self, models_directory):
        self.category = 'cats'
        super().__init__(models_directory)

class Handle(object):
    def __init__(self, azureml_model_dir):
        self.azureml_model_dir = azureml_model_dir
        self.models = {'cats' : Cats(azureml_model_dir), 
                       'horses' : Horses(azureml_model_dir)}
        self.valid_categories = set(self.models.keys())             

    def __call__(self, raw_data):
        raw_data = json.loads(raw_data)
        try:
            image = raw_data["image"]
            category = raw_data["category"]
        except KeyError:
            raise ValueError("Request must contain fields 'image' and 'category'")
        if category in self.valid_categories:
            return self.models[category].score(image).tolist()
        else:
            raise ValueError("No model for category")
        