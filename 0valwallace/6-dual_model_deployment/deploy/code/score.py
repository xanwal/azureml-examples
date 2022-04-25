from multiprocessing.sharedctypes import Value
import os
import logging
import json
from model import Handle

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION) 

    global handle
    handle = Handle(os.getenv("AZUREML_MODEL_DIR"))

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """

    logging.info("Request received")
    try:
        response = handle(raw_data)
        logging.info("Request processed")
        return response
    except ValueError as e:
        logging.info("Request failed")
        logging.info(str(e))
    
    
    