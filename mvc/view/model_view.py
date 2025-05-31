from functools import partial
import os
import threading
from genetic_programming.gp_algorithm import algorithm
from genetic_programming.primitive_set_gp import *
from genetic_programming.terminal_set_gp import *
from genetic_programming.general_set import *
from mvc.model.parameters_cache import parameters_cache
from mvc.model.dataset_cache import dataset_cache
from mvc.model.model import Model
from extensions import db
# from flask import current_app


def getTerminalsPrimitives():
    return PRIMITIVES + TERMINALS


def createModel(user_id, dataset_id, model_name):
    try:
        parameters = parameters_cache.get(str(user_id))
        
        if not parameters or not dataset_id or not model_name:
            return None

        model = Model(
            user_id=user_id,
            dataset_id=dataset_id,
            model_name=model_name,
            parameters=parameters,
        )
        
        db.session.add(model)
        db.session.commit()
        
        datasetCached = dataset_cache.get(str(model.user_id))
        
        training_thread = threading.Thread(target=partial(algorithm, model, datasetCached))
        training_thread.daemon = True
        training_thread.start()
        training_thread.join()  # Wait for the thread to finish

        model.setResourcesPath(f"models/{model.id}")
        
        db.session.add(model)
        db.session.commit()
        
        return model
    except Exception:
        return None


# def algorithm_wrapper(model_id, datasetCached):
#     """Wrapper to handle database session in the thread"""
#     # Create app context for the thread
#         try:
#             # Get a fresh copy of the model in this thread's session
#             model = Model.query.get(model_id)
#             if model:
#                 # Run the algorithm with the fresh model
#                 algorithm(model, datasetCached)
#             else:
#                 print(f"Model with ID {model_id} not found")
#         except Exception as e:
#             print(f"Error in training thread: {str(e)}")