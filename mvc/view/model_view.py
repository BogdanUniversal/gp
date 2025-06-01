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


def getModels(user_id):
    try:
        models = Model.query.filter_by(user_id=user_id).all()
        return models
    except Exception:
        return None


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
    
    
def getPerformance(model_id):
    try:
        model = Model.query.filter_by(id=model_id).first()
        
        if not model or not model.resources_path:
            return None
        
        fig_performance = os.path.join(model.resources_path, "fig_performance.html")
        fig_profile = os.path.join(model.resources_path, "fig_profile.html")
        fig_single_explanation = os.path.join(model.resources_path, "fig_single_explanation.html")
        
        if not os.path.exists(fig_performance) or not os.path.exists(fig_profile) or not os.path.exists(fig_single_explanation):
            return None
        
        with open(fig_performance, "r") as file:
            fig_performance_data = file.read()
            
        with open(fig_profile, "r") as file:
            fig_profile_data = file.read()
            
        with open(fig_single_explanation, "r") as file:
            fig_single_explanation_data = file.read()
        
        return fig_performance_data, fig_profile_data, fig_single_explanation_data
    except Exception:
        return None