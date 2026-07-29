import wandb as wb 
import numpy as np 
import joblib 
import pandas as pd


CLASS_NAMES=["healthy","generalize","focal","event"]
def predict(model ,data:pd.DataFrame)-> str :
    """
    """
    CLASS_NAMES=["healthy","generalize","focal","event"]
    
    X = data.to_numpy()
    
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    pred = model.predict(X)

    #single prediction
    
    if len(pred) == 1 :
        return CLASS_NAMES[pred[0]]
    
    else :
        CLASS_NAMES = np.array(CLASS_NAMES)
        return CLASS_NAMES[pred].tolist()