
import wandb as wb 
import numpy as np 
import joblib 
import pandas 
from pathlib import Path
from config_logger import get_logger


logger = get_logger()


class ModelLoader:
    """
    """
    def __init__(
        self,
        model_path :str = Path(__file__).resolve().parent.parent / "models" / "stacking.joblib",
        artfact_name : str = "yassinedatascientist001-datalab/Epeliptic_seizure_classification/Stacking_v1:v0"
    ):
        self.model_path = Path(model_path)
        self.artfact_name = artfact_name
    
    def download_model(self):
        """
        Download the model from W&B
        """
        self.model_path.parent.mkdir(
            exist_ok=True
        )

        api=wb.Api()
        artifact = api.artifact(
            self.artfact_name,
            type="model",

        )
        artifact_dir = Path(
            artifact.download(
                root=str(self.model_path.parent)
            )
        )
        return self.model_path 
    
    def model_exist(self):
        return self.model_path.exists()

    def load_model(self):
        if not self.model_path.exists():
            logger.warn("the model not found , Downloading from W&B .")
            self.download_model()
        
        else :
            logger.info("using the cached model.")
        model = joblib.load(self.model_path)
        logger.info("model loaded successfully.")

        return model 


