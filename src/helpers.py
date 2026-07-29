import pandas as pd 
from pathlib import Path
from fastapi import UploadFile, HTTPException
from config_logger import get_logger

allowed_extentions = [".csv", ".xls", ".xlsx"]
logger = get_logger()

def validate_datafile(file :UploadFile)-> str :
    """
    validate the the uploaded file is either and .csv  ,excel , json 
    """
    extention = Path(file.filename).suffix.lower()

    if extention not in allowed_extentions:
        logger.error("the data file format is not supported Only .csv, .xls, and .xlsx files are allowed ")
        raise HTTPException(
            status_code=400,
            detail="Only .csv, .xls, and .xlsx files are allowed."
        )
    logger.info("the data file is accepted")
    
    return extention


def Structure_validation(file,extention : str) ->pd.DataFrame:
    """
    this function validates the data structure 
        File type
        Not empty
        Exactly 16 features (or 16, depending on your model)
        No missing values
        Numeric values only
    """
    if extention == ".csv":
        data = pd.read_csv(file.file)
    else :
        data = pd.read_excel(file.file)
    
    # checking if the dataset is not empty 
    if data.empty :
        logger.error("the datafile is empty")
        raise HTTPException(
            status_code=400,
            detail="the datafile is empty"
        )
    
    
    # checking if the Unnamed column exist 
    unnamed_cols = data.columns[data.columns.str.startswith("Unnamed")]
    if len(unnamed_cols) > 0:
        logger.warn('the data has unnamed_cols that need further processing')
        data = data.drop(columns=unnamed_cols)


    # checking if the data has inconsistent number of columns
    if data.shape[1] != 16 :
        logger.error("The dataset must contain exactly 15 columns.")
        raise HTTPException(
            status_code=400,
            detail="The dataset must contain exactly 15 columns."
        )

    # checking if the dataset has any null values
    if data.isnull().values.any():
        logger.error("The dataset contains missing values.")
        raise HTTPException(
            status_code=400,
            detail="The dataset contains missing values."
        )
    
    return data 
    
    
    
    