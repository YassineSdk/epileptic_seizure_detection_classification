from fastapi import FastAPI , UploadFile , HTTPException , File
from helpers import validate_datafile, Structure_validation
from config_logger import get_logger 


logger = get_logger()
app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Healthy endpoint"}




@app.post("/inference")
async def predict_class(
    datafile : UploadFile = File(...)
):
    logger.info("a prediction request  recieved:")
    extention = validate_datafile(datafile)
    data = Structure_validation(datafile, extention)
    data.shape[0]
    inference_type = None 
    if data.shape[0] == 1 :
        inference_type = "unit"
    else :
        inference_type = "batch"
    logger.info(f"the data set is valid , starting the inference job {inference_type}")
    
    
    return {"datafile":datafile.filename,
            "data":data.to_dict()}
