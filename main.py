from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
# from app.models.model import predict_pipeline
from pydub import AudioSegment
import io

import uvicorn

app = FastAPI()
class PredictionOut(BaseModel):
    prediction: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "1.0"}

# @app.post("/predict")#, response_model=PredictionOut
# async def predict(file: UploadFile = File(...)):
#     # Read the content of the uploaded file
#     content = await file.read()
#     audio = AudioSegment.from_file(io.BytesIO(content))
#     # Call the prediction function with the audio content
#     result_from_model = predict_pipeline(audio)

#     return PredictionOut(prediction=result_from_model)



    
