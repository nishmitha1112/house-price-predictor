from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load trained model
model = joblib.load("model.pkl")

# input schema
class InputData(BaseModel):
    OverallQual: float
    GrLivArea: float
    GarageCars: float

# prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    arr = np.array([[data.OverallQual, data.GrLivArea, data.GarageCars]])
    prediction = model.predict(arr)[0]
    return {"price": float(prediction)}