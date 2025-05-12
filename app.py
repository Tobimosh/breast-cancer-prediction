from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("model/svc_top10_model.pkl", "rb") as f:
    model = pickle.load(f)

# Static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Input model
class InputFeatures(BaseModel):
    mean_perimeter: float
    worst_texture: float
    mean_texture: float
    worst_perimeter: float
    mean_radius: float
    worst_radius: float
    worst_area: float
    mean_area: float
    mean_smoothness: float
    worst_smoothness: float

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(features: InputFeatures):
    input_data = np.array([[getattr(features, f) for f in features.__annotations__]])
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
