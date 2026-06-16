from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import joblib
from fastapi import FastAPI


app = FastAPI()

Instrumentator().instrument(app).expose(app)

model = joblib.load("model/model.joblib")

class IrisModel(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float

@app.post("/predict")
def predict(data : IrisModel):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {"prediction" : int(prediction[0])}



