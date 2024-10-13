import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from src.model_functions import StackedModel
from src.feature_engineering import feature_engineering_production
import os
import pandas as pd
import json

# setting the fast api characteristics
tags_metadata = [{"name": "home-credit-default-api", "description": "Default Risk Prediction Model"}]
app = FastAPI(title = "Bank Transaction API",
              description = "Default Prediction Model",
              version = "1.0",
              contact = {"name": "Artur"},
              openapi_tags = tags_metadata)


class PredictionInput(BaseModel):
    data: str

# ==========================
# Here, we specify the model we want to use that is located
# in the artifacts folder. For version control, we have the date
# of the training 
model_specification = "20241012"
# ==========================

pickle_path = os.path.join('artifacts', f"model_v{model_specification}.pkl")

#loading the model in pickle format
def load_model():
    with open(pickle_path, 'rb') as model:
        clf_model = pickle.load(model)
    return clf_model

clf_model = load_model()

# setting get method for root
@app.get("/")
def message():
    text = "API for default risk prediction by Home Credit's clients"
    return text

# setting post method - predict using the model
@app.post("/predict", tags = ["Default_Risk"])
async def predict(input_data: PredictionInput):

    data_records = json.loads(input_data.data)
    X_data = pd.DataFrame(data_records)

    features = json.load(open('artifacts/feature_types.json'))

    for col in X_data.columns:
        if col in features["numeric_features"]:
            X_data[col] = pd.to_numeric(X_data[col], errors='coerce')

    X_data = feature_engineering_production(X_data)
    
    y_proba, y_pred = StackedModel(X_data).prediction_w_model(clf_model)

    response = {
                "Probability": [round(x, 4)*100 for x in list(y_proba)],
                "Classification": ["Yes" if x == 1 else "No" for x in list(y_pred)],
               }
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3000)