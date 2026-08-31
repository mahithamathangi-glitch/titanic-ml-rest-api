from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import uuid


# ==================================================
# 1. LOAD SAVED MODEL
# ==================================================

model = joblib.load("model/titanic_pipeline.joblib")


# ==================================================
# 2. CREATE FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="REST API for predicting Titanic passenger survival",
    version="1.0.0"
)


# ==================================================
# 3. INPUT VALIDATION SCHEMA
# ==================================================

class Passenger(BaseModel):

    pclass: int = Field(
        ...,
        ge=1,
        le=3,
        description="Passenger class: 1, 2, or 3"
    )

    sex: Literal["male", "female"]

    age: float = Field(
        ...,
        ge=0,
        le=100
    )

    sibsp: int = Field(
        ...,
        ge=0
    )

    parch: int = Field(
        ...,
        ge=0
    )

    fare: float = Field(
        ...,
        ge=0
    )

    embarked: Literal["C", "Q", "S"]


# ==================================================
# 4. HEALTH CHECK ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Titanic Survival Prediction API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


# ==================================================
# 5. MODEL INFORMATION ENDPOINT
# ==================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "Logistic Regression",
        "dataset": "Titanic",
        "task": "Binary Classification",
        "target": "survived",
        "version": "1.0.0"
    }


# ==================================================
# 6. PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
def predict(passenger: Passenger):

    try:

        # Convert validated request to DataFrame
        input_data = pd.DataFrame(
            [passenger.model_dump()]
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities
        probability = model.predict_proba(input_data)[0]
        prediction_id = str(uuid.uuid4())
        return {
    "prediction_id": prediction_id,

    "prediction": int(prediction),

    "prediction_label": (
        "Survived"
        if prediction == 1
        else "Did not survive"
    ),

    "probability": {
        "did_not_survive": round(
            float(probability[0]), 4
        ),
        "survived": round(
            float(probability[1]), 4
        )
    }
}

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )