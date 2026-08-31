# Titanic Survival Prediction REST API
## Live Deployment

The API is deployed and publicly accessible on Render.

**Live API:** https://titanic-ml-rest-api.onrender.com

**Swagger API Documentation:** https://titanic-ml-rest-api.onrender.com/docs

## API Endpoint

### POST /predict

The `/predict` endpoint accepts passenger information in JSON format and returns a Titanic survival prediction.

Example request:

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 80,
  "embarked": "C"
}

Example response:

{
  "prediction_id": "example-id",
  "prediction": 1,
  "prediction_label": "Survived",
  "probability": {
    "did_not_survive": 0.12,
    "survived": 0.88
  }
}

## Project Overview

This project deploys a trained Machine Learning model as a REST API using FastAPI.

The API accepts passenger information in JSON format and predicts whether the passenger is likely to survive the Titanic disaster.

## Objective

The objective is to demonstrate how a trained Machine Learning model can be exposed as a REST API for use by applications and business systems.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Postman

## Machine Learning Model

A Logistic Regression classifier is used for binary classification.

Target variable:

`survived`

Where:

- 0 = Did not survive
- 1 = Survived

## Features

The model uses:

- pclass
- sex
- age
- sibsp
- parch
- fare
- embarked

## Machine Learning Pipeline

The preprocessing and model are combined into a single Scikit-learn Pipeline.

### Numerical preprocessing

- Missing-value imputation using median
- StandardScaler

### Categorical preprocessing

- Missing-value imputation using most frequent value
- One-hot encoding
- Unknown categories are handled safely

### Model

- Logistic Regression

The complete pipeline is saved using Joblib.

## Project Structure

```text
titanic-ml-rest-api/
│
├── data/
│   └── titanic.csv
│
├── model/
│   └── titanic_pipeline.joblib
│
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
Model Training

Run:

python train_model.py

The trained pipeline is saved as:

model/titanic_pipeline.joblib
Running the API

Install dependencies:

pip install -r requirements.txt

Start the API:

python -m uvicorn app:app --reload

The API runs at:

http://127.0.0.1:8000
API Documentation

FastAPI automatically provides Swagger documentation at:

http://127.0.0.1:8000/docs
API Endpoints
Health Check
GET /

Checks whether the API is running.

Model Information
GET /model-info

Returns information about the deployed model.

Prediction
POST /predict

Accepts passenger information and returns a survival prediction.

Example Request
{
    "pclass": 3,
    "sex": "male",
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S"
}
Example Response
{
    "prediction_id": "example-id",
    "prediction": 0,
    "prediction_label": "Did not survive",
    "probability": {
        "did_not_survive": 0.91,
        "survived": 0.09
    }
}
Input Validation

The API validates:

Passenger class range
Age range
Non-negative fare
Non-negative family counts
Valid gender categories
Valid embarkation categories
Required fields

Invalid requests return HTTP 422 validation errors.

Testing

The API was tested using:

FastAPI Swagger UI
Postman
Valid JSON requests
Invalid passenger class
Invalid gender
Negative age
Missing fields
Deployment Concepts Demonstrated
Model serialization
REST API
JSON input/output
Input validation
Saved preprocessing pipeline
Model loading at startup
Health checks
Model metadata
Error handling
API documentation
API testing
Monitoring Considerations

After deployment, important metrics include:

API response time
Request volume
Error rate
Prediction distribution
Data drift
Model performance
Resource utilization
Future Improvements

Possible production improvements include:

Docker containerization
Authentication
HTTPS
Rate limiting
Centralized logging
Model versioning
CI/CD
Cloud deployment
Monitoring and alerting
Conclusion

This project demonstrates how a trained Machine Learning model can be packaged with its preprocessing pipeline and exposed through a validated REST API using FastAPI.


---

# 🔍 STEP 25 — Final folder check

Your Explorer should now look like:

```text
TITANIC-ML-REST-API
│
├── 📁 data
│   └── 📄 titanic.csv
│
├── 📁 model
│   └── 📄 titanic_pipeline.joblib
│
├── 📄 app.py
├── 📄 train_model.py
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore