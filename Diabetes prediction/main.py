from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

# ✅ CORS (IMPORTANT for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load trained model
model = joblib.load("diabetes_model.pkl")


# ✅ Input schema (validation included)
class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=1000)
    BMI: float = Field(..., ge=0, le=100)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: int = Field(..., ge=1, le=120)


# ✅ Home route
@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}


# ✅ Prediction route
@app.post("/predict")
def predict(data: PatientData):

    # Prepare input for model
    features = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]

    # Make prediction
    prediction = model.predict(features)[0]

    # Convert result
    if prediction == 1:
        result = "Diabetic"
        advice = "Consult a healthcare professional for further evaluation."
    else:
        result = "Not Diabetic"
        advice = "Maintain a healthy lifestyle and regular check-ups."

    return {
        "prediction": int(prediction),
        "result": result,
        "advice": advice
    }