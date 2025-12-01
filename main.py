from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import joblib
import os
from typing import Optional

app = FastAPI(
    title="Student Performance Prediction API",
    description="Öğrenci performans tahmin modeli için REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
MODEL_PATH = 'student_model.pkl'

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model başarıyla yüklendi: {MODEL_PATH}")
    else:
        print(f"Uyarı: Model dosyası bulunamadı: {MODEL_PATH}")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {str(e)}")

class StudentData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hours_studied": 7.5,
                "previous_scores": 85.0,
                "extracurricular": "Yes",
                "sleep_hours": 8.0,
                "sample_papers": 5
            }
        }
    )
    
    hours_studied: float = Field(..., ge=1, le=9, description="Günlük çalışma saati (1-9)")
    previous_scores: float = Field(..., ge=0, le=100, description="Önceki not ortalaması (0-100)")
    extracurricular: str = Field(..., description="Ders dışı aktiviteler (Evet/Hayır)")
    sleep_hours: float = Field(..., ge=4, le=9, description="Günlük uyku saati (4-9)")
    sample_papers: int = Field(..., ge=0, description="Çözülen örnek soru sayısı")

@app.get("/")
def root():
    return {
        "message": "Student Performance Prediction API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_score(data: StudentData):
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenemedi."
        )
    
    try:
        activity_score = 1 if data.extracurricular.lower() == "yes" else 0
        
        features = [[
            data.hours_studied,
            data.previous_scores,
            activity_score,
            data.sleep_hours,
            data.sample_papers
        ]]
        
        prediction = model.predict(features)
        final_score = float(prediction[0])
        
        if data.extracurricular.lower() == "yes":
            final_score = final_score - 5
        
        if final_score < 0:
            final_score = 0
        elif final_score > 100:
            final_score = 100
        
        return {
            "prediction": int(round(final_score)),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin yapılırken hata oluştu: {str(e)}"
        )

