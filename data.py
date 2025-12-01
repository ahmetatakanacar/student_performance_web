import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

def train_model():
    
    csv_file = "student_performance.csv"
    if not os.path.exists(csv_file):
      raise FileNotFoundError(f"Veri dosyası bulunamadı: {csv_file}")
        
    df = pd.read_csv(csv_file)
    
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    
    missing_values = df.isnull().sum()
    if missing_values.any():
      df = df.dropna()
    
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]
    
    print(f"Feature: {list(X.columns)}")
    print(f"Ortalama: {y.mean():.2f}")
    print(f"Standart sapma: {y.std():.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, 
      test_size=0.3, 
      random_state=42
    )
    print(f"Eğitim seti: {len(X_train)} kayıt")
    print(f"Test seti: {len(X_test)} kayıt")

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"R2 Score: {train_r2:.4f}")
    print(f"R2 Score (Test): {test_r2:.4f}")
    print(f"Mean Absolute Error: {test_mae:.4f}")
    print(f"Root Mean Squared Error: {test_rmse:.4f}")
    
    coeff_df = pd.DataFrame({
      'feature': X.columns,
      'coefficient': model.coef_,
      'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    for _, row in coeff_df.iterrows():
      print(f"  {row['feature']:30s}: {row['coefficient']:8.4f}")
    
    model_filename = 'student_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model kaydedildi: {model_filename}")
    return model, coeff_df

if __name__ == "__main__":
    try:
      model, feature_importance = train_model()
    except Exception as e:
      print(f"\nHata: {str(e)}")
      raise

