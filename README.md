# House Price Predictor

A full-stack machine learning web application that predicts house prices based on key features like quality, living area, and garage capacity.

---

## Live Demo

👉 Frontend: https://house-price-predictor-sigma-one.vercel.app  
👉 Backend API: https://house-price-predictor-fejb.onrender.com/docs  

---

## Features

- Predict house prices instantly
- Clean and modern UI
- Fast API backend
- Real-time predictions
- Fully deployed (Vercel + Render)

---

## Tech Stack

### 🔹 Machine Learning
- XGBoost
- Scikit-learn
- Pandas, NumPy

### 🔹 Backend
- FastAPI
- Uvicorn

### 🔹 Frontend
- HTML
- CSS
- JavaScript (Fetch API)

### 🔹 Deployment
- Vercel (Frontend)
- Render (Backend)

---

## Model Details

- Algorithm: XGBoost Regressor
- Dataset: Ames Housing Dataset (Kaggle)
- Features used:
  - Overall Quality
  - Living Area (sq ft)
  - Garage Capacity

---

##  How It Works

1. User enters house details in the UI
2. Frontend sends request to FastAPI backend
3. Backend processes input and runs ML model
4. Prediction is returned and displayed instantly

---

##  Installation (Local Setup)

```bash
git clone https://github.com/nishmitha1112/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
uvicorn app:app --reload
