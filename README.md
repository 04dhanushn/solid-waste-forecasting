# ♻️ Solid Waste Management and Forecasting

A data-driven forecasting project for predicting future waste generation using Machine Learning, Deep Learning, and Transfer Learning techniques.

---

## 📌 Project Overview

This project analyzes historical solid waste collection data and predicts future waste generation trends using multiple forecasting approaches.

The objective is to improve:

- Waste collection planning
- Resource allocation
- Operational efficiency
- Sustainable waste management decisions

Models were developed and compared using statistical and deep learning approaches.

---

## 🚀 Features

✔ Data preprocessing and cleaning  
✔ Outlier detection using IQR  
✔ Feature engineering for time-series forecasting  
✔ Linear Regression forecasting  
✔ LSTM forecasting with multiple optimizers  
✔ RNN forecasting  
✔ GRU forecasting  
✔ Hybrid LSTM–GRU forecasting  
✔ Transfer Learning for forecasting  
✔ Performance comparison using MAE and RMSE  
✔ Visualization of training and prediction results  

---

## 📊 Dataset

Dataset contains historical waste collection records.

### Features

| Feature | Description |
|---------|-------------|
| area | Waste collection area |
| ticket_date | Collection date |
| waste_type | Category of waste |
| net_weight_kg | Waste quantity |

---

## ⚙️ Data Processing

The following preprocessing techniques were applied:

- Date conversion
- Time-series sorting
- Aggregation by date
- Missing value handling
- Feature scaling
- Noise augmentation
- Lag features
- Moving averages
- IQR outlier treatment

---

## 🧠 Models Implemented

### 1. Linear Regression
Custom Gradient Descent implementation with gradient clipping.

### 2. LSTM
Optimizers:
- Adam
- SGD
- RMSprop
- Adagrad

### 3. RNN
Optimizers:
- Adam
- SGD
- RMSprop
- Adagrad

### 4. GRU
Optimizers:
- Adam
- SGD
- RMSprop
- Adagrad

### 5. Hybrid LSTM–GRU
Optimizers:
- Adam
- SGD
- RMSprop
- Adagrad

### 6. Transfer Learning (Hybrid LSTM–GRU)
Optimizers:
- Adam
- SGD
- RMSprop
- Adagrad

Pre-trained on:
2014–2017

Fine-tuned on:
2018

---

## 📈 Evaluation Metrics

Model performance was evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## 📉 Results

The forecasting models were compared using:

- Actual vs Predicted graphs
- Training Loss
- Validation Loss
- MAE Comparison
- RMSE Comparison

Transfer Learning improved forecasting performance compared to direct training.

---

## 📦 Requirements

```txt
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
```

---

## 🔮 Future Improvements

- Add Prophet forecasting
- Deploy using Streamlit
- Add explainable forecasting
- Add real-time prediction dashboard

---

## 👨‍💻 Author

Dhanush N

M.Tech – Artificial Intelligence & Machine Learning

Vellore Institute of Technology

---

⭐ If you found this project useful, consider giving it a star.
