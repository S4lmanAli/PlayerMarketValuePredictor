# Football Player Market Value Estimator

A lightweight Machine Learning app that estimates a football player's market value based on key performance metrics. Built as a course project for AI & Machine Learning, this project demonstrates an end-to-end ML workflow from preprocessing and training to an interactive UI with Streamlit.

---

## Features

- **Instant Predictions:** Uses pre-trained Joblib model files for quick inference without retraining.
- **Automated Preprocessing:** Applies saved scaling transformations (like `StandardScaler` on Age) seamlessly.
- **Interactive UI:** Simple, clean web interface powered by Streamlit.

---

## Stack

- **Language:** Python
- **ML Core:** Scikit-learn, Pandas, NumPy
- **Serialization:** Joblib
- **Interface:** Streamlit

---

## Project Structure

```text
PlayerMarketValuePredictor/Phase 2
├── app.py                    # Streamlit web app
├── train.py                  # Model training pipeline
├── utils.py                  # Helper & utility functions
├── requirements.txt          # Project dependencies
├── README.md                 # This file
├── Phase 2 Report            # PDF report of this phase
├── notebooks/Reference Notebook.ipynb  # Notebook which was the base to modularize the scripts
├── models/
│   ├── player_values.joblib  # Saved linear regression model
│   └── age_scaler.joblib     # Pre-fitted feature scaler
└── data/                     # Dataset directory
```

---

## How It Works

1. Input player parameters (**Age**, **Overall Rating**, and **Potential**).
2. The app scales numerical inputs and feeds them into the trained Linear Regression model.
3. Output provides a clear, formatted market value estimate.

---

## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch App**
   ```bash
   streamlit run app.py
   ```
