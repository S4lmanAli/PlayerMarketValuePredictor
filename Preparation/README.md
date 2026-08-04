# Player Market Value Predictor (FC 26)

Predicting a football player's market value (`value_eur`) from attribute data in the FC 26 (FIFA 26) player database, using a machine learning approach.

**Phase:** 1 — Data Understanding & Preprocessing

## Project Structure

```
PlayerMarketValuePredictor/
├── Phase1.ipynb          # Data loading, cleaning, encoding, EDA, and preprocessing
├── FC26.csv               # Raw dataset (not included — see Dataset section below)
├── Updated.csv             # Cleaned/preprocessed dataset produced by Phase1.ipynb
├── Phase1_Report.pdf      # One-page Phase 1 report (problem statement, methodology, challenges)
└── README.md               # This file
```


## Dataset

- **Source:** [FC 26 (FIFA 26) Player Data — Kaggle](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data)
- Download `FC26.csv` from the link above and place it in the project root before running the notebook (the raw CSV is not committed to this repository due to size).

## How to Run

1. **Add the dataset**
   Download `FC26.csv` from the [Kaggle link](https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data) and place it in the project root, next to `Phase1.ipynb`.

2. **Launch the notebook**
   ```
   Run all cells in order (top to bottom). The notebook will:
   - Load and clean `FC26.csv`
   - Handle missing values and encode categorical features
   - Generate a correlation heatmap and remove outliers
   - Apply a log transform to the target variable
   - Save the final preprocessed dataset as `Updated.csv`

## Libraries & Tools

Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, imbalanced-learn, Jupyter Notebook.

## Documentation

See [`Phase1_Report.pdf`](./Phase1_Report.pdf) for the full write-up of the problem statement, dataset justification, preprocessing steps, and challenges encountered.
