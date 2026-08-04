from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from preprocess import preprocess_pipeline
from utils import format_value
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import pandas as pd
import joblib

DATA_PATH = "data/FC26.csv"


def train_model():
    
    df_final = preprocess_pipeline(DATA_PATH)


    X = df_final.drop(columns = ["log_value"])
    y = df_final["log_value"]
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    actual_values  = np.exp(y_test)-1 #convert back to Euros
    actual_values_display = pd.Series(np.vectorize(format_value)(actual_values), index = x_test.index, name = "Actual Value")

    #train model
    model = LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    #convert predictions to euros and display resulsts
    lr_predicted_values = np.exp(y_pred)-1
    lr_error = lr_predicted_values - actual_values
    lr_results = pd.concat([
        x_test,
        actual_values_display,
        pd.Series(np.vectorize(format_value)(lr_predicted_values), index=x_test.index, name="Predicted Value"),
        pd.Series(np.vectorize(format_value)(np.abs(lr_error)), index=x_test.index, name="Prediction Error")
    ], axis=1)
    print("Test results:")
    print(lr_results.head())

    #Evaluation Metrics
    print("*"*35)
    print("For our linear regression model, the evaluation metrics are:")
    print("R2 score is: ", r2_score(actual_values,lr_predicted_values))
    print("Mean Absolute Error: ", mean_absolute_error(actual_values,lr_predicted_values))
    print("Mean Squared Error is: ", mean_squared_error(actual_values,lr_predicted_values))
    print("Root Mean Squared Error is: ", np.sqrt(mean_squared_error(actual_values,lr_predicted_values)))
    print("*"*35)

    save_path = (r"models/player_values.joblib")
    joblib.dump(model, save_path)
    print(f"Model saved successfully to {save_path} ")

if __name__ == "__main__":
    train_model()