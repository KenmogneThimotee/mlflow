import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    EXPERIMENT_NAME = "demo"
    EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME)

    boston_data = pd.read_csv('boston.csv')

    y = boston_data.MEDV
    X = boston_data.drop(['MEDV'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    model = LinearRegression(fit_intercept=False)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        
        mlflow.log_param("Intercept", model.intercept_)
        mlflow.log_param("Coefficient", model.coef_)
        
        predictions = model.predict(X_test)
        signature = infer_signature(X_test, predictions)

        # mlflow.log_param("Prediction Test set", predictions)
        
        mae = mean_absolute_error(y_test, predictions)
        mlflow.log_metric("mae", mae)
        
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mlflow.log_metric("rmse", rmse)
        
        rmsle = np.log(np.sqrt(mean_squared_error(y_test, predictions)))
        mlflow.log_metric("rmsle", rmsle)
        
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model", signature=signature)

        fig, ax = plt.subplots()
        ax.plot(predictions, label='Predictions') 
        ax.plot(y_test.to_numpy(), label='Actual values')
        ax.set_xlabel('Count')
        ax.set_ylabel('Values [MEDV]')
        ax.legend()
        ax.set_title('Actual vs Predicted Values')
        #plt.show()
        mlflow.log_figure(fig, "figure.png")

        print('Mean Absolute Error')
        print('MAE: ', mae)

        print('\nMean Squared Error')
        print('MSE: ', mse)

        print('\nRoot Mean Squared Error')
        print("RMSE: ", rmse)

        print('\nRoot Mean Squared Log Error ')
        print("RMSLE: ", rmsle)

    model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
    mv = mlflow.register_model(model_uri, "linearRegressionModel")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))