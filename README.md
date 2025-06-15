# Power Consumption Trend Analysis for Energy Reduction

This project analyzes power consumption trends using an LSTM (Long Short-Term Memory) neural network to predict future power output based on various environmental factors. The goal is to understand and predict energy consumption patterns to identify opportunities for reduction and contribute to a greener environment.

## Project Description

The code performs the following steps:

1.  **Data Loading and Preprocessing:**
    *   Loads training and testing data from `train.csv` and `test.csv`.
    *   Calculates `power_output` from `voltage` and `current`.
    *   Handles missing and bad values by replacing them with `NaN` and then forward-filling.
    *   Scales the features using `MinMaxScaler`.

2.  **Sequence Creation:**
    *   Creates time sequences from the scaled features for use in the LSTM model. The `lookback` parameter determines the length of the historical data used for prediction.

3.  **Model Building:**
    *   Constructs a Sequential LSTM model with Dropout layers to prevent overfitting.
    *   Compiles the model using the Adam optimizer and Mean Squared Error (MSE) loss function.

4.  **Model Training:**
    *   Trains the LSTM model on the prepared training data.

5.  **Model Evaluation:**
    *   Evaluates the trained model using the test data.
    *   Calculates and prints the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and a custom "Score" based on the RMSE and the mean of the test data's target variable.

## Code Explanation

The code is organized into several cells:

*   **Cell 1:** Imports necessary libraries (pandas, numpy, matplotlib, sklearn, tensorflow).
*   **Cell 2:** Handles data loading and preprocessing for both training and testing datasets. It calculates power output, deals with missing values, converts data types, fills missing values, and scales the features. It also defines the `create_sequences` function.
*   **Cell 3:** Creates the input sequences `X` and target sequences `y` for the LSTM model from the scaled training features. It then splits the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).
*   **Cell 4:** Defines, compiles, and trains the LSTM model.
*   **Cell 5:** Evaluates the trained model by predicting on the test set and calculating MSE, RMSE, and a custom score.

## Getting Started

1.  Ensure you have the required libraries installed (tensorflow, pandas, numpy, scikit-learn, matplotlib).
2.  Have your training data in a CSV file named `train.csv` and test data in a CSV file named `test.csv` in the same directory as your notebook. The CSVs should contain columns for `voltage`, `current`, `module_temperature`, `cloud_coverage`, `wind_speed`, and `pressure`.
3.  Run the code cells sequentially in the provided notebook environment.

## Future Improvements

*   Explore different LSTM architectures and hyperparameters.
*   Incorporate more features that might influence power consumption.
*   Implement more sophisticated missing value imputation techniques.
*   Visualize the predicted power output against the actual power output.
*   Integrate the model into a real-time monitoring system for proactive energy management.

This project provides a solid foundation for analyzing and predicting power consumption, paving the way for more efficient energy usage and a positive impact on the environment.
