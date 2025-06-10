# smart-urban-planner

Creating a smart urban planner involves several complex components, including data collection, model training, and optimization. Given the scope and potential complexity, I'll provide a simplified version of such a system. This example will focus on simulating basic infrastructure planning using machine learning concepts in an educational manner. For a real-world application, consider integrating actual datasets and more robust ML models.

Below is a Python program that outlines the basic framework for an urban planner using a simple regression model to predict and optimize traffic flow.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# A function to simulate data collection
def collect_data():
    # Simulate a dataset representing road intersections and traffic volumes
    logging.info("Collecting data...")
    np.random.seed(0)
    data_size = 1000
    intersections = np.random.randint(1, 100, size=data_size)
    daily_traffic = np.random.randint(100, 5000, size=data_size)
    
    # Simulate some relationship
    congestion_index = 3.5 * intersections + 2 * np.sqrt(daily_traffic) + np.random.normal(0, 50, data_size)
    
    df = pd.DataFrame({
        'intersections': intersections,
        'daily_traffic': daily_traffic,
        'congestion_index': congestion_index
    })
    logging.info("Data collection completed.")
    return df

# A function to preprocess data
def preprocess_data(df):
    try:
        logging.info("Preprocessing data...")
        X = df[['intersections', 'daily_traffic']]
        y = df['congestion_index']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data preprocessing completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        raise

# A function to train a machine learning model
def train_model(X_train, y_train):
    try:
        logging.info("Training model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise

# A function to evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Evaluating model...")
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Mean Squared Error: {mse}")
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")

# A function for optimization recommendations
def optimize_infrastructure(model, intersections, daily_traffic):
    try:
        logging.info("Optimizing infrastructure...")
        
        # Predict congestion index
        predicted_congestion = model.predict(np.array([[intersections, daily_traffic]]))
        logging.info(f"Predicted congestion index for intersections: {intersections} and daily traffic: {daily_traffic}: {predicted_congestion[0]}")
        
        # Provide a simple recommendation
        if predicted_congestion[0] > 300:
            recommendation = f"Consider increasing lanes or adding public transport options for {intersections} intersections."
        else:
            recommendation = f"Current infrastructure for {intersections} intersections is sufficient."
        
        logging.info(f"Recommendation: {recommendation}")
        return recommendation
    
    except Exception as e:
        logging.error(f"An error occurred during optimization: {e}")
        raise

def main():
    try:
        # Collect and prepare the data
        df = collect_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

        # Sample optimization
        optimize_infrastructure(model, intersections=50, daily_traffic=2000)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
```

**Key Components Explained:**

- **Data Collection**: Simulated using random data; this would ideally be replaced with real traffic and infrastructure data.
- **Model Training**: Uses a simple linear regression model to demonstrate the concept.
- **Optimization**: Provides basic recommendations based on the predicted congestion index.
- **Error Handling**: Each function includes try-except blocks to handle and log errors gracefully.

Remember, actual urban planning tools would require much more data, sophisticated models, and collaboration with experts.