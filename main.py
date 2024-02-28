import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 1000

weather_conditions = np.random.choice(['Clear', 'Rain', 'Snow'], size=num_samples)
road_types = np.random.choice(['Urban', 'Suburban', 'Rural'], size=num_samples)
speed_limits = np.random.choice([30, 50, 70], size=num_samples)
accident_severity = np.random.choice([1, 2, 3, 4, 5], size=num_samples)

# Create a DataFrame
df = pd.DataFrame({
    'weather_condition': weather_conditions,
    'road_type': road_types,
    'speed_limit': speed_limits,
    'accident_severity': accident_severity
})

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['weather_condition', 'road_type'])

# Display the first few rows of the dataset
print(df.head())

# Save the dataset to a CSV file
df.to_csv('road_accidents.csv', index=False)

# Load the synthetic dataset
df = pd.read_csv('road_accidents.csv')

# Explore the dataset
print(df.info())
print(df.describe())
print(df.head())

# Dependent Variable (Accident Severity)
y = df['accident_severity']

# Independent Variables
X = df[['weather_condition_Clear', 'weather_condition_Rain', 'weather_condition_Snow',
        'road_type_Urban', 'road_type_Suburban', 'road_type_Rural', 'speed_limit']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

import joblib

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')

# Hypothetical set of independent variables for prediction
new_data = pd.DataFrame({'weather_condition_Clear': [1], 'weather_condition_Rain': [0], 'weather_condition_Snow': [0],
                         'road_type_Urban': [1], 'road_type_Suburban': [0], 'road_type_Rural': [0], 'speed_limit': [50]})

prediction = model.predict(new_data)
print(f'Predicted Accident Severity: {prediction[0]}')
