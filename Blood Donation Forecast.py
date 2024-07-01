import pandas as pd

# Load dataset
data = pd.read_csv('blood_donation_data.csv')

# Convert donation_date to datetime
data['donation_date'] = pd.to_datetime(data['donation_date'])

# Sort by donor_id and donation_date
data = data.sort_values(by=['donor_id', 'donation_date'])

# Create additional features
data['days_since_last_donation'] = data.groupby('donor_id')['donation_date'].diff().dt.days
data['donation_frequency'] = data.groupby('donor_id')['donation_date'].transform(lambda x: x.diff().mean().days)

# Handle missing values for the first donation (no previous donation)
data['days_since_last_donation'].fillna(0, inplace=True)
data['donation_frequency'].fillna(data['donation_frequency'].mean(), inplace=True)

# Fill in the target variable for simplicity in this example
data['donated_in_next_month'].fillna(0, inplace=True)

# Drop rows with NaN in target variable
data.dropna(subset=['donated_in_next_month'], inplace=True)

print(data.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Feature selection
features = ['days_since_last_donation', 'donation_frequency']
X = data[features]
y = data['donated_in_next_month']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Prepare data for Prophet
donation_history = data[['donation_date', 'donation_amount']].groupby('donation_date').sum().reset_index()
donation_history.rename(columns={'donation_date': 'ds', 'donation_amount': 'y'}, inplace=True)

# Initialize and train the model
model = Prophet()
model.fit(donation_history)

# Make future predictions
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.show()
