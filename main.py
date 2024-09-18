# thyroid_bayesian_network.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Function to load and preprocess the dataset
def load_data(url):
    columns = ['Class', 'T3', 'T4', 'TSH']
    data = pd.read_csv(url, delim_whitespace=True, names=columns)

    # Relabel the 'Class' column for better interpretability
    label_mapping = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
    data['Class'] = data['Class'].map(label_mapping)

    return data

# Function to split the data into training and testing sets
def split_data(data):
    features = data.drop(columns=['Class'])
    labels = data['Class']
    return train_test_split(features, labels, test_size=0.3, random_state=42)

# Function to define the Bayesian Network structure and train the model
def train_model(train_data, train_labels):
    train_data['Class'] = train_labels
    structure = [('T3', 'Class'), ('T4', 'Class'), ('TSH', 'Class')]
    model = BayesianNetwork(structure)
    model.fit(train_data, estimator=MaximumLikelihoodEstimator)
    return model

# Function to predict class labels using the trained model
def make_predictions(model, test_data):
    inference = VariableElimination(model)
    predictions = []

    for _, instance in test_data.iterrows():
        prediction = inference.map_query(variables=['Class'], evidence=instance.to_dict())
        predictions.append(prediction['Class'])

    return predictions

# Function to evaluate and print model performance
def evaluate_model(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

# Main function to run the complete workflow
def main():
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
    data = load_data(data_url)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the Bayesian Network model
    model = train_model(X_train, y_train)

    # Make predictions on the test data
    y_pred = make_predictions(model, X_test)

    # Evaluate the model's performance
    evaluate_model(y_test, y_pred)

# Run the workflow
if _name_ == '_main_':
    main()
