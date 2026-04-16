import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate(data_path='student_data.csv'):
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return

    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # 2. Preprocess Data
    # Encode categorical 'Extracurricular' (Yes -> 1, No -> 0)
    le = LabelEncoder()
    df['Extracurricular'] = le.fit_transform(df['Extracurricular'])

    # Separate features (X) and target (y)
    X = df.drop('Final_Score', axis=1)
    y = df['Final_Score']

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Building & Evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100)
    }

    best_model = None
    best_r2_score = -np.inf
    best_model_name = ""

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R2 Score: {r2:.4f}")

        if r2 > best_r2_score:
            best_r2_score = r2
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with R2 Score: {best_r2_score:.4f}")

    # 4. Save the best model and preprocessors
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("\nModel and Preprocessors saved in 'models' directory.")
    
    # 5. Visualization (Actual vs Predicted for the best model)
    plt.figure(figsize=(8, 6))
    y_pred_best = best_model.predict(X_test_scaled)
    sns.scatterplot(x=y_test, y=y_pred_best, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual Final Score')
    plt.ylabel('Predicted Final Score')
    plt.title(f'{best_model_name} - Actual vs Predicted Scores')
    plt.savefig('actual_vs_predicted.png')
    print("Visualization saved to 'actual_vs_predicted.png'.")

if __name__ == '__main__':
    train_and_evaluate()
