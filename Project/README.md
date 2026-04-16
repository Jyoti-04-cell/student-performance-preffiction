# AI-Based Student Performance Prediction System

An end-to-end Machine Learning web application designed to predict a student's final academic performance based on historical and behavioral factors. The project generates synthetic educational data, trains several models to deduce patterns, and provides a sleek UI for teachers or students to make predictions.

## Features
- **Synthetic Data Generation:** Generates a realistic dataset mimicking educational scenarios.
- **Multiple ML Models:** Compares Linear Regression, Decision Trees, and Random Forests.
- **Modern Web Interface:** A pristine, glassmorphism-inspired dark mode UI built with Flask.
- **Actionable Insights:** Doesn't just give a score - it categorizes the student's trajectory (e.g., "At Risk", "Excellent").

## Tech Stack
- **Backend:** Python, Flask
- **Data Science:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML5, CSS3, Google Fonts, FontAwesome

## Getting Started

### 1. Requirements
Ensure you have the libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn flask joblib
```

### 2. Generate the Dataset
If you haven't yet, generate the `student_data.csv` dataset:
```bash
python data_generator.py
```

### 3. Train the Model
Run the Model Trainer to process the data, evaluate models, and save the best performer into the `models/` directory:
```bash
python model_trainer.py
```
*Note: This will also output an `actual_vs_predicted.png` chart showing the model accuracy.*

### 4. Start the Application
Run the Flask server:
```bash
python app.py
```
Open a browser and navigate to `http://127.0.0.1:5000/`.

## Future Improvements
- **Deep Learning Integration:** Implement a Neural Network using TensorFlow or PyTorch.
- **Database Backend:** Store predicted results in an SQLite or PostgreSQL database.
- **User Authentication:** Allow teachers to log in and track specific classes.
- **More Features:** Integrate metrics like "Socioeconomic Status" or "Sleep Hours".
