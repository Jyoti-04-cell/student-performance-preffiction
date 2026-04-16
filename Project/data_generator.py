import pandas as pd
import numpy as np

def generate_student_data(num_students=1000, output_path='student_data.csv'):
    np.random.seed(42)

    # Generate features
    attendance = np.random.randint(50, 101, num_students)  # 50% to 100%
    study_hours = np.random.randint(1, 21, num_students)   # 1 to 20 hours per week
    previous_grade = np.random.randint(40, 101, num_students) # 40% to 100%
    participation = np.random.randint(0, 101, num_students)   # 0 to 100 base score
    
    # Categorical feature
    extracurricular = np.random.choice(['Yes', 'No'], num_students)
    extracurricular_num = np.array([1 if x == 'Yes' else 0 for x in extracurricular])

    # Target generation (Final Score)
    # Logical weightings to make the ML models work well
    # Base = 10, Attendance helps, Previous Grade is a strong indicator, Study Hours help
    final_score_base = (
        10 +
        (attendance * 0.25) + 
        (previous_grade * 0.45) + 
        (study_hours * 0.8) + 
        (participation * 0.1) +
        (extracurricular_num * 3)
    )
    
    # Add some random noise to make it realistic
    noise = np.random.normal(0, 3, num_students)
    final_score = final_score_base + noise

    # Clip scores to be strictly between 0 and 100
    final_score = np.clip(final_score, 0, 100)

    # Create DataFrame
    df = pd.DataFrame({
        'Attendance': attendance,
        'Study_Hours': study_hours,
        'Previous_Grade': previous_grade,
        'Participation': participation,
        'Extracurricular': extracurricular,
        'Final_Score': final_score
    })

    # Round numerical features to 1 or 0 decimal places
    df['Final_Score'] = df['Final_Score'].round(1)

    df.to_csv(output_path, index=False)
    print(f"Dataset with {num_students} records generated and saved to {output_path}")

if __name__ == "__main__":
    generate_student_data()
