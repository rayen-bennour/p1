import pandas as pd
from pycaret.classification import *

# Load your data
data = pd.read_csv('student_data.csv')

# Print column names to ensure the target column is correct
print(data.columns)

# Define the target column
target_column = 'GRADE'  # Update with the exact name of your target column

# Define numeric and categorical features
numeric_features = ['Weekly study hours', 'Total salary if available']
categorical_features = ['Student Age', 'Sex', 'Graduated high-school type', 'Scholarship type', 'Additional work', 
                        'Regular artistic or sports activity', 'Do you have a partner', 'Transportation to the university',
                        'Accommodation type in Cyprus', 'Mother’s education', 'Father’s education ', 
                        'Number of sisters/brothers', 'Parental status', 'Mother’s occupation', 'Father’s occupation',
                        'Reading frequency', 'Reading frequency.1',
                        'Attendance to the seminars/conferences related to the department']

# Setup the PyCaret environment
exp_clf = setup(data, target=target_column, 
                numeric_features=numeric_features, 
                categorical_features=categorical_features,
                session_id=123)

# Compare different models and select the best one
best_model = compare_models()

# Finalize the best model
final_model = finalize_model(best_model)

# Save the model for future use
save_model(final_model, 'student_performance_model')

create_api(final_model, 'student_performance_model')