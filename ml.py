import pandas as pd

data = pd.read_csv('student_performance_data.csv')

print("Top 5 records:")
print(data.head())

print("Bottom 5 records:")
print(data.tail())

print("Data Information:")
print(data.info())

print("Dataset Column Names:")
print(data.columns)

print("To find the null values:")
print(data.isnull().sum())

'''print("To drop null values:")
print(data.dropna(inplace=True))'''

print("To fill null values:")
print(data.fillna(65, inplace=True))

print("To find duplicates:", data.duplicated().sum())
print(data.drop_duplicates(inplace=True))

print("To find data after removing duplicates:", data.duplicated().sum())

# Feature Selection

inputs = data[['Study_Hours', "Assignment_Marks", 'Attendance']]
print(inputs)

output = data['Exam_Marks']
print(output)

from sklearn.model_selection import train_test_split
training_input, testing_input, training_output, tesing_output = train_test_split(inputs, output, test_size=0.2, random_state=42)
print("Input Training Data:\n", training_input)
print("Output Training Data:\n", training_output)
print("Input Testing Data:\n", testing_input)
print("Output Testing Data:\n", tesing_output)

# Standard Scaler not used because range difference isn't huge.

# Model Selection

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Model Training

model.fit(training_input, training_output)
print("Model Trained Successfuclly!")

# Model Evaluation
print("Testing Data Output Records:")
testing_prediction = model.predict(testing_input)
print("Model Prediction On Training Data:\n", testing_prediction)

from sklearn.metrics import mean_absolute_error
ModelError = mean_absolute_error(tesing_output, testing_prediction)