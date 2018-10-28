# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('C:/Users/Jose/Documents/Statistics, Data Science, and Programming/Deep Learning Course/bmi_and_life_expectancy.csv')

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
LE  = bmi_life_data.iloc[:,[1]]
BMI = bmi_life_data.iloc[:,[2]]
bmi_life_model = LinearRegression()
bmi_life_model.fit(BMI,LE)

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([ [21.07931] ])
print(laos_life_exp)
# 60.57949416
