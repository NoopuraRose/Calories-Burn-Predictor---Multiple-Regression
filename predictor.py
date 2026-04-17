# library imports
import pandas as pd 
from sklearn.linear_model import LinearRegression

# data set reading
mydata = pd.read_csv("Calories_Burn_Prediction.csv")
x = mydata[["duration", "heart_rate", "age"]]
y = mydata[["calories"]]

# model creation and training
model = LinearRegression()
model.fit(x,y)

cf = model.coef_
print("Coefficient = ", cf)
intercept = model.intercept_
print("Intercept = ", intercept)

# predicting new value with user input
duration = int(input("Enter the duration:"))
heart_rate = int(input("Enter the heart rate:"))
age = int(input("Enter you age:"))

new_calories = model.predict([[duration, heart_rate, age]])
print("Predicted calories = ", new_calories)