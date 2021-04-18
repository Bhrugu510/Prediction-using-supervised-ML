# Prediction-using-supervised-ML
#Predict the percentage of an student based on the number of study hours.
#Linear Regression
# Importing all libraries required in this notebook

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

#Loading Data
# Reading data from remote link
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data imported successfully")

#Plotting Graph for Visualization
# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

#Preparing the data
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 
                            
---------------------------------------------------------------------------
#Training the Algorithm
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")

---------------------------------------------------------------------------
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_      # y = m*x+c

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

---------------------------------------------------------------------------
Making Predictions
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores

---------------------------------------------------------------------------
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 
      
