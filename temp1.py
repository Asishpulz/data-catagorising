

import numpy as mp
import matplotlib.pyplot as plt
import pandas as pd
#importing  datasets to the file
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[: ,:-1].values
y=dataset.iloc[: ,1].values

#splitting the data set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results

y_pred=regressor.predict(x_test)

#visualising the training results 
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("salary vs experience" )
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()
#test fitting
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("sal" )
plt.xlabel("ex")
plt.ylabel("sal")
plt.show()
