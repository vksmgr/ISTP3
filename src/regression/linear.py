##################################
## this module explain the simpe linear regression model
## This is the Single variable regression Y = b0 + b1X
#######################
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# load the dataset form the csv
adv = pd.read_csv("C:\\Users\\205218023\PycharmProjects\ISTP\data\\adv.csv", index_col='index')

# getting tv as 2d variable of shape 200x1
tv = adv[['TV']]
sales = adv[['sales']]

# spliting the data in train and test(20%)
train_tv, test_tv, train_sales, test_sales = train_test_split(tv, sales, test_size=0.2)

# now create linear regression object
reg_linear = linear_model.LinearRegression()    #this will create the object of the regression

# we will train the model using the tain data
reg_linear.fit(train_tv, train_sales)

#now we will predict
sales_predict = reg_linear.predict(test_tv)

#Printing the coefficients (b1)
print('Coefficients : {}'.format(reg_linear.coef_))

# getting the intercept of the model it is indepent of the TV data passed (b0)
print('Intercept : {}'.format(reg_linear.intercept_))

# the mean squared error
print('Mean Square Error: {}'.format(mean_squared_error(test_sales, sales_predict)))

#Explained Variance Score : 1 is perfect prediction
print('Variance score: {}'.format(r2_score(test_sales, sales_predict)))

# Ploting the out puts
plt.scatter(test_tv, test_sales, color='green', marker='*')
plt.plot(test_tv, sales_predict, color='blue', linewidth=3)


plt.xticks(())
plt.yticks(())

plt.show()