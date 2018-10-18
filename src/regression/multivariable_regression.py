###############################
##this is the multi variable regression
## Y = b0 + b1X1 + b1X2 + .... +b1Xn
##############################

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#read The data
df = pd.read_csv("C:\\Users\\205218023\PycharmProjects\ISTP\data\\adv.csv", index_col='index')

# here we want to predict the sales Here the variable X has two variable
train_X, test_X, train_Y, test_Y = train_test_split(df[['TV', 'radio']], df[['sales']], test_size=0.2)

#get the learnear regression object to train
model = linear_model.LinearRegression()

#train the model
model.fit(train_X, train_Y)

#predicting the values note We never pass the test_y
pred_sales  = model.predict(test_X)

# calculating the evaluation parameteres
print('coefficient : {}'.format(model.coef_))
print('Intercept {}'.format(model.intercept_))
print('Mean Square Error : {}'.format(mean_squared_error(test_Y, pred_sales)))
print('Variance Score : {}'.format(r2_score(test_Y, pred_sales)))
print("Okay!")