import pandas as pd
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv("C:\\Users\\205218023\PycharmProjects\ISTP\data\College.csv")

# this will just rename the first column
columns = list(df.columns)
columns1 = columns.copy()
columns1[0] = "college"
newcolums =dict(zip(columns, columns1))
df=df.rename(columns=newcolums)
col1 = df.iloc[:, [2,7]] # selection columns using the index
# this  will teach us how to index into the data frame
#print(col1)

# the describe function will provide summary of the dataframe

#print(df.describe())

# want to print the scatter plot of the first 10 colums i.e 5 scatter plots

#print(df.iloc[:,[2]])

plt.subplot(321)
plt.scatter(x=df.iloc[:,1], y=df.iloc[:,2])

plt.subplot(322)
plt.scatter(x=df.iloc[:,3], y=df.iloc[:,4])

plt.subplot(323)
plt.scatter(x=df.iloc[:,5], y=df.iloc[:,6])

plt.subplot(324)
plt.scatter(x=df.iloc[:,7], y=df.iloc[:,8])

plt.subplot(325)
plt.scatter(x=df.iloc[:,9], y=df.iloc[:,10])

#plt.show()

## box plot basics
df.boxplot(by='Private', column='Outstate')
#plt.show()

# Create new column "Elite" by binning the Top10Operc variable. we are going
# to divide universities into two groups based on wether or not the proportion
# of the students comming form the top 10% of their high school classes
# exceeds 50%
df['Elite']='No'
df[df['Top10perc']>50]="Yes"

#print(df.describe())

#df.boxplot(by='Elite', column='Outstate')
#plt.show()

#question 9

df = pd.read_csv('C:\\Users\\205218023\PycharmProjects\ISTP\data\Auto.csv')
# what are the qualitative (catogirical) predictors
first_row = df.iloc[:1,:].unstack()
second_row = df.iloc[1:2,:].unstack()
# print(first_row)
# print(second_row)
#only name and origin are catogirical predictor

## Range of the predictors
#describe return the minimum and maximum
desc = df.describe()
# This function will give the range of the function
min, max = (desc.loc['min'], desc.loc['max'])

# Finding mean and Standard deviation
# these functions will return series as an object
# The mean function will give the mean of the dataframe
#print(df.mean())
# the std() function will provide the standard deviation of the function
#print(type(df.std()))

## whnt to remove 10th to 85th observation
# to remove row from the dataframe we use the drop method
print(df.index)
newdf= df.drop([x for x in range(10,85)])
print(newdf.mean())
print(newdf.std())

## creating scatter plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(df.iloc[:,1], df.iloc[:,2], color='lightblue', linewidth=3)
ax.scatter(df.iloc[:,1], df.iloc[:,2],  marker='.')
plt.show()