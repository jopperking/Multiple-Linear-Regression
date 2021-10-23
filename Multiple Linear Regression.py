import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


# Dataset Download :   !wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv

# Read csv File

df = pd.read_csv("FuelConsumption.csv")
print("\nFuelConsumption.csv :\n\n" , df.head(10))

# summarize the data
print("\nSummurize :\n\n" , df.describe())

# decreasing features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print("\nSmaller Dataset :\n\n" ,cdf.head(10))

# Splite Data to train & test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Training
from sklearn  import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x,train_y)
#The coefficients
teta0 = regr.intercept_[0]
teta1 , teta2 , teta3 = regr.coef_[0]
print ('\nCoefficients: ',teta1 , teta2 , teta3)
print ('Intercept: ' , teta0)

# Prediction

x = np.asanyarray(test[['ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB"]])
y = np.asanyarray(test[["CO2EMISSIONS"]]) # real values
y_hat = regr.predict(test[['ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB"]])  # Guessed values via regression

print ("\nResidual sum of squares (MSE): %.2f " %np.mean((y_hat - y) ** 2))
# variance score: 1 is perfect prediction
print ("Variance score: %.2f \n\n" %regr.score(x,y))

