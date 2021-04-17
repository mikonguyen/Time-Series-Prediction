# -*- coding: utf-8 -*-
"""
Two Fundamental Factors: PIOtroski and ROE, model: random forest classifier
"""

import numpy as np
import pandas as pd
import functions as fn
import matplotlib.pyplot as plt
import seaborn as sns

def MAD_mean_ratio(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.mean(y_true))) * 100

sns.set(style='darkgrid')
#pd.set_option('display.max_rows', 16)

#original data from sharadar quandl fundamental data had a different format
#the fundamental data from WRDS comes already staked
#stacked means the data for each ticker=stock is stacked in a pile one on top of the next
#the  model here is completely cross sectional, does not consider the timeseries structure of the data

dfroe = pd.read_csv("mixed_cap_roe_UNSTACKED.csv")
dfdates = dfroe.datetime
dfroe.drop(['datetime'], axis=1, inplace=True)
dfroe.replace([np.inf, -np.inf, np.NINF, -np.NINF], np.nan, inplace=True)
dfpio = pd.read_csv("mixed_cap_piotroski_UNSTACKED.csv")
dfpio.drop(['datetime'], axis=1, inplace=True)
dfpio.replace([np.inf, -np.inf, np.NINF, -np.NINF], np.nan, inplace=True)

dfpermnos = pd.read_csv("mixed_cap_permnos_with_wrds_data.csv")
syms1=dfpermnos.permno.tolist()
symsP=dfpermnos.permno.tolist()

dfidates = pd.DatetimeIndex(dfdates)
dates1=dfidates #


#need to build a df with mid prices unstacked, each column is mid prices for one stock
#this is the daily close mid_price, upon which all return calcs will be based
dfmp = pd.read_csv("mixed_cap_mid_price_UNSTACKED.csv")
dfmp.drop(['datetime'], axis=1, inplace=True)
dfmp.replace([np.inf, -np.inf, np.NINF, -np.NINF], np.nan, inplace=True)
mid=dfmp

#need to build a df with 1Day returns, each column is for one stock
Ret1 = fn.calculateReturns(mid.copy(), 1) #df, 


# hold for 21 days (1 month)
holdingDays = 21

# monthly return
RetM = fn.calculateReturns(mid.copy(), holdingDays) #df, monthly returns calculated daily
#del matP

# shifted next month's return to today's row to use as response variable.
# Can enter only at next day's close.
RetFut = RetM.copy().shift(-(holdingDays+1)) #df, monthly returns calculated daily shifted to today

#del RetM, mid


#############################
m = RetFut.shape[0]
trainSize = m // 2               #
testSize = m - trainSize         #
#############################


RetFutTrain = RetFut.iloc[:trainSize] #df, 
RetFutTest = RetFut.iloc[trainSize:] #df, 
Ret1train = Ret1.iloc[:trainSize] #df, 
Ret1test = Ret1.iloc[trainSize:] #df, 

#flattening and repetition of the date vector, this is the height of daily calculated return column after stacking of stocks
flat_train = RetFutTrain.size #
flat_test = RetFutTest.size #
m0 = RetFut.shape[0] #original number of observations, 
nStocks = len(syms1) #original number of stocks,

datesTrain = dates1.to_numpy()[:trainSize].flatten() #daily dates, 
datesTest = dates1.to_numpy()[trainSize:].flatten() #daily dates, 

#del dates1, datesP, RetFut

# Combine different independent variables into one matrix X for training
Xtrain = np.nan * np.empty((trainSize * len(symsP), 2)) #
Xtest = np.nan * np.empty((testSize * len(symsP), 2)) #
# Combine different independent variables into one matrix X for training
#Xtrain = np.nan * np.empty((trainSize * len(symsP), 4)) #
#Xtest = np.nan * np.empty((testSize * len(symsP), 4)) #

# dependent variable, stacked and divided into train and test
RetFutTrain = pd.DataFrame(np.where(RetFutTrain>0,1,0)) #categorical tag for logistic
RetFutTest = pd.DataFrame(np.where(RetFutTest>0,1,0)) #categorical tag for logistic
Ytrain = RetFutTrain.to_numpy().flatten()[:,np.newaxis] #arr, 
Ytest = RetFutTest.to_numpy().flatten()[:,np.newaxis] #arr, 

ROE = dfroe
ROE.fillna(value=-99.5, inplace=True) #substitutes NA (not available) and NAN (not a number)
ROE[ROE <= 0] = np.nan #gets rid of all negative values of ROE when dropna is applied eventually

 
PIO = dfpio
PIO.fillna(value=-99.5, inplace=True) #substitutes NA (not available) and NAN (not a number)
PIO[PIO <= 0] = np.nan #gets rid of all negative values of PIO when dropna is applied eventually


#At this point PIO and ROE are dfs with shape row*cols
#But now we are going to take the columns and stack them to verical shape
Xtrain[:,0] = np.log1p(PIO.iloc[:trainSize].to_numpy()).flatten() #numpy.ndarray, 
Xtrain[:,1] = np.log1p(ROE.iloc[:trainSize].to_numpy()).flatten() #numpy.ndarray, 

Xtest[:,0] = np.log1p(PIO.iloc[trainSize:].to_numpy()).flatten() #numpy.ndarray, 
Xtest[:,1] = np.log1p(ROE.iloc[trainSize:].to_numpy()).flatten() #numpy.ndarray, 

#Dropna allows the number of rows in XYtrain to be reduced, the third column being the added Y
#the dropna is eliminating any rows where PIO and ROE columns have NaN, keeping only those rows
#where there is PIO, ROE and return data (monthly data). We need to keep the dropna
XYtrain = pd.DataFrame(np.hstack([Xtrain, Ytrain])).dropna(how='any', axis=0) #df, 
XYtest = pd.DataFrame(np.hstack([Xtest, Ytest])).dropna(how='any', axis=0) #df, 

ixCleanTrain = XYtrain.index.to_numpy()
ixCleanTest = XYtest.index.to_numpy()

XYtrain = XYtrain.to_numpy()
XYtest = XYtest.to_numpy()

Xtrain, ytrain = XYtrain[:, 0:-1], XYtrain[:, -1][:,np.newaxis]
Xtest, ytest = XYtest[:, 0:-1], XYtest[:, -1][:,np.newaxis]



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
np.random.seed(2)

# Create the pipe to tune

pipe = Pipeline([("scaler",StandardScaler()),("rf",RandomForestClassifier())])

#prepare parameter_grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]

# Number of features to consider at every split
max_features = [round(x,2) for x in np.linspace(start = 0.1, stop = .5, num = 5)]

# Max depth of the tree
max_depth = [round(x,2) for x in np.linspace(start = 2, stop = 5, num = 3)]

# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(start = 15, stop = 100, num = 10)]

# Method of selecting training subset for training each tree
bootstrap = [False,True]


# Save these parameters in a dictionary
param_grid = {'rf__n_estimators': n_estimators,
               'rf__max_features': max_features,
               'rf__max_depth': max_depth,
               'rf__min_samples_leaf': min_samples_leaf,
               'rf__bootstrap': bootstrap,
              }
 
# Print the dictionary
print(param_grid)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

#Random Search
# Uncomment below line to see detail about RandomizedSearchCV function
# help(RandomizedSearchCV)

# Random search of parameters by searching across 50 different combinations

kfold = KFold(n_splits = 5, shuffle = True)
rs = 41

rso = RandomizedSearchCV(pipe, 
                               param_distributions = param_grid, 
                               n_iter = 100,                               
                               random_state= rs,
                               cv = kfold
                               )
# Fit the model to find the best hyperparameter values
rso.fit(Xtrain, ytrain.ravel())

print(rso.best_params_)

# Assign the best model to best_random_forest
best_random_forest = rso.best_estimator_

# Initialize random_state to 42
best_random_forest.random_state = rs


scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

# Fit the best random forest model on train dataset

treeModel = best_random_forest.fit(Xtrain, ytrain)

from sklearn.metrics import accuracy_score

print( ('Number of observations: {:d}').format(trainSize))

print(('score-training-w/score = {}').format(treeModel.score(Xtrain, ytrain)))

# Make "predictions" based on model on training set, reshape back to original matrix dimensions
Ypred = treeModel.predict(Xtrain)
#Ypred = treeModel.predict(Xtrain).reshape(-1,1)

print(('accuracy_score-training = {:.4f}').format(accuracy_score(ytrain, Ypred)))


retPred = np.nan * np.empty((flat_train, 1))
Ypred = np.reshape(Ypred,(Ypred.shape[0],1))#######################################
retPred[ixCleanTrain] = Ypred
retPred = retPred.reshape(RetFutTrain.shape)

def compare_nan_array(func, a, thresh):
    out = ~np.isnan(a)
    out[out] = func(a[out] , thresh)
    return out

longs = pd.DataFrame(compare_nan_array(np.greater, retPred, .5).astype(int)).shift(1) #1 day later
shorts = pd.DataFrame(compare_nan_array(np.less, retPred, .5).astype(int)).shift(1)

longs.iloc[0] = 0
shorts.iloc[0] = 0

positions = np.zeros(retPred.shape)

for h in range(holdingDays):
    long_lag = longs.shift(h)
    long_lag.fillna(value=0, inplace=True)
    long_lag = long_lag.to_numpy(dtype=bool)
    
    short_lag = shorts.shift(h)
    short_lag.fillna(value=0, inplace=True)
    short_lag = short_lag.to_numpy(dtype=bool)
    
    positions[long_lag] += 1
    positions[short_lag] -= 1

dRetA = np.sum(np.multiply(pd.DataFrame(positions).shift(1).fillna(0).to_numpy().squeeze(), Ret1train.fillna(0).to_numpy()), axis=1)
dPos = np.sum(pd.DataFrame(positions).shift(1).abs().to_numpy().squeeze(), axis=1)

dailyRet = np.divide(dRetA, dPos, out=np.zeros_like(dRetA), where=dPos!=0)
dailyRet[~np.isfinite(dailyRet)] = 0
cumret = np.cumprod(1 + dailyRet) - 1

plt.figure(1)
plt.xticks(rotation=70) 
plt.plot(datesTrain, cumret)
#plt.plot(range(trainSize), cumret)
plt.title('RandomForestClassifier: In-Sample on mixedCap IJR&IJH log(PIO) and log(ROE)')
plt.ylabel('Cumulative Returns')
plt.xlabel('Days')

cagr = (1 + cumret[-1]) ** (252 / trainSize) - 1
maxDD, maxDDD, _ = fn.calculateMaxDD(cumret)
ratio = (252.0 ** 0.5) * np.mean(dailyRet) / np.std(dailyRet)
print(('In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

print("shorts?: ", shorts.max().max())


##################################################################

# Make real predictions on test (out-of-sample)
#Ypred = linregModel.predict(Xtest)
Ypred = treeModel.predict(Xtest)
#Ypred = treeModel.predict(Xest).reshape(-1,1)

print(('accuracy_score-testing = {:.4f}').format(accuracy_score(ytest, Ypred)))

retPred = np.nan * np.empty((RetFutTest.size, 1))
Ypred = np.reshape(Ypred,(Ypred.shape[0],1))
retPred[ixCleanTest] = Ypred
retPred = retPred.reshape(RetFutTest.shape)

longs = pd.DataFrame(compare_nan_array(np.greater, retPred, .5).astype(int)).shift(1) #1 day later: probabilities
shorts = pd.DataFrame(compare_nan_array(np.less, retPred, .5).astype(int)).shift(1) #1 day later: probabilities

longs.iloc[0] = 0
shorts.iloc[0] = 0

positions = np.zeros(retPred.shape)

for h in range(holdingDays):
    long_lag = longs.shift(h)
    long_lag.fillna(value=0, inplace=True)
    long_lag = long_lag.to_numpy(dtype=bool)
    
    short_lag = shorts.shift(h)
    short_lag.fillna(value=0, inplace=True)
    short_lag = short_lag.to_numpy(dtype=bool)
    
    positions[long_lag] += 1
    positions[short_lag] -= 1

dRetA = np.sum(np.multiply(pd.DataFrame(positions).shift(1).fillna(0).to_numpy().squeeze(), Ret1test.fillna(0).to_numpy()), axis=1)
dPos = np.sum(pd.DataFrame(positions).shift(1).abs().to_numpy().squeeze(), axis=1)

dailyRet = np.divide(dRetA, dPos, out=np.zeros_like(dRetA), where=dPos!=0)
dailyRet[~np.isfinite(dailyRet)] = 0
cumret = np.cumprod(1 + dailyRet) - 1

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(2)
plt.xticks(rotation=70) 
plt.plot(datesTest, cumret)
plt.title('RandomForestClassifier: Out-of-Sample on mixedCap IJR&IJH log(ROE) and log(PIO)')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')

cagr = (1 + cumret[-1]) ** (252 / testSize) - 1
maxDD, maxDDD, _ = fn.calculateMaxDD(cumret)
ratio = (252.0 ** 0.5) * np.mean(dailyRet) / np.std(dailyRet)
print(('Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))

plt.show() 
print("shorts?: ",shorts.max().max())

###########################################################################
#market returns long only passive

"""
longs = longs*0
longs = longs+1  #long all the time
shorts = shorts*0 #no shorts

longs.iloc[0] = 0
shorts.iloc[0] = 0

positions = np.ones(retPred.shape)

for h in range(holdingDays):
    long_lag = longs.shift(h)
    long_lag.fillna(value=0, inplace=True)
    long_lag = long_lag.to_numpy(dtype=bool)
    
    short_lag = shorts.shift(h)
    short_lag.fillna(value=0, inplace=True)
    short_lag = short_lag.to_numpy(dtype=bool)
    
    positions[long_lag] += 1
    positions[short_lag] -= 1
    
dRetA = np.sum(np.multiply(pd.DataFrame(positions).shift(1).fillna(0).to_numpy().squeeze(), Ret1test.fillna(0).to_numpy()), axis=1)
dPos = np.sum(pd.DataFrame(positions).shift(1).abs().to_numpy().squeeze(), axis=1)

dailyRet = np.divide(dRetA, dPos, out=np.zeros_like(dRetA), where=dPos!=0)
dailyRet[~np.isfinite(dailyRet)] = 0
cumret = np.cumprod(1 + dailyRet) - 1

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(3)
plt.plot(datesTest, cumret)
plt.title('RandomForestClassifier: Out-of-Sample MARKET on mixedCap IJR&IJH log(ROE) and log(PIO')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')

cagr = (1 + cumret[-1]) ** (252 / testSize) - 1
maxDD, maxDDD, _ = fn.calculateMaxDD(cumret)
ratio = (252.0 ** 0.5) * np.mean(dailyRet) / np.std(dailyRet)
print(('Out-of-sample-MARKET: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}'\
).format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr/maxDD))


#Out-of-sample-MARKET: CAGR=0.212432 Sharpe ratio=1.42583 maxDD=-0.117532 maxDDD=121 Calmar ratio=1.80744
#logistic Out-of-sample: CAGR=0.252154 Sharpe ratio=1.5051 maxDD=-0.13174 maxDDD=128 Calmar ratio=1.91403
#RandomForest: Out-of-sample: CAGR=0.251848 Sharpe ratio=1.49727 maxDD=-0.132466 maxDDD=128 Calmar ratio=1.90123
#comp. to market neutral strategy, random forest has higher cagr and sharpe and sometimes goes short
"""