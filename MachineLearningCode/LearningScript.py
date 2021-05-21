#import packages
import pandas as pd
import numpy as np

#setting figure size
from fastai.tabular.core import add_datepart
from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

#for normalizing data
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')

#print the head
print(df.head())
print('\n Shape of the Data: ')
print(df.shape)

# Setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

#Sort dataset
data = df.sort_index(ascending=True, axis=0)
print(data)

#creating a seperate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

print(new_data)
#HYPOTHESIS: Monday and Friday affect the closing prices more than other days
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)

train = new_data[:987]
valid = new_data[987:]
x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#Linear regression
model = LinearRegression()
model.fit(x_train, y_train)

#Scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#Gridsearch to find best params
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit model and predict
model.fit(x_train,y_train)
preds = model.predict(x_valid)

#make predeictions
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)), 2)))
print

#plot
valid = valid.assign(Predictions=preds)
print(valid)
valid.index = new_data[987:].index
train.index = new_data[:987].index
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.show()
