
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt 
from category_encoders import OrdinalEncoder, OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from xgboost import XGBRegressor

data = pd.read_csv("C:/Users/HP/Downloads/train.csv", encoding= 'unicode_escape')
dataset = pd.read_csv("C:/Users/HP/Downloads/test.csv", encoding= 'unicode_escape')

Y=data["Rented Bike Count"]
data.drop(['Rented Bike Count','Dew point temperature(Â°C)','Holiday','Snowfall (cm)'] ,axis=1, inplace=True)

ARKAM = ['Wind speed (m/s)','Humidity(%)','Visibility (10m)','Temperature(Â°C)','Solar Radiation (MJ/m2)','Rainfall(mm)']
data.boxplot(ARKAM)

for x in ['Wind speed (m/s)']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan
print(data.isnull().sum())
#data.dropna(inplace=True)

data['Date']=pd.to_datetime(data['Date'])
data["month"] = data['Date'].dt.month_name()
data["year"] = data['Date'].map(lambda x: x.year).astype("object")
data.drop(columns=['Date'],inplace=True)
data['Hour']=data['Hour'].astype('object')
seasons = data['Seasons']
data.drop(['Seasons'] ,axis=1, inplace=True)

labelen=LabelEncoder()
data['Functioning Day']=labelen.fit_transform(data['Functioning Day'])
data['month']=labelen.fit_transform(data['month'])
data['Seasons']=seasons

for i in range(len(data)):
    if data['Seasons'][i]=="Winter":
        data['Seasons'][i]=0
    if data['Seasons'][i]=="Spring":
        data['Seasons'][i]=1
    if data['Seasons'][i]=="Summer":
        data['Seasons'][i]=2

X=data

############################################
ID=dataset['ID']
dataset.drop(['ID','Dew point temperature(Â°C)','Holiday','Snowfall (cm)'] ,axis=1,inplace=True)

dataset['Date']=pd.to_datetime(dataset['Date'])
dataset["month"] = dataset['Date'].dt.month_name()
dataset["year"] = dataset['Date'].map(lambda x: x.year).astype("object") 

dataset.drop(columns=['Date'],inplace=True)
dataset['Hour']=dataset['Hour'].astype('object')
seasons = dataset['Seasons']
dataset.drop(['Seasons'] ,axis=1, inplace=True)


labelen=LabelEncoder()
dataset['Functioning Day']=labelen.fit_transform(dataset['Functioning Day'])
dataset['month']=labelen.fit_transform(dataset['month'])
dataset['Seasons']=seasons


for i in range(len(dataset)):
    if dataset['Seasons'][i]=="Autumn":
        dataset['Seasons'][i]=3
    if dataset['Seasons'][i]=="Summer":
        dataset['Seasons'][i]=2

data = data.dropna(axis = 1)
print(data.isnull().sum())
X2 =dataset    

#scaling data
std = StandardScaler()
X = std.fit_transform(X)
X2 = std.transform(X2)
print(data)
print(dataset)

model=XGBRegressor()
model.fit(X,Y)
YY=model.predict(X2)
dataframe=pd.DataFrame(np.zeros((dataset.shape[0],2)),columns=['ID','Rented Bike Count'])
dataframe['ID']=ID
dataframe['Rented Bike Count']=YY
dataframe.to_csv('Downloads/final test xgb.csv')

