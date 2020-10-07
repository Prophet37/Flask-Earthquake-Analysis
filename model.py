import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

raw_data=pd.read_csv('database.csv')

raw_data

raw_data.describe(include='all')

# date, time, Latitude, longitude, type, mag, type, id, source, location source, Mag source, Status
data= raw_data.drop(['Time','Date', 'Type','Magnitude Type' , 'ID','Source','Location Source','Magnitude Source','Status'],axis=1)

data.head()


data.isnull().sum()

data1= data.drop(['Horizontal Distance','Horizontal Error','Magnitude Error','Azimuthal Gap','Depth Error','Magnitude Seismic Stations','Depth Seismic Stations','Root Mean Square'],axis=1)

data1.describe()

data1.isnull().sum()


new_data=data1.sample(frac=1).reset_index(drop=True)

x=new_data.drop(['Magnitude','Depth'],axis=1)
y=new_data[['Magnitude','Depth']]

from sklearn.model_selection import train_test_split as tts

train_x,test_x,train_y,test_y=tts(x,y,random_state=10,train_size=0.75,test_size=0.25)

from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(random_state=42)
reg.fit(train_x, train_y)
k=reg.predict(test_x)

reg.score(test_x,test_y)

pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9]]))