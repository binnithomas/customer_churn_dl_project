import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.models import load_model 
df = pd.read_csv("data/bank_churn_data - bank_chrun_data.csv")

x = pd.drop(['CutomerID','Surname','Exited'],axis=1)
y = df['Exited']

x['Geography'] = LableEncoder().fit_transform(x['Geography'])
x['Gender'] = LableEncoder().fit_transform(x['GENDER'])

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size =
0.2, random_state =42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

model = Sequential([ 
  Dense(16,activation='relu',input_shape(X_train.shape[1],)),
  Dense(8,activation='relu'),
  Desne(1,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=30,batch_size=13,validation_split=0.2,verbose=1)
model.save('models/chrun_model.h5')
pd.to_pickle(scaler,
