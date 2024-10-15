"""
Veysel Uğur
201713709050
Göğüs kanseri tespit modeli
"""

from keras.models import Sequential
from keras.layers import  Dense , Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

veri=pd.read_csv("breast-cancer-wisconsin.data")

veri.replace('?', -99999, inplace='true')
#veri.drop({'id'}, axis=1)
veriyeni=veri.drop(['1000025'],axis=1)

imp = Imputer(missing_values=99999, strategy="mean",axis=0)
veriyeni=imp.fit_transform(veriyeni)

giris=veriyeni[:,0:8]
cikis = veriyeni[:,9]

model=Sequential()
model.add(Dense(64),input_dim=8)
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('softmax'))#Softmax olmak zorunda

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(giris,cikis,epochs=5,batch_size=32,validation_split=0.13)
