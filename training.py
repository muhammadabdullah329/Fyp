import pandas as pd
import time
import os
import keras
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from keras.layers import Flatten
from keras.layers import Bidirectional
import itertools

dataset_main =  pd.read_csv('csv_out/finaldata_csv.csv',index_col = 0)
#print(dataset_main.head(12))

labels = dataset_main.iloc[:,-2].values


y_train = []
for l in labels:
	y_train.append(l)
	
#print(y_train[0])

ds_x_train = dataset_main.iloc[:,:-2].values
ds_y_train = np.array(y_train)



#normalize
scaler = MinMaxScaler()
ds_x_train= scaler.fit_transform(ds_x_train)

#save inference
joblib.dump(scaler, 'output/new_minmaxscalar.pkl')

#block creation
Y_train = []
count = 0
for i in range(11, len(ds_y_train),12):
	count+=1
	Y_train.append(ds_y_train[i])


Y_train = np.array(Y_train)
#print(Y_train)
#print(len(ds_y_train))
#print(count)

blocks = int(len(ds_x_train)/12)
X_train = np.array(np.split(ds_x_train,blocks))
#print(X_train.shape)


# label encoding

label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
print(label_encoder.classes_)

#Splitting training and testing

x_train,x_test,y_train, y_test = train_test_split(X_train, Y_train,test_size=0.2, random_state=0)

#optional
#np.save("output/ x_test_1.npy",x_test)
#np.save("output/ x_train_1.npy",x_train)

#one hot encoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#print(x_train.shape)
#print(y_train[0])

#early stopping

callbacks1=ModelCheckpoint("output/har_simpleLSTM.hdf5",save_best_only=True)
callbacks2=EarlyStopping(monitor='val_loss',patience=100,verbose=1)
callbacks=[callbacks1,callbacks2]

#model

model = Sequential()
#[samples, timesteps, features]

#---------------vanilla lstm---------------
#model.add(LSTM(50, activation='sigmoid', input_shape=(12, 50)))
#model.add(Dense(5,activation='softmax'))
#-------------------------------------------------

#---------------Bidirectional lstm---------------
model.add(Bidirectional(LSTM(50, activation='sigmoid'), input_shape=(12, 50)))
model.add(Dense(5,activation='softmax'))
#-------------------------------------------------

#--------------------Stacked Lstm--------------------------------
#model.add(LSTM(34,input_shape=(12,50),return_sequences=True,activation='sigmoid'))
#model.add(LSTM(34,activation="sigmoid"))#relu
#model.add(Dense(64))
#model.add(Dense(5,activation='softmax'))
#------------------------------------------------------------

#-------------------Adam optimizer------------------------
optimizer = keras.optimizers.Adam(lr=1e-04)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

#-------------------SGD optimizer------------------------
#opt = tf.keras.optimizers.SGD(lr=1e-04, momentum=0.9)
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=500,callbacks=callbacks)


print(x_test[1].shape)
y_pred = []
output = []

for i in range(len(x_test)):
	y_pred.append(label_encoder.inverse_transform(model.predict_classes(np.expand_dims(x_test[i],axis=0))))
y_pred = np.array(y_pred)

for i in range(len(y_test)):
	output.append(label_encoder.inverse_transform(np.expand_dims(np.argmax(y_test[i]),axis=0)))
output = np.array(output)

print (accuracy_score(output,y_pred))


