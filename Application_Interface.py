# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import pandas
import sklearn
import matplotlib
import pandas as pd
import keras
from tkinter import *  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def start():
    df = pd.read_csv('dlprj1.csv')
    dataset = df.values
    X = dataset[:,0:8]
    Y = dataset[:,8]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scale = min_max_scaler.fit_transform(X)
    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    global model
    model = Sequential([
    Dense(8, activation='relu', input_shape=(8,)),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    hist = model.fit(X_train, Y_train,batch_size=10,epochs=5,validation_data=(X_val, Y_val))
def findResult():
    test=np.array([long.get(),latt.get(),slopG.get(),slopA.get(),pgaM.get(),dap.get(),landC.get(),su.get()])
    test1=np.expand_dims(test,axis=0)
    test.shape
    yhat_probs1 = model.predict(test1,verbose=0)
    print(yhat_probs1)
    #Result.delete('1.0',END)
    Result.insert(0,yhat_probs1[0][0])
    #ResultF.delete('1.0',END)
    ResultF.insert(0,"Land Slide" if(yhat_probs1[0][0] == 0) else "Non Land Slide")

top = Tk()  
  
top.geometry("1250x720")  
  
#creating a simple canvas  

#Row - 0
header =Label(top,text = "EARTHQUAKE INDUCED LANDSLIDE PREDICTION").grid(row=0, column = 3,padx=0,pady=30)
longL = Label(top ,text = "Longitude").grid(row=1, column = 0)
lattL = Label(top ,text = "Lattitude").grid(row=1, column = 1)
slopGL = Label(top,text = "Slope Gradient").grid(row=1, column = 2)
slopAL = Label(top ,text = "Slope Aspect").grid(row=1, column = 3)
pgaML = Label(top ,text = "PGA Magnitude").grid(row=1, column = 4)
dafL = Label(top ,text = "Distance Active Fault").grid(row=1, column = 5)
landCL = Label(top ,text = "Land Cover").grid(row=1, column = 6)
suL = Label(top , text = "Strata Unit").grid(row=1, column = 7)

# Row -1
long = Entry(top)
long.grid(row = 2, column = 0,padx=5,pady=10)  

latt = Entry(top)
latt.grid(row = 2, column = 1,padx=5,pady=10)

slopG = Entry(top)
slopG.grid(row = 2, column = 2,padx=5,pady=10)    

slopA = Entry(top)
slopA.grid(row = 2, column = 3,padx=5,pady=10)

pgaM = Entry(top)
pgaM.grid(row = 2, column = 4,padx=5,pady=10)    

dap = Entry(top)
dap.grid(row = 2, column = 5,padx=5,pady=10)

landC = Entry(top)
landC.grid(row = 2, column = 6,padx=5,pady=10)    

su = Entry(top)
su.grid(row = 2, column = 7,padx=5,pady=30)
submit = Button(top,text = "Submit",command=findResult)
submit.grid(row = 4, column = 3)  

ResultL = Label(top, text = "Result").grid(row=5, column = 1,padx=5,pady=10)
Result = Entry(top)
Result.grid(row = 5, column = 2,padx=5,pady=10)  

ResultFL = Label(top, text = "Output").grid(row=8, column = 1,)
ResultF = Entry(top)
ResultF.grid(row = 8, column = 2)  
start()
top.mainloop()  