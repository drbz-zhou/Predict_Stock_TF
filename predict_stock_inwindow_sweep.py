# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import yfinance as yf  
import numpy as np
import tensorflow as tf
import j_finance as jf

    # Get the data for the stock Apple by specifying the stock ticker, start date, and end date

#t_amd = yf.Ticker("AMD")
#dt_amd = t_amd.history(period="20y",interval = "1d")
#print(dt_amd.Open[0:10])

m_lossacc = np.zeros( (2,30) )
m_cm = np.zeros( (5,5,30) )
for m_in_window in range(5, 30): #time window in days
    print(m_in_window)
    m_pd_window = 1 #prediction window in days
    x_data_all = np.zeros( (0, m_in_window, 5) ) 
    y_data_all = np.zeros( (0, 1), dtype = int ) 
    tickers = {"AMD"}
    for m_ticker in tickers:
        print(m_ticker)
        t = yf.Ticker(m_ticker) 
        dt = t.history(period="max",interval = "1d")
    # prepare data matrix
        x_data = np.zeros( (dt.shape[0]-m_in_window, m_in_window, 5) ) 
        y_data = np.zeros( (dt.shape[0]-m_in_window, 1), dtype = int ) # prediction is the next day rise or fall
        
        for i in range(0, dt.shape[0]-m_in_window-m_pd_window+1):
            #x_data[i, 0:(1*m_in_window)] = dt.Open[i:i+m_in_window]
            #x_data[i, (1*m_in_window):(2*m_in_window)] = dt.High[i:i+m_in_window]
            #x_data[i, (2*m_in_window):(3*m_in_window)] = dt.Low[i:i+m_in_window]
            #x_data[i, (3*m_in_window):(4*m_in_window)] = dt.Close[i:i+m_in_window]
            #x_data[i, (4*m_in_window):(5*m_in_window)] = dt.Volume[i:i+m_in_window]
            m_range = dt.High[i:i+m_in_window].max(axis=0) - dt.Low[i:i+m_in_window].min(axis=0)
            m_min = dt.Low[i:i+m_in_window].min(axis=0)
            x_data[i, :, 0] = (dt.Open[i:i+m_in_window]-m_min)/m_range #remove first in window for normalize
            x_data[i, :, 1] = (dt.High[i:i+m_in_window]-m_min)/m_range
            x_data[i, :, 2] = (dt.Low[i:i+m_in_window]-m_min)/m_range
            x_data[i, :, 3] = (dt.Close[i:i+m_in_window]-m_min)/m_range
            x_data[i, :, 4] = (dt.Volume[i:i+m_in_window]) / dt.Volume[i:i+m_in_window].mean(axis=0)
    
            m_gain = dt.Close[i+m_in_window+m_pd_window-2]-dt.Close[i+m_in_window-2]
            m_gain_per = m_gain/dt.Open[i]
            if m_gain_per > 0.01:
                y_data[i] = 4
            elif m_gain_per > 0.005:
                y_data[i] = 3
            elif m_gain_per > -0.005:
                y_data[i] = 2
            elif m_gain_per > -0.01:
                y_data[i] = 1
            else:
                y_data[i] = 0
        x_data_all=np.append(x_data_all, x_data, axis=0)
        y_data_all=np.append(y_data_all, y_data, axis=0)
                
    #%%
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    # random split
    #x_train, x_test, y_train, y_test = train_test_split(x_data_all, y_data_all, test_size=0.1, random_state=42)
    
    # leave the last 100 days for test
    x_leave = 100;
    x_len = x_data_all.shape[0]
    x_train = x_data_all[0:x_len-x_leave,:,:]
    y_train = y_data_all[0:x_len-x_leave,:]
    
    x_test = x_data_all[x_len-x_leave:x_len,:,:]
    y_test = y_data_all[x_len-x_leave:x_len,:]
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(m_in_window,5)),
      #tf.keras.layers.Dense(256, activation='sigmoid'),
      tf.keras.layers.Dense(128, activation='sigmoid'),
    #  tf.keras.layers.Dense(64, activation='sigmoid'),
      tf.keras.layers.Dense(32, activation='sigmoid'),
      #tf.keras.layers.Dense(m_in_window*5, activation='sigmoid'),
      #tf.keras.layers.Dense(m_in_window, activation='sigmoid'),
      #tf.keras.layers.Dense(50, activation='sigmoid'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #%%
    model.fit(x_train, y_train, epochs=200, verbose=0)
    
    m_lossacc[:,m_in_window-5]=model.evaluate(x_test,  y_test, verbose=2)
    #%
    from sklearn.metrics import confusion_matrix
    
    y_predict = model.predict(x_test)
    max_value=y_predict.max(axis=1)
    max_index = y_predict.argmax(axis=1)
    
    print(confusion_matrix(y_test, max_index))
    m_cm[:,:,m_in_window-5] = confusion_matrix(y_test, max_index)
