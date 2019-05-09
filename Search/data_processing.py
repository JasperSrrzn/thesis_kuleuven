import numpy as np
import os
import pandas as pd
import json
import tensorflow as tf

def train_val_test_split(data,train_facts,validation_facts,test_facts):
    test_data = pd.merge(data,test_facts,on=['entity a','entity b','rel id'],how='inner')
    validation_data = pd.merge(data,validation_facts,on=['entity a','entity b','rel id'],how='inner')
    train_data = pd.merge(data,train_facts,on=['entity a','entity b','rel id'],how='inner')
    return train_data, validation_data, test_data

def clear_data(train_data,validation_data,test_data):
    train_data.drop(train_data.columns[[0,1,2,3]],axis=1,inplace=True)
    test_data.drop(test_data.columns[[0,1,2,3]],axis=1,inplace=True)
    validation_data.drop(validation_data.columns[[0,1,2,3]],axis=1,inplace=True)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])
    y_validation = np.array(validation_data['label'])

    y_train = tf.Session().run(tf.one_hot(y_train,depth=2))
    y_test = tf.Session().run(tf.one_hot(y_test,depth=2))
    y_validation = tf.Session().run(tf.one_hot(y_validation,depth=2))

    train_data.drop(train_data.columns[0],axis=1,inplace=True)
    test_data.drop(test_data.columns[0],axis=1,inplace=True)
    validation_data.drop(validation_data.columns[0],axis=1,inplace=True)

    x_train = np.array(train_data)
    x_test = np.array(test_data)
    x_validation = np.array(validation_data)

    return x_train, y_train, x_validation, y_validation, x_test, y_test

def clear_data_part(data):
    meta_data = data[['entity a','entity b','rel id','rule id','label']]
    data.drop(data.columns[[0,1,2,3]],axis=1,inplace=True)
    y = np.array(data['label'])
    y = tf.Session().run(tf.one_hot(y,depth=2))
    data.drop(data.columns[0],axis=1,inplace=True)
    x = np.array(data)
    return x, y, meta_data
