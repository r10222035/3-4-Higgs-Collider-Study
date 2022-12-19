#!/usr/bin/env python
# coding: utf-8

import datetime
import os

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES']='1'    

def main():
    
    # Training sample
    pairing_method = sys.argv[1]
    path_s = f'DNN_signal_{pairing_method}.npy'
    path_b = f'DNN_background_{pairing_method}.npy'
    
    X_s = np.load(path_s)
    Y_s = np.eye(2)[np.array([1] * X_s.shape[0])]
    X_b = np.load(path_b)
    Y_b = np.eye(2)[np.array([0] * X_b.shape[0])]

    X = np.vstack((X_s, X_b))
    Y = np.vstack((Y_s, Y_b))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    print(f'Signal size: {X_s.shape[0]}')
    print(f'Background size: {X_b.shape[0]}')
    
    train_epochs = 500
    patience = 10
    min_delta = 0.
    learning_rate = 5e-3   
    batch_size = 512
    save_model_name = f'DNN_best_model_{pairing_method}/'

    # 建立 DNN
    model= Sequential()
    model.add(Dense(units=256, input_dim=20, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, verbose=1, patience=patience)
    check_point    = tf.keras.callbacks.ModelCheckpoint(save_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=train_epochs, batch_size=batch_size, callbacks=[early_stopping, check_point])
    
    # Plot loss accuracy curve
    fig, ax = plt.subplots(1,1, figsize=(6,5))

    x = range(len(history.history['loss']))
    y_train = history.history['loss']
    y_validation = history.history['val_loss']

    ax.plot(x, y_train, label='Training')
    ax.plot(x, y_validation, label='Validation')

    ax.set_title('Loss across training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (categorical cross-entropy)')
    ax.legend()
    plt.savefig(f'figures/loss_curve_DNN_{pairing_method}.png', facecolor='White', dpi=300, bbox_inches = 'tight')


    fig, ax = plt.subplots(1,1, figsize=(6,5))

    x = range(len(history.history['accuracy']))
    y_train = history.history['accuracy']
    y_validation = history.history['val_accuracy']

    ax.plot(x, y_train, label='Training')
    ax.plot(x, y_validation, label='Validation')

    ax.set_title('Accuracy across training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.savefig(f'figures/accuracy_curve_DNN_{pairing_method}.png', facecolor='White', dpi=300, bbox_inches='tight')
    
    loaded_model = tf.keras.models.load_model(save_model_name)
    results = loaded_model.evaluate(x=X_test, y=y_test)
    print(f'Testing Loss = {results[0]:.3}, Testing Accuracy = {results[1]:.3}')

    # Plot ROC
    labels = y_test
    predictions = loaded_model.predict(X_test)
    
    y_test = np.argmax(labels, axis=1)
    y_prob = np.array(predictions)
    
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    
    i=1
    AUC = roc_auc_score(y_test==i,  y_prob[:,i])
    fpr, tpr, thresholds = roc_curve(y_test==i, y_prob[:,i])
    ax.plot(fpr, tpr, label = f'AUC = {AUC:.3f}')

    ax.set_title(f'ROC of DNN {pairing_method}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    plt.savefig(f'figures/ROC_DNN_{pairing_method}.png', facecolor='White', dpi=300, bbox_inches='tight')
    
    # Write results
    now = datetime.datetime.now()
    file_name = 'DNN_training_results.csv'
    data_dict = {'Pairing': [pairing_method],
             'Signal': [X_s.shape[0]],
             'Background': [X_b.shape[0]],
             'ACC': [results[1]],
             'AUC': [AUC],
             'time': [now],
            }
    
    df = pd.DataFrame(data_dict)
    if os.path.isfile(file_name):
        training_results_df = pd.read_csv(file_name)
        pd.concat([training_results_df, df], ignore_index=True).to_csv(file_name, index=False)
    else:
        df.to_csv(file_name, index=False)

if __name__ == '__main__':
    main()
