import config
from preprocess import get_modified_data
from AFM import AFM

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import MeanSquaredError
from time import perf_counter
import os
import argparse


parser = argparse.ArgumentParser(description='afm')
parser.add_argument('--epochs', type= int, default= 50, help='') 
parser.add_argument('--lr', type= float, default= 1e-3, help='') 
args = parser.parse_args()

def get_data():
    train = pd.read_csv(os.path.join("..","..","data","YN_afm_df2.csv"))
    test = pd.read_csv(os.path.join("..","..","data","locationsinfo.csv"))
    test = test.drop(columns=['place.name'], axis=1)
    X_train = train.iloc[:, 0:7]
    X_test = test.iloc[:, 0:7]
    Y_train = train.iloc[:, 7] 
    # Y_test = [0]*test.shape[0]

    X_train.columns = config.ORIGINAL_FIELDS
    X_train_modified, num_feature_train = get_modified_data(X_train, config.CONT_FIELDS, config.CAT_FIELDS)
    X_test_modified, num_feature_test = get_modified_data(X_test, config.CONT_FIELDS, config.CAT_FIELDS)
    num_feature = num_feature_train #+ num_feature_test

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train_modified.values, tf.float32), tf.cast(Y_train, tf.float32))) \
        .shuffle(30000).batch(config.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test_modified.values, tf.float32))) \
        .shuffle(10000).batch(config.BATCH_SIZE)

    return train_ds, test_ds, num_feature


def train_on_batch(model, optimizer, mse, inputs, targets):
    regularizer = tf.keras.regularizers.l2(0.01)

    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.MSE(
             y_true=targets, y_pred=y_pred) + regularizer(model.w)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # mse
    mse.update_state(targets, y_pred)

    return loss


def train(epochs):
    train_ds, test_ds, num_feature = get_data()

    model = AFM(config.NUM_FIELD, num_feature, config.NUM_CONT,
                config.EMBEDDING_SIZE, config.HIDDEN_SIZE)

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)

    print("Start Training: Batch Size: {}, Embedding Size: {}, Hidden Size: {}".\
        format(config.BATCH_SIZE, config.EMBEDDING_SIZE, config.HIDDEN_SIZE))
    start = perf_counter()
    for i in range(epochs):
        mse = MeanSquaredError()
        loss_history = []

        for x, y in train_ds:
            loss = train_on_batch(model, optimizer, mse, x, y)
            loss_history.append(loss)
 
        print("Epoch {:03d}: 누적 Loss: {:.4f}, mse: {:.4f}".format(
            i, np.mean(loss_history), mse.result().numpy()))

    y_preds = []
    for x in test_ds:
        y_preds.append(model(x))

    return y_preds



if __name__ == '__main__':
    y_preds = train(epochs=args.epochs)  
    locationsinfo = pd.read_csv(os.path.join("..","..","data","locationsinfo.csv"))
    locationsinfo['rating'] = y_preds
    locationsinfo.to_csv(os.path.join("..","..","data","locationsinfo_preds.csv"), index=False)
    print('save locationsinfo_preds')