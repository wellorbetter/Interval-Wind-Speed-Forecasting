import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import copy

def my_bpnn(X, y, alpha1, alpha2):
    X = np.array(X)
    y = np.array(y)
    init_y = copy.deepcopy(y)
    alpha1 = float(alpha1)
    alpha2 = float(alpha2)

    def PICP(lower_bound, upper_bound, y):
        return np.sum(np.logical_and(y >= lower_bound, y <= upper_bound)) / len(y)

    def PINAW(lower_bound, upper_bound, y):
        return np.mean(upper_bound - lower_bound) / (np.max(y) - np.min(y))

    def CWC(picp, pinaw):
        # print(picp)
        # print(pinaw)
        if (picp >= 0.9):
            return pinaw
        else:
            return (0.1 + 6 * pinaw) * (1 + np.exp(-15 * (picp - 0.9)))

    class tfLoss(tf.keras.losses.Loss):
        def __init__(self, **kwargs):
            super(tfLoss, self).__init__(**kwargs)

        def call(self, y_true, y_pred):
            # 输入应该是batch_size * 2 
            return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))

    custom_loss = tfLoss()
    y1 = y * (1 - alpha1)
    y2 = y * (1 + alpha2)

    
    # 使用 reshape 将结果保存回 y1 和 y2
    y1 = y1.reshape(-1, 1)
    y2 = y2.reshape(-1, 1)

    y = np.concatenate((y1, y2), axis=1)

    X = tf.constant(X, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)

    model = Sequential()
    model.add(Dense(30, input_dim=1, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss=custom_loss)

    model.fit(X, y, epochs=30, batch_size=32, verbose=0)

    y_pred = model.predict(X)
    picp = PICP(y_pred[:, 0], y_pred[:, 1], init_y)
    pinaw = PINAW(y_pred[:, 0], y_pred[:, 1], init_y)
    cwc = CWC(picp, pinaw)

    # print(picp, pinaw, cwc)
    return 0.9 - picp if picp >= 0.9 else 1, pinaw
