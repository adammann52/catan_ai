import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers


class QNetwork():

    def __init__(self,lr):

        optim = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam',)

        model = models.Sequential()
        model.add(layers.Conv2D(15,(3,3),activation='relu',input_shape = (21,21,15)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer = optim)
        self.model = model

    def train(self,states,targets):

        states = np.array([np.array([np.array([np.array(row) for row in channel]) for channel in state]) for state in           states])
        states = np.transpose(states,(0,2,3,1))
        return self.model.fit(states,np.array(targets),verbose = 0)

    def evaluate(self,next_states):
        states = np.array([np.array([np.array([np.array(row) for row in channel]) for channel in state]) for state in next_states])
        states = np.transpose(states,(0,2,3,1))
        return self.model.predict(states)


