from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import os
from tensorflow.keras.callbacks import TensorBoard
import time
from data.data_preprocess import Dataset

class Model:
    def __init__(self, epochs):
        self.input_shape = (48, 48, 3)
        self.batch_size = 64
        self.epochs = epochs
        self.verbose = 2
        self.tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.input_shape
        ))

        model.add(Conv2D(
            64,
            (3, 3),
            activation='relu'
        ))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(
            64,
            (3, 3),
            activation='relu'
        ))

        model.add(Conv2D(
            32,
            (3, 3),
            activation='relu'
        ))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(
            optimizer='adam',
            metrics=['accuracy'],
            loss='categorical_crossentropy'
        )

        self.model = model

    def train(self, images, labels):
        self.history = self.model.fit(
            images, labels,
            batch_size=self.batch_size,
            verbose=self.verbose,
            epochs=self.epochs,
            callbacks=[self.tensorboard]
        )

        self.model.save(os.path.join(os.path.dirname(__file__), 'model', 'emotion{}.h5'.format(time.time())))

    