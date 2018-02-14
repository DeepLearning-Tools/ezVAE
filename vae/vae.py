from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np
from keras import backend as K
from keras.models import load_model as lmm
import tensorflow as tf
from . import utility as ut

class VAE:
    
    def __init__(self, name=None, drop_rate = 0.7, epoch=50, dense_size=1000, batch_size=256, model=1, root = "models/"):
        self.name = name
        self.drop_rate = drop_rate
        self.epoch = epoch
        self.dense_size = dense_size
        self.batch_size = batch_size
        self.model = model
        self.info()
        if self.name is None:
            self.name = root+ut.make_file_name(self.__dict__)
    
    def info(self):
        print(self.__dict__)

    def train(self, x_train, x_test):
        self.input_dimension = x_train.shape[1]

        ### -> building network
        input_img = Input(shape=(self.input_dimension,))  # adapt this if using `channels_first` image data format
        x = Dropout(self.drop_rate)(input_img)
        encoded = Dense(self.dense_size, activation='relu', name='encoder')(x) # need to save weigths and biases of this
        if self.model == 1:
            x = Dropout(self.drop_rate)(encoded)
            decoded = Dense(self.input_dimension, activation='relu')(x)
        elif self.model == 2:
            decoded = Dense(self.input_dimension, activation='relu')(encoded) # decode directly ?!

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # here pixels are set between 0 and 1.

        #autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        autoencoder.fit(x_train, x_train,
                        epochs = self.epoch,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test)
        )

        # getting rid of decoding and noising layers, return only the dense layer.
        layer = autoencoder.get_layer(name='encoder')
        input_img = Input(shape=(self.input_dimension,))
        encoder = Dense(self.dense_size, activation='relu', name='first')(input_img)
        encoder_model = Model(input=input_img, output=encoder)
        encoder_model.set_weights([layer.get_weights()[0]*(1-self.drop_rate),layer.get_weights()[1]]) # rescaling weights here
        encoder_model.save(self.name) # use encoder_model.predict(x) to output a 1000 dim representation
        self.encoder_model = encoder_model

    def predict(self, x):
        return self.encoder_model.predict(x)

    def load(self, name=None):
        if name is None:
            self.encoder_model = lmm(self.name)
        else:
            self.encoder_model = lmm(name)
    
