# THINGS TO try and TEST NOW:
# -------------------------------------------
# 1. VAE ARCHITECTURE
# 2. SIZE OF TRAINED DATASET
# 3. GAUSSIAN MIXTURES VS MNIST ETC.
# -------------------------------------------
from vae import VAE
from tsne_visual import TSNE
from fdc import plotting
from keras.datasets import mnist
import numpy as np
import time

def main():

    root = '/Users/alexandreday/Desktop/clustering-references/analysis/'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    """ Note here that everything should be scaled between 0 and 1 ! """
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train=np.reshape(x_train, (len(x_train), 784))
    x_test=np.reshape(x_test, (len(x_test), 784))

    model = VAE(drop_rate = 0.3, epoch = 50, model=2)
    #model.train(x_train[:10000], x_test)
    model.load()
    x_latent = model.predict(x_train[:10000])
    #print(x_latent.shape)
    #exit()
    x_tsne = TSNE(n_iter = 2000).fit_transform(x_latent)
    plotting.cluster_w_label(x_tsne, y_train[:10000])
    
if __name__ == "__main__":
    main() 