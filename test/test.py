# THINGS TO try and TEST NOW:
# -------------------------------------------
# 1. VAE ARCHITECTURE
# 2. SIZE OF TRAINED DATASET
# 3. GAUSSIAN MIXTURES VS MNIST ETC.
# -------------------------------------------
from vae import VAE
from keras.datasets import mnist
import numpy as np
import time

def main():

    root = '/Users/alexandreday/Desktop/clustering-references/analysis/'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train=np.reshape(x_train, (len(x_train), 784))
    x_test=np.reshape(x_test, (len(x_test), 784))

    time.sleep(5)
    model = VAE()
    model.train(x_train[:10000],x_test)
    exit()


    name = 'noiseVAE_noise0p2.h5'
    fname = root+name # for loading vae model in case u need it
    #test(y_train[:10000])
    #exit()
    VAE = load_model(fname) 
    #VAE = train_model(x_train[:10000], x_test, root=root, name=name, drop_rate = 0.2, epoch = 500)
    #exit()
    xvae = VAE.predict(x_train[:10000])
    pickle.dump(xvae, open('xvae_test.pkl','wb'))
    
    exit()
    #xpca = PCA(n_components=80).fit_transform(xvae)
    xtsne = tsne_model(x_train[:10000], savefile = 'test.pkl', n_iter = 5000)
    #xtsne = tsne_model(xvae, savefile = 'tsne_drop0p2_angle0p5.pkl', n_iter = 5000)
    plot_w_label(xtsne, y_train[:10000])

if __name__ == "__main__":
    main()