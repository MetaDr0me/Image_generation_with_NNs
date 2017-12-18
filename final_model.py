import numpy as np
from scipy.stats import norm
from scipy import misc
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model, load_model
from keras import optimizers
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, Callback
from keras import objectives
import tensorflow as tf
import numpy as npM
import pandas as pd
from scipy.stats import norm
import os
import random
import csv

file_path = []
for file in os.listdir('Small_sized'):
    if file.endswith('.png'):
        path = 'Small_sized/' + file
        file_path.append(path)

df = pd.DataFrame()
df['path'] = file_path

data = []
for i in df['path']:
    img =  misc.imread(i, 0)
    data.append(img.flatten()/255.)

columns = []
for i in range(0, len(data)+1):
    columns.append('pixel' + str(i))

pixels = pd.DataFrame(data)

X = np.array(pixels)

random.shuffle(X)
X = X.reshape(X.shape[0], 100, 100,1)
x_train = X[:300]
x_test = X[300:]


#***********


# input image dimensions
img_rows, img_cols, img_chns = 100, 100, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

# can tune batch_size :
batch_size = 100

# set original img size
# if K.image_data_format() == 'channels_first':
#     original_img_size = (img_chns, img_rows, img_cols)
# else:
#     original_img_size = (img_rows, img_cols, img_chns)
original_img_size = (img_rows, img_cols, img_chns)


latent_dim = 2  # dimensions for adding variation
intermediate_dim = 300 #reduce input down to this num dimensions
epsilon_std = 1.0 #for variation calculation
epochs = 800

x = Input(shape=original_img_size)   # input image

# first convolutional layer, pad input to match output length
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
# second conv layer, tunable # filters, pad input again
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
# third conv layer, increase kernal_size to id bigger patterns
# still padding
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
# fourth conv layer, same params as third
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
# flatten
flat = Flatten()(conv_4)

# reduce dimensions down to specified num
hidden = Dense(intermediate_dim, activation='relu')(flat)

# feed output of 'intermediate' layers to the two variation
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)
#
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

#custom lambda layer combines the var layers
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# set up symmetrical to hidden layer on encoded side, where variation will be inserted
decoder_hid = Dense(intermediate_dim, activation='relu')
# set up the mechanism to increase the number of nodes
decoder_upsample = Dense(filters * 50 * 50, activation='relu')


output_shape = (batch_size, 50, 50, filters)


# set up Reshape layer to
decoder_reshape = Reshape(output_shape[1:])

# set up first two deconvolutional layer (inv of last two conv layers)
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')

output_shape = (batch_size, 101, 101, filters)

# third deconv layer is inv of second conv layer.
# padding is removed
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
# final deonv layer opp of first deconv layer.
# sigmoid activation.
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')
#
# adding variety at the point of lowest dimensionality
hid_decoded = decoder_hid(z)

# actually adding more nodes, building up decoder
up_decoded = decoder_upsample(hid_decoded)

# reshape the larger layers
reshape_decoded = decoder_reshape(up_decoded)

# actual decoding process
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
#
# custom loss. can use to fit with or without the Kullback–Leibler piece
def vae_loss(x, x_decoded_mean_squash):
    x = K.flatten(x)
    x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
#     kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss #K.mean(xent_loss + kl_loss)
#
#
#
# # build a model to project inputs on the latent space
encoder = Model(x, z_mean)
# # build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)
#
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        # rand1, rand2 = np.random.uniform(0.01,0.99,2)
        for i in range(0, 1):
            rand1, rand2 = np.random.uniform(0,1000,2)
            gen_seed = np.array([[rand1, rand2]])
            prediction = generator.predict(gen_seed)
            image = prediction.reshape(100, 100)
            file = 'prediction' + str(i) + 'on_epoch'+ str(epoch)+ '.png'
            misc.imsave(file,image)
            if i == 0:
                if epoch % 10 == 0:
                    rand_test = 96
                    px = vae.predict(x_test[rand_test].reshape(-1,100,100,1))
                    p_image = px.reshape(100,100)
                    pfile = 'x_test_' + str(rand_test) + file
                    misc.imsave(pfile, p_image)
                    x_file = 'x_test' + str(rand_test)  + 'on_epoch' + str(epoch) + '.png'
                    x_image = x_test[rand_test].reshape(100,100)
                    misc.imsave(x_file, x_image)
                else:
                    pass
            else:
                pass

history = LossHistory()
#
vae = Model(x, x_decoded_mean_squash)
opt_adam = optimizers.Adam(lr=0.0001)
vae.compile(optimizer=opt_adam, loss = vae_loss)
#
checkpointer= ModelCheckpoint('check.hdf5', monitor='val_loss', mode='min', save_best_only=True)
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='auto')
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
       callbacks=[history, checkpointer, earlyStopping])

vae.save_weights('model.h5')
df = pd.DataFrame(history.losses)
pd.DataFrame.to_csv(df, 'losses.csv', header=False)
