#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from datetime import datetime
from sklearn.manifold import TSNE
from absl import flags
import pandas as pd
import seaborn as sns
import scanpy as sc

import random
random.seed(1234)

q_60h_p= sc.read_h5ad("qui_DellOrso.h5ad")
q_60h_p_sel=q_60h_p

q_60h_p

from numpy import count_nonzero
count_org=q_60h_p.X
# calculate sparsity
sparsity = 1.0 - ( count_nonzero(count_org) / float(count_org.size) )
print(sparsity)

highly_genes=8000
q_60h_p_sel.var_names_make_unique()

sc.pp.filter_genes(q_60h_p_sel, min_cells=3)
sc.pp.normalize_per_cell(q_60h_p_sel, counts_per_cell_after=1e4)
sc.pp.log1p(q_60h_p_sel)
q_60h_p_sel.raw = q_60h_p_sel
sc.pp.highly_variable_genes(q_60h_p_sel, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
q_60h_p_sel = q_60h_p_sel[:, q_60h_p_sel.var['highly_variable']].copy()


counts=q_60h_p_sel.X
cellinfo=pd.DataFrame(q_60h_p_sel.obs['time_points'])
geneinfo=pd.DataFrame(q_60h_p_sel.var['feature_symbol'])

cellinfo.time_points


p_count=pd.DataFrame(counts)

p_count.index=cellinfo.index
p_count.columns=geneinfo.index


m, n = p_count.shape
print('the num. of cell = {}'.format(m))
print('the num. of genes = {}'.format(n))

adata = sc.AnnData(counts,obs=cellinfo,var=geneinfo)

adata


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation,BatchNormalization,GaussianNoise
from keras.optimizers import SGD,RMSprop,Adam,Adagrad
from tensorflow.keras.layers import LeakyReLU,Reshape

import tensorflow as tf


# ##Define Encoder
# 
# ---
# 
# 

def build_encoder(n1,n2,n3,n4,activation):
        # Encoder

        expr_in = Input( shape=(n1,) )

        
        h = Dense(n2)(expr_in)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(n3)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(n4)(h)
        log_var = Dense(n4)(h)
        latent_repr = tf.keras.layers.Add()([mu, log_var])


        return Model(expr_in, latent_repr)

def build_decoder(n1,n2,n3,n4,activation):
        
        model = Sequential()

        model.add(Dense(n3, input_dim=n4))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(n2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod((n1,)), activation='relu'))
        model.add(Reshape((n1,)))

        model.summary()

        z = Input(shape=(n4,))
        expr_lat = model(z)

        return Model(z, expr_lat)

def build_discriminator(n1,n2,n3,n4,activation):

        model = Sequential()

        model.add(Dense(n3, input_dim=n4))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="relu"))
        model.summary()

        encoded_repr = Input(shape=(n4, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)


import time

n1=highly_genes
latent_dim = 512

n2=1024
n3=512
n4=latent_dim
batch_size = 32

n_epochs = 50
X_train = p_count
x=X_train.to_numpy()
activation='relu'
#activation='PReLU'
#activation='ELU'
start_time = time.time()

optimizer = RMSprop(learning_rate=0.00002)

# Build and compile the discriminator
discriminator = build_discriminator(n1,n2,n3,n4,activation)
discriminator.compile(loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

# Build the encoder / decoder
encoder = build_encoder(n1,n2,n3,n4,activation)
decoder = build_decoder(n1,n2,n3,n4,activation)

autoencoder_input = Input(shape=(n1,))
reconstructed_input = Input(shape=(n1,))
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
encoded_repr = encoder(autoencoder_input)
reconstructed = decoder(encoded_repr)

# For the adversarial_autoencoder model we will only train the generator
discriminator.trainable = False

# The discriminator determines validity of the encoding
validity = discriminator(encoded_repr)

# The adversarial_autoencoder model  (stacked generator and discriminator)
adversarial_autoencoder = Model(autoencoder_input, [reconstructed, validity])
adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))
past = datetime.now()
for epoch in np.arange(1, n_epochs + 1):
    for i in range(int(len(x) / batch_size)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
        batch = x[i*batch_size:i*batch_size+batch_size]

    
        latent_fake = encoder.predict(batch)
        latent_real = np.random.normal(size=(batch_size, latent_dim))

            # Train the discriminator
        d_loss_real = discriminator.train_on_batch(latent_real, valid)
        d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
        g_loss = adversarial_autoencoder.train_on_batch(batch, [batch, valid])

            # Plot the progress
    now = datetime.now()
    print("\nEpoch {}/{} - {:.1f}s".format(epoch, n_epochs, (now - past).total_seconds()))
    print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
    past = now


adversarial_autoencoder.summary()

counts_org=q_60h_p.X
cellinfo_org=pd.DataFrame(q_60h_p.obs['time_points'])
geneinfo_org=pd.DataFrame(q_60h_p.var['feature_symbol'])

adata_org = sc.AnnData(counts_org,obs=cellinfo_org,var=geneinfo_org)
adata_org

counts_ae = adversarial_autoencoder.predict(X_train)

counts_ae[0]

adata_ae = sc.AnnData(counts_ae[0],obs=cellinfo,var=geneinfo)

adata_ae.write("./data/aae_qui_DellOrso.h5ad")

rlt_dir='./model/'

adversarial_autoencoder.save(rlt_dir+'{}_q_60h_p_relu_autoencoder_RMSprop_gene8000_e50.h5'.format("aae"))



