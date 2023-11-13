#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import random
random.seed(1234)


# ## goolam

# In[ ]:


goolam= sc.read_h5ad("./data/goolam.h5ad")
goolam_sel=goolam


# In[ ]:


highly_genes=5000
goolam_sel.var_names_make_unique()
# sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(goolam_sel, min_cells=3)
sc.pp.normalize_per_cell(goolam_sel, counts_per_cell_after=1e4)
sc.pp.log1p(goolam_sel)
goolam_sel.raw = goolam_sel
sc.pp.highly_variable_genes(goolam_sel, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
goolam_sel = goolam_sel[:, goolam_sel.var['highly_variable']].copy()


# In[ ]:


#counts=sim_time.layers['counts']
counts=goolam_sel.X
cellinfo=pd.DataFrame(goolam_sel.obs['cell_type1'])
#groupinfo=pd.DataFrame(sim_time.obs['cluster'])
#LibSizeinfo=pd.DataFrame(sim_g3_no_dropout.obs['ExpLibSize'])
geneinfo=pd.DataFrame(goolam_sel.var['feature_symbol'])


# In[ ]:


p_count=pd.DataFrame(counts)


# In[ ]:


p_count.index=cellinfo.index
p_count.columns=geneinfo.index


# In[ ]:


p_count


# In[ ]:


m, n = p_count.shape
print('the num. of cell = {}'.format(m))
print('the num. of genes = {}'.format(n))


# In[ ]:


adata = sc.AnnData(counts,obs=cellinfo,var=geneinfo)
#adata.obs['Group']=groupinfo
#adata.obs['ExpLibSize']=LibSizeinfo
#sc.pp.filter_genes(adata, min_counts=1)#
#adata.obsm['tsne']=tsne
#adata.obsm['X_pca']=pca
adata


# In[ ]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation,BatchNormalization,GaussianNoise
from keras.optimizers import SGD,RMSprop,Adam
from tensorflow.keras.layers import LeakyReLU,Reshape
import keras_tuner as kt
import tensorflow as tf


# ## Define Encoder

# In[ ]:


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


# ## Define Decoder

# In[ ]:


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


# ## Define Discriminator

# In[ ]:


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


# In[ ]:


import time
#n1 = 41428
n1=5000
latent_dim = 512
#n1 = 23300
n2=1024
n3=512
n4=latent_dim
batch_size = 32
#n_epochs = 100
n_epochs = 20
X_train = p_count
x=X_train.to_numpy()
activation='relu'
#activation='PReLU'
#activation='ELU'
start_time = time.time()


# In[ ]:


#optimizer = Adam(0.0002, 0.5)
optimizer = RMSprop(learning_rate=0.0002)
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


# In[ ]:


valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

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
    print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))


# In[ ]:


rlt_dir='tunes/'


# In[ ]:


adversarial_autoencoder.summary()


# In[ ]:


counts_org=goolam.X
cellinfo_org=pd.DataFrame(goolam.obs['cell_type1'])
geneinfo_org=pd.DataFrame(goolam.var['feature_symbol'])


# In[ ]:


adata_org = sc.AnnData(counts_org,obs=cellinfo_org,var=geneinfo_org)
adata_org


# In[ ]:


counts_ae = adversarial_autoencoder.predict(X_train)


# In[ ]:


counts_ae[0]


# In[ ]:


adata_ae = sc.AnnData(counts_ae[0],obs=cellinfo,var=geneinfo)


# In[ ]:


sc.pp.normalize_per_cell(adata_org)
sc.pp.log1p(adata_org)
sc.pp.pca(adata_org)


# In[ ]:


sc.pl.pca_variance_ratio(adata_org)


# In[ ]:


sc.pp.normalize_per_cell(adata_ae)
sc.pp.log1p(adata_ae)
sc.pp.pca(adata_ae)


# In[ ]:


sc.pl.pca_variance_ratio(adata_ae)


# In[ ]:


sc.tl.tsne(adata_org, n_pcs = 10)


# In[ ]:


sc.pp.neighbors(adata_org, n_neighbors=60, n_pcs=10)


# In[ ]:


sc.tl.umap(adata_org)


# In[ ]:


sc.pl.tsne(adata_org, color='cell_type1', title='Original counts')


# In[ ]:


sc.pl.umap(adata_org, color='cell_type1', title='Original counts')


# In[ ]:


sc.pp.neighbors(adata_ae, n_neighbors=60, n_pcs=10)


# In[ ]:


sc.tl.tsne(adata_ae, n_pcs = 10)


# In[ ]:


sc.tl.umap(adata_ae)


# In[ ]:


sc.pl.tsne(adata_ae, color='cell_type1',title='Reconstructed counts')


# In[ ]:


sc.pl.umap(adata_ae, color='cell_type1',title='Reconstructed counts')


# In[ ]:




