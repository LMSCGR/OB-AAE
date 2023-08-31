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

muscle_df=sc.read_h5ad("qui_DellOrso.h5ad")
muscle_df_sel=muscle_df


muscle_df

highly_genes=15000
muscle_df_sel.var_names_make_unique()

sc.pp.filter_genes(muscle_df_sel, min_cells=3)
sc.pp.normalize_per_cell(muscle_df_sel, counts_per_cell_after=1e4)
sc.pp.log1p(muscle_df_sel)
muscle_df_sel.raw = muscle_df_sel
sc.pp.highly_variable_genes(muscle_df_sel, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
muscle_df_sel = muscle_df_sel[:, muscle_df_sel.var['highly_variable']].copy()

counts_org=muscle_df_sel.X
cellinfo_org=pd.DataFrame(muscle_df_sel.obs['time_points'])
geneinfo_org=pd.DataFrame(muscle_df_sel.var['feature_symbol'])

adata_org = sc.AnnData(counts_org,obs=cellinfo_org,var=geneinfo_org)
adata_org


# ## AAE

aae_muscle_df=sc.read_h5ad("aae_qui_DellOrso_g15000.h5ad")
adata_ae=aae_muscle_df


# ### compare

adata_org


adata_ae


sc_org=adata_org
sc_aae=adata_ae


sc_org.obs['type']="Org"
sc_aae.obs['type']="DB-AAE"

sc_qui_60h=sc_org.concatenate(sc_aae)


sc_qui_60h


sc_qui_60h.obs['type']=sc_qui_60h.obs['type'].astype('category')
sc_qui_60h.obs['type']

sc_qui_60h.obs['type'] = sc_qui_60h.obs['type'].cat.reorder_categories(['Org', 'DB-AAE'], ordered=True)


sc_qui=sc_qui_60h[sc_qui_60h.obs["time_points"] == "qui"]

sc_qui.obs["time_points"]="total_Q"


markers = ['Maff','Msc','Zfp36','Stat3','Fxyd1','Gfpt2','Slc41a1']

sc.pl.stacked_violin(sc_qui, markers,groupby=['type','time_points'], swap_axes=True,cmap='Reds',colorbar_title='Median expression')





