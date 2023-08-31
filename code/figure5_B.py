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




qui= sc.read_h5ad("qui_sce.h5ad")
qui_sel=qui


highly_genes=9000
qui_sel.var_names_make_unique()
# sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(qui_sel, min_cells=3)
sc.pp.normalize_per_cell(qui_sel, counts_per_cell_after=1e4)
sc.pp.log1p(qui_sel)
qui_sel.raw = qui_sel
sc.pp.highly_variable_genes(qui_sel, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
qui_sel = qui_sel[:, qui_sel.var['highly_variable']].copy()


counts_org=qui_sel.X
cellinfo_org=pd.DataFrame(qui_sel.obs['time_points'])
geneinfo_org=pd.DataFrame(qui_sel.var['feature_symbol'])



adata_org = sc.AnnData(counts_org,obs=cellinfo_org,var=geneinfo_org)
adata_org


# ## AAE




aae_qui=sc.read_h5ad("../aae_qui_g9000.h5ad")
adata_ae=aae_qui


# ### compare methods

adata_org


adata_ae


sc_org=adata_org
sc_aae=adata_ae


sc_org.obs['type']="Org"
sc_aae.obs['type']="DB-AAE"


sc_qui=sc_org.concatenate(sc_aae)


sc_qui.obs['type']=sc_qui.obs['type'].astype('category')
sc_qui.obs['type']

 mask3 = (adata.obs["louvain"] == "1") | (adata.obs["louvain"] == "2") final3 = adata[mask3].copy()

sc_qui.obs['type'] = sc_qui.obs['type'].cat.reorder_categories(['Org', 'DB-AAE'], ordered=True)



sc_cq=sc_qui[sc_qui.obs["time_points"] == "cQ"]


sc_cq.obs["time_points"]="Close_Q"

markers = ['Gas1','Fosl2','Dag1','Zfp36','Azin1','Tuba1a','Nfib']

sc.pl.stacked_violin(sc_cq, markers,groupby=['type','time_points'], swap_axes=True,cmap='Reds',colorbar_title='Median expression')





