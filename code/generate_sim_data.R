library(splatter)
library(scater)
library(Seurat)
library(patchwork)
library(DESeq2)
library(zellkonverter)

##group= 2 no dropout
nGroups=2
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
writeH5AD(sim, file = "sim_g2_no_dropout.h5ad")

##group= 2 dropout=1
nGroups=2
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=1) 
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
writeH5AD(sim, file = "sim_g2_dropout_1.h5ad")

##group= 6 no dropout
nGroups=6
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
writeH5AD(sim, file = "sim_g6_no_dropout.h5ad")

##group= 6 dropout=1
nGroups=6
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=1) 
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
writeH5AD(sim, file = "sim_g6_dropout_1.h5ad")

##group= 8 no dropout
nGroups=8
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",seed=42) 
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
writeH5AD(sim, file = "sim_g6_no_dropout.h5ad")

##group= 8 dropout=1
nGroups=8
group.prob <- rep(1, nGroups) / nGroups
sim <- splatSimulate(group.prob=group.prob,nGenes=200,batchCells=2000,method = "groups",dropout.type='experiment',seed=42, dropout.shape=-1, dropout.mid=1) 
counts<-counts(sim)
libsizes <- colSums(counts)
size.factors <- libsizes/mean(libsizes)
logcounts(sim) <- log2(t(t(counts)/size.factors) + 1)
writeH5AD(sim, file = "sim_g6_dropout_1.h5ad")
