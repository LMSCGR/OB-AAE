library(scRNAseq)
##examples for preprocessing data

baron_mouse<-readRDS("./data/baron-mouse.rds")
rowData(baron_mouse)
library(zellkonverter)
writeH5AD(baron_mouse, file = "baron_mouse.h5ad")


campbell<-readRDS("./data/campbell.rds")
rowData(campbell)
colData(campbell)
library(zellkonverter)
writeH5AD(campbell, file = "campbell.h5ad")

klein<-readRDS("./data/klein.rds")
rowData(klein)
colData(klein)
library(zellkonverter)
writeH5AD(klein, file = "klein.h5ad")

wang<-readRDS("./data/wang.rds")
rowData(wang)
colData(wang)
library(zellkonverter)
writeH5AD(wang, file = "wang.h5ad")

xin<-readRDS("./data/xin.rds")
rowData(xin)
colData(xin)
library(zellkonverter)
writeH5AD(xin, file = "xin.h5ad")

sanderson<-readRDS("./data/sanderson.rds")
rowData(sanderson)
colData(sanderson)
library(zellkonverter)
writeH5AD(sanderson, file = "sanderson.h5ad")
          
goolam<-readRDS("./data/goolam.rds")
rowData(goolam)
colData(goolam)
library(zellkonverter)
writeH5AD(goolam, file = "goolam.h5ad")

deng<-readRDS("./data/deng-reads.rds")
assay(deng)
rowData(deng)
colData(deng)
library(zellkonverter)
writeH5AD(deng, file = "deng.h5ad")

yan<-readRDS("./data/yan.rds")
assay(yan)
rowData(yan)
colData(yan)
library(zellkonverter)
writeH5AD(yan, file = "yan.h5ad")

tmuris<-readRDS("./data/tmuris.rds")
rowData(tmuris)
colData(tmuris)
tmuris_mtx<-assay(tmuris)
tmuris_sce<-create_sce_from_counts(tmuris_mtx,colData(tmuris))
rowData(tmuris_sce)
library(zellkonverter)
writeH5AD(tmuris_sce, file = "tmuris.h5ad")


zilionis<-readRDS("./data/zilionis.rds")
rowData(zilionis)
colData(zilionis)
library(zellkonverter)
writeH5AD(zilionis, file = "zilionis.h5ad")

slyper<-readRDS("./data/slyper.rds")
rowData(slyper)
colData(slyper)
library(zellkonverter)
writeH5AD(slyper, file = "slyper.h5ad")

##qui
qui_sce<-readRDS(file="./data/qui_sce.rds")
library(scuttle)

library(scater)
logcounts(qui_sce) <- log2(calculateCPM(qui_sce) + 1)

rowData(qui_sce)$feature_symbol <- rownames(qui_sce)
rownames(qui_sce)
colData(qui_sce)
library(zellkonverter)
writeH5AD(qui_sce, file = "qui_sce.h5ad")
