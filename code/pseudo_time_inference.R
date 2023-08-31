###pseudo time inference
library(zellkonverter)
goolam_sce<-readRDS("./data/goolam.rds")

colData(goolam_sce)$cell_type1<-ordered(colData(goolam_sce)$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))
cell.stages= c("2cell", "4cell","8cell","16cell","blast")

label=ordered(goolam_sce$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(Polychrome)
set.seed(723451) # for reproducibility
my_color <- createPalette(10, c("#010101", "#ff0000"), M=1000)
names(my_color) <- unique(as.character(goolam_sce$cell_type1))


library("slingshot")
library("scater")

goolam_sce <- runPCA(goolam_sce, ncomponents = 20)

goolam_sce <- runUMAP(goolam_sce)

goolam_sce <- slingshot(goolam_sce,reducedDim = "PCA")
embedded <- embedCurves(goolam_sce, "UMAP")
embedded <- slingCurves(embedded)[[1]] # only 1 path.
embedded <- data.frame(embedded$s[embedded$ord,])

plotUMAP(goolam_sce, colour_by="cell_type1") +
  geom_path(data=embedded, aes(x=Dim.1, y=Dim.2), size=1.2)+
  scale_colour_manual(values = my_color)+
  theme(text = element_text(size = 18))

slingshot_df <- data.frame(slingPseudotime_1=goolam_sce@colData$slingPseudotime_1,cell_type=goolam_sce@colData$cell_type1)

library(ggplot2)
library(ggbeeswarm)
library(ggthemes)

ggplot(slingshot_df, aes(x = slingPseudotime_1, y = cell_type,
                         colour = cell_type)) +
  geom_quasirandom(groupOnX = FALSE) + scale_color_tableau() + theme_classic() +
  xlab("Pseudotime") + ylab("cell type") +
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 20))

cor(slingshot_df$slingPseudotime_1,as.numeric(factor(label, levels = cell.stages)))^2

###AAE
library(zellkonverter)
aae_goolam_sce<-readH5AD("./data/aae_goolam.h5ad")
logcounts(aae_goolam_sce)<-assay(aae_goolam_sce)

aae_goolam_sce@colData$cell_type1<-ordered(aae_goolam_sce@colData$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))
cell.stages= c("2cell", "4cell","8cell","16cell","blast")
label=ordered(aae_goolam_sce$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(Polychrome)
set.seed(723451) # for reproducibility
my_color <- createPalette(10, c("#010101", "#ff0000"), M=1000)
names(my_color) <- unique(as.character(aae_goolam_sce$cell_type1))
library("slingshot")
library("scater")

aae_goolam_sce <- runPCA(aae_goolam_sce, ncomponents = 20)

aae_goolam_sce <- runUMAP(aae_goolam_sce)
aae_goolam_sce <- slingshot(aae_goolam_sce,reducedDim = "PCA")

embedded <- embedCurves(aae_goolam_sce, "UMAP")
embedded <- slingCurves(embedded)[[1]] # only 1 path.
embedded <- data.frame(embedded$s[embedded$ord,])

plotUMAP(aae_goolam_sce, colour_by="cell_type1") +
  geom_path(data=embedded, aes(x=Dim.1, y=Dim.2), size=1.2)+
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 18))

slingshot_df <- data.frame(slingPseudotime_1=aae_goolam_sce@colData$slingPseudotime_1,cell_type=aae_goolam_sce@colData$cell_type1)
slingshot_df$cell_type <- ordered(slingshot_df$cell_type, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(ggplot2)
library(ggbeeswarm)
library(ggthemes)

ggplot(slingshot_df, aes(x = slingPseudotime_1, y = cell_type,
                         colour = cell_type)) +
  geom_quasirandom(groupOnX = FALSE) + scale_color_tableau() + theme_classic() +
  xlab("Pseudotime") + ylab("cell type") +
  #  ggtitle("Cells ordered by Slingshot pseudotime")+
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 20))

cor(slingshot_df$slingPseudotime_1,as.numeric(factor(label, levels = cell.stages)))^2

###DCA
library(zellkonverter)
dca_goolam_sce<-readH5AD("./data/dca_goolam.h5ad")
logcounts(dca_goolam_sce)<-assay(dca_goolam_sce)

dca_goolam_sce@colData$cell_type1<-ordered(dca_goolam_sce@colData$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))
cell.stages= c("2cell", "4cell","8cell","16cell","blast")
label=ordered(dca_goolam_sce$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(Polychrome)
set.seed(723451) # for reproducibility
my_color <- createPalette(10, c("#010101", "#ff0000"), M=1000)
names(my_color) <- unique(as.character(dca_goolam_sce$cell_type1))
library("slingshot")
library("scater")

dca_goolam_sce <- runPCA(dca_goolam_sce, ncomponents = 20)

dca_goolam_sce <- runUMAP(dca_goolam_sce)
dca_goolam_sce <- slingshot(dca_goolam_sce,reducedDim = "PCA")

embedded <- embedCurves(dca_goolam_sce, "UMAP")
embedded <- slingCurves(embedded)[[1]] # only 1 path.
embedded <- data.frame(embedded$s[embedded$ord,])

plotUMAP(dca_goolam_sce, colour_by="cell_type1") +
  geom_path(data=embedded, aes(x=Dim.1, y=Dim.2), size=1.2)+
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 18))

slingshot_df <- data.frame(slingPseudotime_1=dca_goolam_sce@colData$slingPseudotime_1,cell_type=dca_goolam_sce@colData$cell_type1)
slingshot_df$cell_type <- ordered(slingshot_df$cell_type, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(ggplot2)
library(ggbeeswarm)
library(ggthemes)

ggplot(slingshot_df, aes(x = slingPseudotime_1, y = cell_type,
                         colour = cell_type)) +
  geom_quasirandom(groupOnX = FALSE) + scale_color_tableau() + theme_classic() +
  xlab("Pseudotime") + ylab("cell type") +
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 20))

cor(slingshot_df$slingPseudotime_1,as.numeric(factor(label, levels = cell.stages)))^2

###scvi
library(zellkonverter)
scvi_goolam_sce<-readH5AD("./data/scvi_goolam.h5ad")
logcounts(scvi_goolam_sce)<-assay(scvi_goolam_sce)

scvi_goolam_sce@colData$cell_type1<-ordered(scvi_goolam_sce@colData$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))
cell.stages= c("2cell", "4cell","8cell","16cell","blast")
label=ordered(scvi_goolam_sce$cell_type1, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(Polychrome)
set.seed(723451) # for reproducibility
my_color <- createPalette(10, c("#010101", "#ff0000"), M=1000)
names(my_color) <- unique(as.character(scvi_goolam_sce$cell_type1))
library("slingshot")
library("scater")

scvi_goolam_sce <- runPCA(scvi_goolam_sce, ncomponents = 20)

scvi_goolam_sce <- runUMAP(scvi_goolam_sce)
scvi_goolam_sce <- slingshot(scvi_goolam_sce,reducedDim = "PCA")

embedded <- embedCurves(scvi_goolam_sce, "UMAP")
embedded <- slingCurves(embedded)[[1]] # only 1 path.
embedded <- data.frame(embedded$s[embedded$ord,])

plotUMAP(scvi_goolam_sce, colour_by="cell_type1") +
  geom_path(data=embedded, aes(x=Dim.1, y=Dim.2), size=1.2)+
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 18))

slingshot_df <- data.frame(slingPseudotime_1=scvi_goolam_sce@colData$slingPseudotime_1,cell_type=scvi_goolam_sce@colData$cell_type1)
slingshot_df$cell_type <- ordered(slingshot_df$cell_type, levels = c("2cell", "4cell","8cell","16cell","blast"))

library(ggplot2)
library(ggbeeswarm)
library(ggthemes)

ggplot(slingshot_df, aes(x = slingPseudotime_1, y = cell_type,
                         colour = cell_type)) +
  geom_quasirandom(groupOnX = FALSE) + scale_color_tableau() + theme_classic() +
  xlab("Pseudotime") + ylab("cell type") +
  scale_colour_manual(values = my_color)+theme(text = element_text(size = 20))

cor(slingshot_df$slingPseudotime_1,as.numeric(factor(label, levels = cell.stages)))^2
