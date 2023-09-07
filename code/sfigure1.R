###org
library(Seurat)
library(zellkonverter)
library(scater)
baron_mouse<-readH5AD("../org_baron_mouse_louvain.h5ad")
counts(baron_mouse)<-assay(baron_mouse)
logcounts(baron_mouse)<-assay(baron_mouse)
baron_mouse <- runPCA(baron_mouse)
baron_mouse <- runUMAP(baron_mouse)
baron_mouse.seurat <- as.Seurat(baron_mouse)
baron_mouse.seurat@meta.data
baron.all.genes<- rownames(baron_mouse.seurat)
baron_mouse.seurat<- ScaleData(baron_mouse.seurat, features = baron.all.genes)
baron_mouse.seurat<- RunPCA(object = baron_mouse.seurat, features = baron.all.genes, verbose = FALSE)
ElbowPlot(object = baron_mouse.seurat,ndims =50)

baron_mouse.seurat<- FindNeighbors(object = baron_mouse.seurat, dims = 1:30)
baron_mouse.seurat<- FindClusters(object = baron_mouse.seurat, resolution = 0.25)
baron_mouse.seurat<- RunUMAP(object = baron_mouse.seurat, dims = 1:30)
Idents(baron_mouse.seurat)<-baron_mouse.seurat$louvain
DimPlot(baron_mouse.seurat, reduction = "umap")
baron_mouse.markers<- FindAllMarkers(baron_mouse.seurat, only.pos = TRUE,test.use ="MAST" ,min.pct = 0.2, logfc.threshold = 0.2)
##GO analysis
library(clusterProfiler)
library(org.Mm.eg.db)
baron_mouse.markers_g1<-subset(baron_mouse.markers,baron_mouse.markers$cluster==1)

gene.df <- bitr(baron_mouse.markers_g1$gene, fromType = "SYMBOL",
                toType = c("ENSEMBL", "ENTREZID"),
                OrgDb = org.Mm.eg.db)
ego_g1 <- enrichGO(gene          = unique(gene.df$ENTREZID),
                   OrgDb         = org.Mm.eg.db,
                   ont           = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.01,
                   qvalueCutoff  = 0.05)

tiff(filename = "ego_g1_mast_MF.tiff",res = 300,
     width = 1800, height = 2800)
dotplot(ego_g1, showCategory=20)
dev.off()


baron_mouse.markers_g0<-subset(baron_mouse.markers,baron_mouse.markers$cluster==0)

gene.df <- bitr(baron_mouse.markers_g0$gene, fromType = "SYMBOL",
                toType = c("ENSEMBL", "ENTREZID"),
                OrgDb = org.Mm.eg.db)
ego_g0 <- enrichGO(gene          = unique(gene.df$ENTREZID),
                   OrgDb         = org.Mm.eg.db,
                   ont           = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.01,
                   qvalueCutoff  = 0.05)
test<-as.data.frame(ego_g0)
tiff(filename = "ego_g0_mast_MF.tiff",res = 300,
     width = 1800, height = 2800)
dotplot(ego_g0, showCategory=20)
dev.off()
baron_mouse.markers_g3<-subset(baron_mouse.markers,baron_mouse.markers$cluster==3)

gene.df <- bitr(baron_mouse.markers_g3$gene, fromType = "SYMBOL",
                toType = c("ENSEMBL", "ENTREZID"),
                OrgDb = org.Mm.eg.db)
ego_g3 <- enrichGO(gene          = unique(gene.df$ENTREZID),
                      OrgDb         = org.Mm.eg.db,
                      ont           = "MF",
                      pAdjustMethod = "BH",
                      pvalueCutoff  = 0.01,
                      qvalueCutoff  = 0.05)

tiff(filename = "ego_g3_mast_MF.tiff",res = 300,
     width = 1800, height = 2800)
dotplot(ego_g3, showCategory=15)
dev.off()


###aae
library(Seurat)
library(zellkonverter)
library(scater)
aae_baron_mouse<-readH5AD("../aae_baron_mouse_louvain.h5ad")
counts(aae_baron_mouse)<-assay(aae_baron_mouse)
logcounts(aae_baron_mouse)<-assay(aae_baron_mouse)
aae_baron_mouse <- runPCA(aae_baron_mouse)
aae_baron_mouse <- runUMAP(aae_baron_mouse)
aae_baron_mouse.seurat <- as.Seurat(aae_baron_mouse)
aae_baron_mouse.seurat@meta.data
aae_baron.all.genes<- rownames(aae_baron_mouse.seurat)
aae_baron_mouse.seurat<- ScaleData(aae_baron_mouse.seurat, features = aae_baron.all.genes)
aae_baron_mouse.seurat<- RunPCA(object = aae_baron_mouse.seurat, features = aae_baron.all.genes, verbose = FALSE)
ElbowPlot(object = aae_baron_mouse.seurat,ndims =50)

aae_baron_mouse.seurat<- FindNeighbors(object = aae_baron_mouse.seurat, dims = 1:30)
aae_baron_mouse.seurat<- FindClusters(object = aae_baron_mouse.seurat, resolution = 0.25)
aae_baron_mouse.seurat<- RunTSNE(object = aae_baron_mouse.seurat, dims = 1:30)
aae_baron_mouse.seurat<- RunUMAP(object = aae_baron_mouse.seurat, dims = 1:30)
DimPlot(object=aae_baron_mouse.seurat,reduction='umap',label=T)
Idents(aae_baron_mouse.seurat)<-aae_baron_mouse.seurat$louvain
DimPlot(aae_baron_mouse.seurat, reduction = "umap",label=T)
aae_baron_mouse.markers<- FindAllMarkers(aae_baron_mouse.seurat, only.pos = TRUE, test.use ="MAST",min.pct = 1e-10, logfc.threshold = 1e-10)
##GO analysis
library(clusterProfiler)
library(org.Mm.eg.db)
aae_baron_mouse.markers_g0<-subset(aae_baron_mouse.markers,aae_baron_mouse.markers$cluster==0)

gene.df <- bitr(aae_baron_mouse.markers_g0$gene, fromType = "SYMBOL",
                toType = c("ENSEMBL", "ENTREZID"),
                OrgDb = org.Mm.eg.db)
ego_g0 <- enrichGO(gene          = unique(gene.df$ENTREZID),
                   OrgDb         = org.Mm.eg.db,
                   ont           = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.01,
                   qvalueCutoff  = 0.05)
tiff(filename = "aae_ego_g0_mast_MF.tiff",res = 300,
     width = 1800, height = 2800)
dotplot(ego_g0, showCategory=20)
dev.off()
aae_baron_mouse.markers_g1<-subset(aae_baron_mouse.markers,aae_baron_mouse.markers$cluster==1)

gene.df <- bitr(aae_baron_mouse.markers_g1$gene, fromType = "SYMBOL",
                toType = c("ENSEMBL", "ENTREZID"),
                OrgDb = org.Mm.eg.db)
ego_g1 <- enrichGO(gene          = unique(gene.df$ENTREZID),
                   OrgDb         = org.Mm.eg.db,
                   ont           = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.03,
                   qvalueCutoff  = 0.05)

tiff(filename = "aae_ego_g1_mast_MF.tiff",res = 300,
     width = 1800, height = 2800)
dotplot(ego_g1, showCategory=20)
dev.off()
aae_baron_mouse.markers_g3<-subset(aae_baron_mouse.markers,aae_baron_mouse.markers$cluster==3)

gene.df <- bitr(aae_baron_mouse.markers_g3$gene, fromType = "SYMBOL",
                toType = c("ENSEMBL", "ENTREZID"),
                OrgDb = org.Mm.eg.db)
ego_g3 <- enrichGO(gene          = unique(gene.df$ENTREZID),
                   OrgDb         = org.Mm.eg.db,
                   ont           = "MF",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.01,
                   qvalueCutoff  = 0.05)

tiff(filename = "aae_ego_g3_mast_MF.tiff",res = 300,
     width = 1800, height = 2800)
dotplot(ego_g3, showCategory=20)
dev.off()
