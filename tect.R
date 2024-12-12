labels1 <- c(
  "T_cell", "T_cell", "B_cell", "Macrophage", 
  "Neutrophil", "T_cell", "B_cell", "Macrophage", 
  "Dendritic_cell", "NK_cell"
)

library(SingleCellExperiment)

## logcounts方法
raw_counts <- matrix(c(
  10, 20, 5,    # 细胞1的基因表达
  2, 50, 100,   # 细胞2的基因表达
  500, 1, 3,    # 细胞3的基因表达
  0, 15, 25,    # 细胞4的基因表达
  30, 40, 60    # 细胞5的基因表达
), nrow = 5, byrow = TRUE)

rownames(raw_counts) <- paste0("Cell", 1:5)
colnames(raw_counts) <- c("GeneA", "GeneB", "GeneC")
logcounts <-log2(raw_counts + 1)
print(logcounts)

library(scater)

# gene distribution
set.seed(100)
coldata.sum <-1:100
rowdata.sum <- sapply(1:100, function(x) sum(runif(100) > 0.5))
hist(rowdata.sum, main="基因在细胞中的表达分布")

counts_matrix <- data.frame(
  cell_1 = rpois(10 , 10) ,
  cell_2 = rpois(10 , 10) ,
  cell_3 = rpois(10 , 30)
)
rownames(counts_matrix) <- paste0("gene_", 1:10)
counts_matrix <- as.matrix(counts_matrix) # must be a matrix object!
sce <- SingleCellExperiment(assays<-list(counts = counts_matrix))
sce
colData(sce)

library(SingleCellExperiment)

# 创建一个示例的SingleCellExperiment对象
counts_matrix <- matrix(rpois(30, lambda = 10), nrow = 10, ncol = 3)
rownames(counts_matrix) <- paste0("gene_", 1:10)
colnames(counts_matrix) <- paste0("cell_", 1:3)
sce <- SingleCellExperiment(assays = list(counts = counts_matrix))

# 添加基因注释信息到rowData
rowData(sce) <- DataFrame(gene_id = rownames(counts_matrix), gene_name = paste0("Gene_", 1:10))

# 查看基因信息
rowData(sce)

library(SingleCellExperiment)

# 创建一个示例SingleCellExperiment对象
sce_coldata <- SingleCellExperiment(assays = list(counts = matrix(rnorm(1000), nrow = 100, ncol = 10)))

# 添加细胞元数据
colData(sce_coldata) <- DataFrame(cell_type = rep(c("Type1", "Type2"), each = 5), sample_id = 1:10)

# 查看colData
print(colData(sce_coldata)$cell_type)



### 降维技术
reducedDims(sce)


### 特别展示的数据类型
library(SingleCellExperiment)
library(SingleCellExperiment)

# 创建一个示例的SingleCellExperiment对象
counts_matrix <- matrix(rpois(30, lambda = 10), nrow = 10, ncol = 3)
rownames(counts_matrix) <- paste0("gene_", 1:10)
colnames(counts_matrix) <- paste0("cell_", 1:3)
sce <- SingleCellExperiment(assays = list(counts = counts_matrix))

# 添加细胞元数据到colData，确保行数等于细胞数（列数）
colData(sce) <- DataFrame(cell_type1 = rep(c("Type1", "Type2", "Type3"), each = 1))

# 查看colData
print(colData(sce))

