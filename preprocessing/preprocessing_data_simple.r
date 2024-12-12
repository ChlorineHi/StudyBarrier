# 单细胞RNA测序数据预处理脚本（简化版）

# 读取单细胞RNA测序数据的RDS文件
# 从预处理好的数据文件中加载Darmanis数据集
scRNA <- readRDS("E:/Study/scDCCA-main/example data/darmanis.rds")

# 直接从 scRNA 对象中提取对数转换后的表达矩阵
# 这一步假设数据已经进行了对数转换和标准化
logcounts <- scRNA@assays[[".->data"]]@listData[["logcounts"]]

# 检查 logcounts 的数据类型和维度
# 确保数据已正确加载
class(logcounts)
dim(logcounts)

# 提取细胞类型标签
# 使用细胞类型注释
labels1 <- scRNA@colData@listData[["cell_type1"]]

# 检查标签的数据类型和维度
# 确保标签数据已正确提取
class(labels1)
dim(labels1)

# 将数据转换为数据框格式
# 便于后续数据处理和操作
logcounts.dataframe <- as.data.frame(logcounts)

# 创建 logcounts 数据框的备份
# 保留原始数据，以便后续需要还原
logcounts.dataframe_beifen <- logcounts.dataframe

# 基因过滤：只保留在至少5%的细胞中表达的基因
# 筛选标准：一个基因在总细胞数5%以上的细胞中有表达
# 使用原始备份数据框进行过滤，保留更多表达信息细节
rowdata.sum = rowSums(logcounts.dataframe > 0)  # 计算每个基因的非零表达细胞数
logcounts.dataframe.genefilt <- logcounts.dataframe_beifen[which(rowdata.sum > dim(logcounts.dataframe)[2] * 0.05), ]

# 检查过滤后的数据维度
# 确认基因过滤的效果
dim(logcounts.dataframe.genefilt) 

# 转置数据框，使每行代表一个细胞，每列代表一个基因
# 为后续机器学习模型准备数据
logcounts.dataframe.genefilt.t <- t(logcounts.dataframe.genefilt)

# 将处理后的数据写入文件
# 写入 CSV 格式（带行名和列名）
# 方便在不同软件和平台间交换数据
write.table(logcounts.dataframe.genefilt.t, "muraro_preprocessed_log_genefilt_simple.csv", row.names=TRUE, col.names=TRUE, sep=" ")

# 写入细胞类型标签
# 保存真实标签，用于后续监督学习或评估
write.table(labels1, "muraro_truelabels1_simple.csv", row.names=FALSE, col.names=FALSE, sep=" ")

# 写入 TXT 格式（不带行名和列名）
# 提供另一种格式的数据文件，增加数据可访问性
write.table(logcounts.dataframe.genefilt.t, "muraro_preprocessed_log_genefilt_simple.txt", row.names=FALSE, col.names=FALSE, sep=" ")
write.table(labels1, "muraro_truelabels_simple.txt", row.names=FALSE, col.names=FALSE, sep=" ")
