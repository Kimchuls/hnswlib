# Dataset configuration for HNSW experiment
workload_type = SIFT1M
merge_method = MULTI_MERGE
dim = 128
max_elements = 10000000
nb = 10000000
M = 32
ef_construction = 64
k = 100
kk = 1000
nq = 10000
iterations = 1
lrange = 5000000
rrange = 10000000
rerun = true
save_index = true


base_filepath = /ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs
query_filepath = /ssd_root/dataset/ann_sift1b/bigann_query.bvecs
groundtruth_filepath = /ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs
# index_path = /ssd_root/jin467/merger/indexes/test/bigann-index_1-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_2-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_3-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_4-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_5-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_6-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_7-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_8-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_9-1M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_10-1M.hnsw
# index_path = /ssd_root/jin467/merger/indexes/bigann-index_1-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_2-10M.hnsw
index_path = /ssd_root/jin467/merger/indexes/bigann-index_1-1M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_2-1M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_3-1M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_4-1M.hnsw
# index_path = /ssd_root/jin467/merger/indexes/bigann-index_500K.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_500K-1M.hnsw

# efs_array 以逗号分隔
efs_array = 100,110,120,130,140,150
