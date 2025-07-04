# Dataset configuration for HNSW experiment
workload_type = SIFT10M
merge_method = MULTI_TWO_MERGE
multi_test_method = RANDOM
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
groundtruth_filepath = /ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs
# index_path = /ssd_root/jin467/merger/indexes/bigann-index_1-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_2-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_3-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_4-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_5-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_6-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_7-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_8-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_9-10M.hnsw,/ssd_root/jin467/merger/indexes/bigann-index_10-10M.hnsw
# index_path = /ssd_root/jin467/merger/indexes/test/bigann-index_2500K.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_2500K-5000K.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_5000K-7500K.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_7500K-10M.hnsw
index_path = /ssd_root/jin467/merger/indexes/test/bigann-index_0-1_10M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_1-2_10M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_2-3_10M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_3-5_10M.hnsw,/ssd_root/jin467/merger/indexes/test/bigann-index_5-10_10M.hnsw

# efs_array 以逗号分隔
efs_array = 100,120,140,160,180,200,250,300,400,500,600,700,800
# efs_array = 100,120,140,160
