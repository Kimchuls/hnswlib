#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include "../hnswlib/hnswlib.h"
using namespace hnswlib;

void exportHNSWLayerGraph(HierarchicalNSW<float>* index, const std::string& prefix) {
    size_t max_level = index->maxlevel_;
    size_t cur_element_count = index->cur_element_count;
    size_t maxM = index->maxM_;
    size_t maxM0 = index->maxM0_;
    size_t dim = index->data_size_ / sizeof(float);  // 获取维度
    size_t M_layer;

    for (size_t level = 0; level <= max_level; ++level) {
        std::ofstream out(prefix + std::to_string(level) + ".csv");
        out << "internal_id,neighbor_size";

        M_layer = (level == 0) ? maxM0 : maxM;
        for (size_t j = 0; j < M_layer; ++j) {
            out << ",neighbor_id" << j << ",neighbor_dist" << j;
        }
        for (size_t d = 0; d < dim; ++d) {
            out << ",vec" << d;
        }
        out << "\n";

        for (size_t internal_id = 0; internal_id < cur_element_count; ++internal_id) {
            if (index->element_levels_[internal_id] < level) continue;

            linklistsizeint* ll_cur = index->get_linklist_at_level(internal_id, level);
            size_t linklistCount = index->getListCount(ll_cur);
            tableint* data = (tableint*)(ll_cur + sizeof(linklistsizeint));
            float* dist = index->get_dist_at_level(internal_id, level);

            out << internal_id << "," << linklistCount;
            for (size_t i = 0; i < M_layer; ++i) {
                if (i < linklistCount) {
                    out << "," << data[i] << "," << dist[i];
                } else {
                    out << ",-1,-1";
                }
            }

            // 追加向量数据
            char* data_ptrv = index->getDataByInternalId(internal_id);
            float* vec = reinterpret_cast<float*>(data_ptrv);
            for (size_t d = 0; d < dim; ++d) {
                out << "," << vec[d];
            }

            out << "\n";
        }

        out.close();
    }
}


int main() {
    std::string index_path = "/scratch1/jin467/indexes/gist1m.hnsw";
    size_t dim = 960;
    size_t max_elements = 1e6;
    size_t M = 32;
    size_t ef_construction = 200;

    L2Space space(dim);
    HierarchicalNSW<float>* index = new HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    index->loadIndex(index_path, &space);

    // exportHNSWLayerGraph(index, "sift1M_");
    index->indexFileSize();

    delete index;
    return 0;
}
