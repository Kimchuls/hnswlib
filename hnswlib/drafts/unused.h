#include "hnswalg.h"

namespace hnswlib {

template <typename dist_t>
void HierarchicalNSW<dist_t>::mergeIndex1BasedOnIndex2Connection(HierarchicalNSW<dist_t> *index1,
                                                                 HierarchicalNSW<dist_t> *index2,
                                                                 tableint cur_c,
                                                                 char *data_point,
                                                                 int offset_index1,
                                                                 int offset_index2,
                                                                 int level,
                                                                 tableint &last_entry_point,
                                                                 int lambda,
                                                                 float alpha,
                                                                 std::unordered_map<tableint, std::vector<std::pair<dist_t, tableint>>> *candidateSetIndex2) {
    int higherLevel = level;
    if (last_entry_point == -1) {
        higherLevel = index2->maxlevel_;
    }
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;

    linklistsizeint *ll_cur;
    size_t linklistCount;
    tableint *data;
    dist_t *dist;
    tableint candidate_id;
    dist_t dist1;
    dist_t lowerBound;
    size_t Mcurmax = level ? maxM_ : maxM0_;

    int cnt = lambda;
    index2->search2Layer(data_point,
                         last_entry_point,
                         higherLevel,
                         level,
                         cnt,
                         &top_candidates,
                         offset_index2);

    auto temp_queue = top_candidates;
    if (candidateSetIndex2) {
        while (!temp_queue.empty()) {
            std::pair<dist_t, tableint> current = temp_queue.top();
            temp_queue.pop();
            (*candidateSetIndex2)[current.second - offset_index2].push_back(std::make_pair(current.first, cur_c + offset_index1));
        }
    }

    if (top_candidates.size() + index1->getListCount(index1->get_linklist_at_level(cur_c, level)) > Mcurmax) {
        ll_cur = index1->get_linklist_at_level(cur_c, level);
        linklistCount = index1->getListCount(ll_cur);
        data = (tableint *)(ll_cur + 1);
        dist = index1->get_dist_at_level(cur_c, level);
        for (size_t iter = 0; iter < linklistCount; iter++) {
            candidate_id = data[iter] + offset_index1;
            dist1 = dist[iter];
            top_candidates.emplace(dist1, candidate_id);
        }
        getNeighborsByHeuristic2(top_candidates, Mcurmax, false, -1, alpha);
        ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
        setListCount(ll_cur, top_candidates.size());
        data = (tableint *)(ll_cur + 1);
        dist_t *distData = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
        for (size_t idx = 0; top_candidates.size() > 0; idx++) {
            data[idx] = top_candidates.top().second;
            distData[idx] = top_candidates.top().first;
            top_candidates.pop();
        }
    } else {
        linklistsizeint *ll_cur_index1 = index1->get_linklist_at_level(cur_c, level);
        size_t linklistCount_index1 = index1->getListCount(ll_cur_index1);
        tableint *data_index1 = (tableint *)(ll_cur_index1 + 1);
        dist_t *dist_index1 = index1->get_dist_at_level(cur_c, level);

        ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
        setListCount(ll_cur, top_candidates.size() + linklistCount_index1);
        data = (tableint *)(ll_cur + 1);
        dist_t *distData = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
        size_t offset = top_candidates.size();
        for (size_t idx = 0; top_candidates.size() > 0; idx++) {
            data[idx] = top_candidates.top().second;
            distData[idx] = top_candidates.top().first;
            top_candidates.pop();
        }
        for (size_t idx = 0; idx < linklistCount_index1; idx++) {
            data[offset + idx] = data_index1[idx] + offset_index1;
            distData[offset + idx] = dist_index1[idx];
        }
    }
}
    
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

template <typename dist_t>
HierarchicalNSW<dist_t> *MultiIndexMerger(std::vector<HierarchicalNSW<dist_t> *> indices, L2Space *space, size_t M = -1, size_t ef_construction = -1) {
    double t00 = elapsed();
    if (indices.empty()) {
        throw std::runtime_error("No indices provided for merging");
    }

    if (indices.size() == 1) {
        return indices[0];
    }

    double t0 = elapsed();
    std::sort(indices.begin(), indices.end(),
              [](auto a, auto b) { return a->cur_element_count != b->cur_element_count ? a->cur_element_count < b->cur_element_count : a->maxlevel_ < b->maxlevel_; });

    size_t total_elements = 0;
    size_t maxLevel = 0;
    std::vector<size_t> index_offsets = {0};

    for (auto index : indices) {
        total_elements += index->getCurrentElementCount();
        maxLevel = std::max(maxLevel, (size_t)index->maxlevel_);
        index_offsets.push_back(total_elements);
    }

    M = (M == -1) ? (*std::max_element(indices.begin(), indices.end(), [](auto a, auto b) { return a->M_ < b->M_; }))->M_ : M;

    ef_construction = (ef_construction == -1) ? (*std::max_element(indices.begin(), indices.end(), [](auto a, auto b) {
                                                    return a->ef_construction_ < b->ef_construction_;
                                                }))->ef_construction_ :
                                                ef_construction;

    size_t new_max_elements = std::accumulate(indices.begin(), indices.end(), 0, [](size_t sum, auto index) {
        return sum + index->max_elements_;
    });

    HierarchicalNSW<dist_t> *merged_index = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    merged_index->setMaxLevel(maxLevel);
    merged_index->cur_element_count.store(total_elements);

    auto allocateMemory = [&](char *&memory, const char *errorMsg, size_t size) {
        memory = (char *)malloc(size);
        if (memory == nullptr)
            throw std::runtime_error(errorMsg);
    };

    auto allocateMemory_layer = [&](hnswlib::HierarchicalNSW<dist_t> *index, int element_count_offset) {
#pragma omp parallel for schedule(dynamic)
        for (int id = 0; id < index->cur_element_count; id++) {
            if (index->element_levels_[id] < 1)
                continue;
            int level = index->element_levels_[id];
            merged_index->element_levels_[id + element_count_offset] = level;
            tableint new_c = id + element_count_offset;
            allocateMemory(merged_index->linkLists_[new_c],
                           "Not enough memory: addPoint failed to allocate linklist",
                           merged_index->size_links_per_element_ * level + 1);
            allocateMemory(merged_index->dist_linkLists_[new_c],
                           "Not enough memory: addPoint failed to allocate dist_linklist",
                           merged_index->size_dist_links_per_element_ * level + 1);
        }
    };

    allocateMemory(merged_index->data_level0_memory_,
                   "Not enough memory: MultiIndexMerger failed to allocate level0 data",
                   merged_index->max_elements_ * merged_index->size_data_per_element_);

    allocateMemory(merged_index->dist_level0_memory_,
                   "Not enough memory: MultiIndexMerger failed to allocate level0 distance",
                   merged_index->max_elements_ * merged_index->size_dist_per_element_);

    std::vector<std::vector<std::vector<int>>> layer_nodes_for_indices(indices.size());
    std::vector<std::vector<std::vector<tableint>>> entry_points_collect(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        layer_nodes_for_indices[i].resize(maxLevel + 2);
        indices[i]->searchNodeOnEachLayer(layer_nodes_for_indices[i]);

        entry_points_collect[i].resize(indices.size());
        size_t element_count = indices[i]->getCurrentElementCount();
        for (size_t j = 0; j < indices.size(); j++) {
            if (i != j) {
                entry_points_collect[i][j].resize(element_count, -1);
            }
        }
        allocateMemory_layer(indices[i], index_offsets[i]);

        // memcpy(merged_index->element_levels_.data() + index_offsets[i],
        //        indices[i]->element_levels_.data(),
        //        sizeof(int) * indices[i]->getCurrentElementCount());
        memcpy(merged_index->data_level0_memory_ + index_offsets[i] * (merged_index->size_data_per_element_),
               indices[i]->data_level0_memory_,
               indices[i]->getCurrentElementCount() * (merged_index->size_data_per_element_));

        memcpy(merged_index->dist_level0_memory_ + index_offsets[i] * (merged_index->size_dist_per_element_),
               indices[i]->dist_level0_memory_,
               indices[i]->getCurrentElementCount() * (merged_index->size_dist_per_element_));
    }

    std::vector<std::vector<int>> mergedDataPointsFromIndices(indices.size());

    std::vector<int> newLayer;
    size_t total_top_layer_size =
        std::accumulate(layer_nodes_for_indices.begin(),
                        layer_nodes_for_indices.end(),
                        0,
                        [maxLevel](size_t sum, auto &nodes) {
                            return sum + nodes[maxLevel].size();
                        });

    if (total_top_layer_size > M) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        int cnt = 0;

        auto processTopLayerNode = [&](size_t idx, auto &layer_nodes, size_t offset) {
            auto it = layer_nodes.begin();
            while (it != layer_nodes.end()) {
                if (distribution(merged_index->level_generator_) < 1.0 / M || cnt == M) {
                    newLayer.push_back((*it) + offset);
                    cnt = 0;
                    tableint new_c = *it;

                    allocateMemory(merged_index->linkLists_[new_c + offset], "Not enough memory: addPoint failed to allocate linklist",
                                   merged_index->size_links_per_element_ * (maxLevel + 1) + 1);

                    allocateMemory(merged_index->dist_linkLists_[new_c + offset], "Not enough memory: addPoint failed to allocate dist_linkLists",
                                   merged_index->size_dist_links_per_element_ * (maxLevel + 1) + 1);

                    merged_index->element_levels_[new_c + offset] = (maxLevel + 1);

                    memcpy(merged_index->getDataByInternalId(new_c + offset), indices[idx]->getDataByInternalId(new_c), merged_index->data_size_);
                    merged_index->setExternalLabel(new_c + offset, indices[idx]->getExternalLabel(new_c));
                    mergedDataPointsFromIndices[idx].push_back(new_c);

                    it = layer_nodes.erase(it);
                } else {
                    cnt++;
                    ++it;
                }
            }
        };

        for (size_t idx = 0; idx < indices.size(); idx++) {
            processTopLayerNode(idx, layer_nodes_for_indices[idx][maxLevel], index_offsets[idx]);
        }

        if (!newLayer.empty()) {
            for (int newLayerIter = 0; newLayerIter < newLayer.size(); newLayerIter++) {
                linklistsizeint *ll_cur = merged_index->get_linklist_at_level(newLayer[newLayerIter], maxLevel + 1);
                merged_index->setListCount(ll_cur, newLayer.size() - 1);
                tableint *data = (tableint *)(ll_cur + 1);
                dist_t *dist = merged_index->get_dist_at_level(newLayer[newLayerIter], maxLevel + 1);

                for (size_t it = 0, idx = 0; it < newLayer.size(); it++) {
                    if (it == newLayerIter)
                        continue;
                    dist_t dist0 = merged_index->fstdistfunc_(merged_index->getDataByInternalId(newLayer[newLayerIter]), merged_index->getDataByInternalId(newLayer[it]),
                                                              merged_index->dist_func_param_);
                    data[idx] = newLayer[it];
                    dist[idx] = dist0;
                    idx++;
                }
            }

            merged_index->setMaxLevel(maxLevel + 1);
            merged_index->enterpoint_node_ = newLayer[0];
            maxLevel += 1;
        } else {
            merged_index->enterpoint_node_ = layer_nodes_for_indices.back()[maxLevel][0] + index_offsets[indices.size() - 1];
        }
    } else {
        merged_index->enterpoint_node_ = layer_nodes_for_indices.back()[maxLevel][0] + index_offsets[indices.size() - 1];
    }

    // printf("maxlevel: %zu\n", maxLevel);

    printf("time stage 1: %f\n", elapsed() - t0);
    t0 = elapsed();

    int numThreads = omp_get_max_threads();
    struct alignas(64) LocalMap : std::unordered_map<std::pair<size_t, tableint>,
                                                     std::vector<std::pair<dist_t, tableint>>,
                                                     pair_hash> {};
    std::vector<LocalMap> localCandidateSets(numThreads);

    for (int level = maxLevel; level > 0; --level) {
        // 统计层中活跃索引
        int index_with_nodes = -1;
        int count_indices_with_nodes = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!layer_nodes_for_indices[idx][level].empty()) {
                index_with_nodes = static_cast<int>(idx);
                ++count_indices_with_nodes;
            }
        }
        if (count_indices_with_nodes == 1) {
            merged_index->deepCopyOneLayerOnIndex(
                indices[index_with_nodes],
                level, index_offsets[index_with_nodes],
                layer_nodes_for_indices[index_with_nodes]);
            continue;
        }
        if (count_indices_with_nodes == 0) continue;

        // 收集并预分配 mergedDataPoints
        size_t totalPoints = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &vec = mergedDataPointsFromIndices[idx];
            vec.reserve(vec.size() + layer_nodes_for_indices[idx][level].size());
            for (auto old_c : layer_nodes_for_indices[idx][level]) {
                vec.push_back(old_c);
            }
            totalPoints += vec.size();
        }

        // 清空且保留本地 map 空间
        for (int t = 0; t < numThreads; ++t) {
            localCandidateSets[t].clear();
        }
        std::unordered_map<
            std::pair<size_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            pair_hash>
            candidateSetsForNode;
        candidateSetsForNode.reserve(totalPoints);

        // 并行前向搜索：内层循环并行，chunk=1024
        for (size_t idx = 0; idx + 1 < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
#pragma omp parallel for
            for (size_t xi = 0; xi < points.size(); ++xi) {
                int tid = omp_get_thread_num();
                auto &localMap = localCandidateSets[tid];
                tableint cur_c = points[xi];
                char *data_point = indices[idx]->getDataByInternalId(cur_c);
                for (size_t target_idx = idx + 1; target_idx < indices.size(); ++target_idx) {
                    auto &last_entry_point = entry_points_collect[idx][target_idx][cur_c];
                    int higherLevel = (last_entry_point == -1) ? indices[target_idx]->maxlevel_ : level;
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        typename HierarchicalNSW<dist_t>::CompareByFirst>
                        top_candidates;

                    indices[target_idx]->search2Layer(
                        data_point,
                        last_entry_point,
                        higherLevel,
                        level,
                        10,
                        &top_candidates,
                        index_offsets[target_idx]);

                    auto temp_q = top_candidates;
                    while (!temp_q.empty()) {
                        auto current = temp_q.top();
                        temp_q.pop();
                        tableint raw_target = current.second;
                        tableint target_node = raw_target - index_offsets[target_idx];
                        tableint source_node = cur_c + index_offsets[idx];
                        localMap[{idx, cur_c}].emplace_back(current);
                        localMap[{target_idx, target_node}]
                            .emplace_back(current.first, source_node);
                    }
                }
            }
        }

        // 并行合并本地到全局 map
        // #pragma omp parallel for schedule(static)
        for (int t = 0; t < numThreads; ++t) {
            for (auto &kv : localCandidateSets[t]) {
                auto &globalVec = candidateSetsForNode[kv.first];
                globalVec.insert(
                    globalVec.end(),
                    kv.second.begin(),
                    kv.second.end());
            }
        }

        // 第二阶段：链接合并，内层大循环并行，chunk=512
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
            size_t offset = index_offsets[idx];
#pragma omp parallel for
            for (size_t pi = 0; pi < points.size(); ++pi) {
                tableint cur_c = points[pi];
                std::priority_queue<
                    std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst>
                    top_candidates;

                auto ll_cur = indices[idx]->get_linklist_at_level(cur_c, level);
                size_t linklistCount = indices[idx]->getListCount(ll_cur);
                auto data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto dist = indices[idx]->get_dist_at_level(cur_c, level);

                for (size_t i = 0; i < linklistCount; ++i) {
                    top_candidates.emplace(dist[i], data[i] + offset);
                }

                auto key = std::make_pair(idx, cur_c);
                if (auto it = candidateSetsForNode.find(key); it != candidateSetsForNode.end()) {
                    for (auto &p : it->second) {
                        top_candidates.emplace(p.first, p.second);
                    }
                }

                size_t Mcurmax = level ? merged_index->maxM_ : merged_index->maxM0_;
                merged_index->getNeighborsByHeuristic2(
                    top_candidates, Mcurmax, true, -1, 1.1);

                ll_cur = merged_index->get_linklist_at_level(cur_c + offset, level);
                merged_index->setListCount(ll_cur, top_candidates.size());
                auto new_data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto new_dist = reinterpret_cast<dist_t *>(
                    merged_index->get_dist_at_level(cur_c + offset, level));
                for (size_t i = 0; !top_candidates.empty(); ++i) {
                    new_data[i] = top_candidates.top().second;
                    new_dist[i] = top_candidates.top().first;
                    top_candidates.pop();
                }
            }
        }
    }

    for (int level = 0; level >= 0; level--) {
        int index_with_nodes = -1;
        int count_indices_with_nodes = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!layer_nodes_for_indices[idx][level].empty()) {
                index_with_nodes = static_cast<int>(idx);
                ++count_indices_with_nodes;
            }
        }
        if (count_indices_with_nodes == 1) {
            merged_index->deepCopyOneLayerOnIndex(
                indices[index_with_nodes],
                level, index_offsets[index_with_nodes],
                layer_nodes_for_indices[index_with_nodes]);
            continue;
        }
        if (count_indices_with_nodes == 0) continue;

        // 收集并预分配 mergedDataPoints
        size_t totalPoints = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &vec = mergedDataPointsFromIndices[idx];
            vec.reserve(vec.size() + layer_nodes_for_indices[idx][level].size());
            for (auto old_c : layer_nodes_for_indices[idx][level]) {
                vec.push_back(old_c);
            }
            totalPoints += vec.size();
        }

        // 清空且保留本地 map 空间
        for (int t = 0; t < numThreads; ++t) {
            localCandidateSets[t].clear();
        }
        std::unordered_map<
            std::pair<size_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            pair_hash>
            candidateSetsForNode;
        candidateSetsForNode.reserve(totalPoints);

        // 并行前向搜索：内层循环并行，chunk=1024
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
#pragma omp parallel for
            for (size_t xi = 0; xi < points.size(); ++xi) {
                int tid = omp_get_thread_num();
                auto &localMap = localCandidateSets[tid];
                tableint cur_c = points[xi];
                char *data_point = indices[idx]->getDataByInternalId(cur_c);
                size_t linklistCount = indices[idx]->getListCount(indices[idx]->get_linklist_at_level(cur_c, 0));
                int lowerLevel = (indices[idx]->element_levels_[cur_c] > 0 || linklistCount < int(0.75 * M)) ? 0 : 1;
                // int lowerLevel = (indices[idx]->element_levels_[cur_c] > 0) ? 0 : 1;

                for (size_t target_idx = idx + 1; target_idx < indices.size(); ++target_idx) {
                    auto &last_entry_point = entry_points_collect[idx][target_idx][cur_c];
                    int higherLevel = (last_entry_point == -1) ? indices[target_idx]->maxlevel_ : std::max(0, std::min(indices[target_idx]->maxlevel_, lowerLevel));
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        typename HierarchicalNSW<dist_t>::CompareByFirst>
                        top_candidates;

                    indices[target_idx]->search2Layer(
                        data_point,
                        last_entry_point,
                        higherLevel,
                        lowerLevel,
                        3,
                        &top_candidates,
                        index_offsets[target_idx]);

                    auto temp_q = top_candidates;
                    while (!temp_q.empty()) {
                        auto current = temp_q.top();
                        temp_q.pop();
                        tableint raw_target = current.second;
                        tableint target_node = raw_target - index_offsets[target_idx];
                        tableint source_node = cur_c + index_offsets[idx];
                        localMap[{idx, cur_c}].emplace_back(current);
                        localMap[{target_idx, target_node}]
                            .emplace_back(current.first, source_node);
                    }
                }
            }
        }

        // 并行合并本地到全局 map
        // #pragma omp parallel for schedule(static)
        for (int t = 0; t < numThreads; ++t) {
            for (auto &kv : localCandidateSets[t]) {
                auto &globalVec = candidateSetsForNode[kv.first];
                globalVec.insert(
                    globalVec.end(),
                    kv.second.begin(),
                    kv.second.end());
            }
        }

        // 第二阶段：链接合并，内层大循环并行，chunk=512
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
            size_t offset = index_offsets[idx];
#pragma omp parallel for
            for (size_t pi = 0; pi < points.size(); ++pi) {
                tableint cur_c = points[pi];
                std::priority_queue<
                    std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst>
                    top_candidates;

                auto ll_cur = indices[idx]->get_linklist_at_level(cur_c, level);
                size_t linklistCount = indices[idx]->getListCount(ll_cur);
                auto data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto dist = indices[idx]->get_dist_at_level(cur_c, level);

                for (size_t i = 0; i < linklistCount; ++i) {
                    top_candidates.emplace(dist[i], data[i] + offset);
                }

                auto key = std::make_pair(idx, cur_c);
                if (auto it = candidateSetsForNode.find(key); it != candidateSetsForNode.end()) {
                    for (auto &p : it->second) {
                        top_candidates.emplace(p.first, p.second);
                    }
                }

                size_t Mcurmax = level ? merged_index->maxM_ : merged_index->maxM0_;
                merged_index->getNeighborsByHeuristic2(
                    top_candidates, Mcurmax, true, -1, 1.1);

                ll_cur = merged_index->get_linklist_at_level(cur_c + offset, level);
                merged_index->setListCount(ll_cur, top_candidates.size());
                auto new_data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto new_dist = reinterpret_cast<dist_t *>(
                    merged_index->get_dist_at_level(cur_c + offset, level));
                for (size_t i = 0; !top_candidates.empty(); ++i) {
                    new_data[i] = top_candidates.top().second;
                    new_dist[i] = top_candidates.top().first;
                    top_candidates.pop();
                }
            }
        }
    }

    printf("time stage 2: %f\n", elapsed() - t0);
    printf("total time: %f\n", elapsed() - t00);
    return merged_index;
}

template <typename dist_t>
HierarchicalNSW<dist_t> *HNSWRefinement(HierarchicalNSW<dist_t> *index1, L2Space *space, int tuning_param = 0) {
    // 1) 构建新索引，并复制基本参数
    size_t max_elements = index1->max_elements_;
    size_t M = index1->M_;
    size_t ef_construction = index1->ef_construction_;

    double t0 = elapsed();
    auto index2 = new HierarchicalNSW<dist_t>(space, max_elements, M, ef_construction);
    index2->setMaxLevel(index1->maxlevel_);
    index2->cur_element_count.store(index1->cur_element_count);
    index2->enterpoint_node_ = index1->enterpoint_node_;

    // 2) 复制数据存储 (level0)
    index2->data_level0_memory_ = (char *)malloc(max_elements * index2->size_data_per_element_);
    if (!index2->data_level0_memory_) {
        throw std::runtime_error("Refinement failed: cannot allocate data_level0_memory");
    }
    memcpy(index2->data_level0_memory_, index1->data_level0_memory_, index1->cur_element_count * index1->size_data_per_element_);

    index2->dist_level0_memory_ = (char *)malloc(max_elements * index2->size_dist_per_element_);
    if (!index2->dist_level0_memory_) {
        throw std::runtime_error("Refinement failed: cannot allocate dist_level0_memory");
    }
    memcpy(index2->dist_level0_memory_, index1->dist_level0_memory_, index1->cur_element_count * index1->size_dist_per_element_);

    index2->element_levels_ = index1->element_levels_;
    index2->label_lookup_ = index1->label_lookup_;

    for (int id = 0; id < index1->cur_element_count; id++) {
        if (index1->element_levels_[id] < 1)
            continue;
        int level = index1->element_levels_[id];
        index2->linkLists_[id] = (char *)malloc(index2->size_links_per_element_ * level + 1);
        if (index2->linkLists_[id] == nullptr) {
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        }
        memcpy(index2->linkLists_[id], index1->linkLists_[id], index2->size_links_per_element_ * level + 1);
        index2->dist_linkLists_[id] = (char *)malloc(index2->size_dist_links_per_element_ * level + 1);
        if (index2->dist_linkLists_[id] == nullptr) {
            throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linklist");
        }
        memcpy(index2->dist_linkLists_[id], index1->dist_linkLists_[id], index2->size_dist_links_per_element_ * level + 1);
    }
    printf("[%.3f s] step 1\n", elapsed() - t0);

    //     size_t matchCount = 0;
    t0 = elapsed();
    vector<vector<tableint>> nodeNeighCount(index1->M_ * 2 + 1);
    for (tableint id = 0; id < index1->cur_element_count; ++id) {
        size_t linklistCount = index1->getListCount(index1->get_linklist_at_level(id, 0));
        nodeNeighCount[linklistCount].push_back(id);
    }
    const int length = index1->cur_element_count.load();
    std::vector<short> nodeMark(length, 0);
    int total_count = 0;
    // int limit_count = 1e6;
    // for (int neighborCnt = index1->M_ * 2; neighborCnt > 0; neighborCnt--) {
    for (int neighborCnt = 1; neighborCnt <= std::min((size_t)40, index1->M_ * 2); neighborCnt++) {
        for (int iter = 0; iter < nodeNeighCount[neighborCnt].size(); iter++) {
            tableint id = nodeNeighCount[neighborCnt][iter];
            if (nodeMark[id] == -1)
                continue;
            linklistsizeint *ll_cur = index1->get_linklist_at_level(id, 0);
            size_t linklistCount = index1->getListCount(ll_cur);
            int k = (int)(linklistCount / 5);
            if (nodeMark[id] < k) {
                nodeMark[id] = -1;
                tableint *data = (tableint *)(ll_cur + 1);
                for (int dataCnt = 0; dataCnt < linklistCount; dataCnt++) {
                    if (nodeMark[data[dataCnt]] == -1) continue;
                    nodeMark[data[dataCnt]]++;
                }
                total_count++;
            }
        }
    }
    printf("total_count: %d\n", total_count);

    std::atomic<size_t> matchCount{0};
    float totalSearchTime = 0.0f;
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : totalSearchTime)
    for (tableint id = 0; id < index1->cur_element_count; ++id) {
        int level = index1->element_levels_[id];
        if (nodeMark[id] != -1) {
            continue;
        }

        int cnt = 128;

        linklistsizeint *ll_cur_index1 = index1->get_linklist_at_level(id, 0);
        size_t linklistCount = index1->getListCount(ll_cur_index1);

        size_t prev = matchCount.fetch_add(1, std::memory_order_relaxed);

        int curlevel = 0;
        size_t Mcurmax = curlevel ? index2->maxM_ : index2->maxM0_;
        std::priority_queue<
            std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            typename HierarchicalNSW<dist_t>::CompareByFirst>
            top_candidates;

        index1->search2Layer(
            index1->getDataByInternalId(id),
            id,
            curlevel,
            curlevel,
            cnt,
            &top_candidates,
            0);

        // float t0 = omp_get_wtime();
        index2->getNeighborsByHeuristic2(
            top_candidates,
            Mcurmax,
            false,
            id,
            1.1);
        // double t1 = omp_get_wtime();
        // totalSearchTime += (t1 - t0);

        linklistsizeint *ll_cur_index2 = index2->get_linklist_at_level(id, 0);
        index2->setListCount(ll_cur_index2, top_candidates.size());
        tableint *nbrs = (tableint *)(ll_cur_index2 + 1);
        dist_t *dists = (dist_t *)(index2->get_dist_at_level(id, curlevel));

        size_t idx = 0;
        while (!top_candidates.empty()) {
            auto pr = top_candidates.top();
            top_candidates.pop();
            nbrs[idx] = pr.second;
            dists[idx] = pr.first;
            ++idx;
        }
    }
    // printf("Total search2Layer time: %f seconds\n", totalSearchTime);
    // 循环结束后可以读取 matchCount
    printf("实际执行到阈值前的 matchCount = %zu\n", matchCount.load());
    printf("[%.3f s] step 2\n", elapsed() - t0);
    // printf("matchCount: %zu\n", matchCount);

    return index2;
}

    template <typename dist_t>
HierarchicalNSW<dist_t> *MultiIndexMerger(std::vector<HierarchicalNSW<dist_t> *> indices, L2Space *space, size_t M = -1, size_t ef_construction = -1) {
    double t00 = elapsed();
    if (indices.empty()) {
        throw std::runtime_error("No indices provided for merging");
    }

    if (indices.size() == 1) {
        return indices[0];
    }

    double t0 = elapsed();
    std::sort(indices.begin(), indices.end(),
              [](auto a, auto b) { return a->cur_element_count != b->cur_element_count ? a->cur_element_count < b->cur_element_count : a->maxlevel_ < b->maxlevel_; });

    size_t total_elements = 0;
    size_t maxLevel = 0;
    std::vector<size_t> index_offsets = {0};

    for (auto index : indices) {
        total_elements += index->getCurrentElementCount();
        maxLevel = std::max(maxLevel, (size_t)index->maxlevel_);
        index_offsets.push_back(total_elements);
    }

    M = (M == -1) ? (*std::max_element(indices.begin(), indices.end(), [](auto a, auto b) { return a->M_ < b->M_; }))->M_ : M;

    ef_construction = (ef_construction == -1) ? (*std::max_element(indices.begin(), indices.end(), [](auto a, auto b) {
                                                    return a->ef_construction_ < b->ef_construction_;
                                                }))->ef_construction_ :
                                                ef_construction;

    size_t new_max_elements = std::accumulate(indices.begin(), indices.end(), 0, [](size_t sum, auto index) {
        return sum + index->max_elements_;
    });

    HierarchicalNSW<dist_t> *merged_index = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    merged_index->setMaxLevel(maxLevel);
    merged_index->cur_element_count.store(total_elements);

    auto allocateMemory = [&](char *&memory, const char *errorMsg, size_t size) {
        memory = (char *)malloc(size);
        if (memory == nullptr)
            throw std::runtime_error(errorMsg);
    };

    auto allocateMemory_layer = [&](hnswlib::HierarchicalNSW<dist_t> *index, int element_count_offset) {
#pragma omp parallel for schedule(dynamic)
        for (int id = 0; id < index->cur_element_count; id++) {
            if (index->element_levels_[id] < 1)
                continue;
            int level = index->element_levels_[id];
            merged_index->element_levels_[id + element_count_offset] = level;
            tableint new_c = id + element_count_offset;
            allocateMemory(merged_index->linkLists_[new_c],
                           "Not enough memory: addPoint failed to allocate linklist",
                           merged_index->size_links_per_element_ * level + 1);
            allocateMemory(merged_index->dist_linkLists_[new_c],
                           "Not enough memory: addPoint failed to allocate dist_linklist",
                           merged_index->size_dist_links_per_element_ * level + 1);
        }
    };

    allocateMemory(merged_index->data_level0_memory_,
                   "Not enough memory: MultiIndexMerger failed to allocate level0 data",
                   merged_index->max_elements_ * merged_index->size_data_per_element_);

    allocateMemory(merged_index->dist_level0_memory_,
                   "Not enough memory: MultiIndexMerger failed to allocate level0 distance",
                   merged_index->max_elements_ * merged_index->size_dist_per_element_);

    std::vector<std::vector<std::vector<int>>> layer_nodes_for_indices(indices.size());
    std::vector<std::vector<std::vector<tableint>>> entry_points_collect(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        layer_nodes_for_indices[i].resize(maxLevel + 2);
        indices[i]->searchNodeOnEachLayer(layer_nodes_for_indices[i]);

        entry_points_collect[i].resize(indices.size());
        size_t element_count = indices[i]->getCurrentElementCount();
        for (size_t j = 0; j < indices.size(); j++) {
            if (i != j) {
                entry_points_collect[i][j].resize(element_count, -1);
            }
        }
        allocateMemory_layer(indices[i], index_offsets[i]);

        // memcpy(merged_index->element_levels_.data() + index_offsets[i],
        //        indices[i]->element_levels_.data(),
        //        sizeof(int) * indices[i]->getCurrentElementCount());
        memcpy(merged_index->data_level0_memory_ + index_offsets[i] * (merged_index->size_data_per_element_),
               indices[i]->data_level0_memory_,
               indices[i]->getCurrentElementCount() * (merged_index->size_data_per_element_));

        memcpy(merged_index->dist_level0_memory_ + index_offsets[i] * (merged_index->size_dist_per_element_),
               indices[i]->dist_level0_memory_,
               indices[i]->getCurrentElementCount() * (merged_index->size_dist_per_element_));
    }

    std::vector<std::vector<int>> mergedDataPointsFromIndices(indices.size());

    std::vector<int> newLayer;
    size_t total_top_layer_size =
        std::accumulate(layer_nodes_for_indices.begin(),
                        layer_nodes_for_indices.end(),
                        0,
                        [maxLevel](size_t sum, auto &nodes) {
                            return sum + nodes[maxLevel].size();
                        });

    if (total_top_layer_size > M) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        int cnt = 0;

        auto processTopLayerNode = [&](size_t idx, auto &layer_nodes, size_t offset) {
            auto it = layer_nodes.begin();
            while (it != layer_nodes.end()) {
                if (distribution(merged_index->level_generator_) < 1.0 / M || cnt == M) {
                    newLayer.push_back((*it) + offset);
                    cnt = 0;
                    tableint new_c = *it;

                    allocateMemory(merged_index->linkLists_[new_c + offset], "Not enough memory: addPoint failed to allocate linklist",
                                   merged_index->size_links_per_element_ * (maxLevel + 1) + 1);

                    allocateMemory(merged_index->dist_linkLists_[new_c + offset], "Not enough memory: addPoint failed to allocate dist_linkLists",
                                   merged_index->size_dist_links_per_element_ * (maxLevel + 1) + 1);

                    merged_index->element_levels_[new_c + offset] = (maxLevel + 1);

                    memcpy(merged_index->getDataByInternalId(new_c + offset), indices[idx]->getDataByInternalId(new_c), merged_index->data_size_);
                    merged_index->setExternalLabel(new_c + offset, indices[idx]->getExternalLabel(new_c));
                    mergedDataPointsFromIndices[idx].push_back(new_c);

                    it = layer_nodes.erase(it);
                } else {
                    cnt++;
                    ++it;
                }
            }
        };

        for (size_t idx = 0; idx < indices.size(); idx++) {
            processTopLayerNode(idx, layer_nodes_for_indices[idx][maxLevel], index_offsets[idx]);
        }

        if (!newLayer.empty()) {
            for (int newLayerIter = 0; newLayerIter < newLayer.size(); newLayerIter++) {
                linklistsizeint *ll_cur = merged_index->get_linklist_at_level(newLayer[newLayerIter], maxLevel + 1);
                merged_index->setListCount(ll_cur, newLayer.size() - 1);
                tableint *data = (tableint *)(ll_cur + 1);
                dist_t *dist = merged_index->get_dist_at_level(newLayer[newLayerIter], maxLevel + 1);

                for (size_t it = 0, idx = 0; it < newLayer.size(); it++) {
                    if (it == newLayerIter)
                        continue;
                    dist_t dist0 = merged_index->fstdistfunc_(merged_index->getDataByInternalId(newLayer[newLayerIter]), merged_index->getDataByInternalId(newLayer[it]),
                                                              merged_index->dist_func_param_);
                    data[idx] = newLayer[it];
                    dist[idx] = dist0;
                    idx++;
                }
            }

            merged_index->setMaxLevel(maxLevel + 1);
            merged_index->enterpoint_node_ = newLayer[0];
            maxLevel += 1;
        } else {
            merged_index->enterpoint_node_ = layer_nodes_for_indices.back()[maxLevel][0] + index_offsets[indices.size() - 1];
        }
    } else {
        merged_index->enterpoint_node_ = layer_nodes_for_indices.back()[maxLevel][0] + index_offsets[indices.size() - 1];
    }

    // printf("maxlevel: %zu\n", maxLevel);

    printf("time stage 1: %f\n", elapsed() - t0);
    t0 = elapsed();

    int numThreads = omp_get_max_threads();
    struct alignas(64) LocalMap : std::unordered_map<std::pair<size_t, tableint>,
                                                     std::vector<std::pair<dist_t, tableint>>,
                                                     pair_hash> {};
    std::vector<LocalMap> localCandidateSets(numThreads);

    for (int level = maxLevel; level > 0; --level) {
        // 统计层中活跃索引
        int index_with_nodes = -1;
        int count_indices_with_nodes = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!layer_nodes_for_indices[idx][level].empty()) {
                index_with_nodes = static_cast<int>(idx);
                ++count_indices_with_nodes;
            }
        }
        if (count_indices_with_nodes == 1) {
            merged_index->deepCopyOneLayerOnIndex(
                indices[index_with_nodes],
                level, index_offsets[index_with_nodes],
                layer_nodes_for_indices[index_with_nodes]);
            continue;
        }
        if (count_indices_with_nodes == 0) continue;

        // 收集并预分配 mergedDataPoints
        size_t totalPoints = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &vec = mergedDataPointsFromIndices[idx];
            vec.reserve(vec.size() + layer_nodes_for_indices[idx][level].size());
            for (auto old_c : layer_nodes_for_indices[idx][level]) {
                vec.push_back(old_c);
            }
            totalPoints += vec.size();
        }

        // 清空且保留本地 map 空间
        for (int t = 0; t < numThreads; ++t) {
            localCandidateSets[t].clear();
        }
        std::unordered_map<
            std::pair<size_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            pair_hash>
            candidateSetsForNode;
        candidateSetsForNode.reserve(totalPoints);

        // 并行前向搜索：内层循环并行，chunk=1024
        for (size_t idx = 0; idx + 1 < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
#pragma omp parallel for
            for (size_t xi = 0; xi < points.size(); ++xi) {
                int tid = omp_get_thread_num();
                auto &localMap = localCandidateSets[tid];
                tableint cur_c = points[xi];
                char *data_point = indices[idx]->getDataByInternalId(cur_c);
                for (size_t target_idx = idx + 1; target_idx < indices.size(); ++target_idx) {
                    auto &last_entry_point = entry_points_collect[idx][target_idx][cur_c];
                    int higherLevel = (last_entry_point == -1) ? indices[target_idx]->maxlevel_ : level;
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        typename HierarchicalNSW<dist_t>::CompareByFirst>
                        top_candidates;

                    indices[target_idx]->search2Layer(
                        data_point,
                        last_entry_point,
                        higherLevel,
                        level,
                        10,
                        &top_candidates,
                        index_offsets[target_idx]);

                    auto temp_q = top_candidates;
                    while (!temp_q.empty()) {
                        auto current = temp_q.top();
                        temp_q.pop();
                        tableint raw_target = current.second;
                        tableint target_node = raw_target - index_offsets[target_idx];
                        tableint source_node = cur_c + index_offsets[idx];
                        localMap[{idx, cur_c}].emplace_back(current);
                        localMap[{target_idx, target_node}]
                            .emplace_back(current.first, source_node);
                    }
                }
            }
        }

        // 并行合并本地到全局 map
        // #pragma omp parallel for schedule(static)
        for (int t = 0; t < numThreads; ++t) {
            for (auto &kv : localCandidateSets[t]) {
                auto &globalVec = candidateSetsForNode[kv.first];
                globalVec.insert(
                    globalVec.end(),
                    kv.second.begin(),
                    kv.second.end());
            }
        }

        // 第二阶段：链接合并，内层大循环并行，chunk=512
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
            size_t offset = index_offsets[idx];
#pragma omp parallel for
            for (size_t pi = 0; pi < points.size(); ++pi) {
                tableint cur_c = points[pi];
                std::priority_queue<
                    std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst>
                    top_candidates;

                auto ll_cur = indices[idx]->get_linklist_at_level(cur_c, level);
                size_t linklistCount = indices[idx]->getListCount(ll_cur);
                auto data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto dist = indices[idx]->get_dist_at_level(cur_c, level);

                for (size_t i = 0; i < linklistCount; ++i) {
                    top_candidates.emplace(dist[i], data[i] + offset);
                }

                auto key = std::make_pair(idx, cur_c);
                if (auto it = candidateSetsForNode.find(key); it != candidateSetsForNode.end()) {
                    for (auto &p : it->second) {
                        top_candidates.emplace(p.first, p.second);
                    }
                }

                size_t Mcurmax = level ? merged_index->maxM_ : merged_index->maxM0_;
                merged_index->getNeighborsByHeuristic2(
                    top_candidates, Mcurmax, true);

                ll_cur = merged_index->get_linklist_at_level(cur_c + offset, level);
                merged_index->setListCount(ll_cur, top_candidates.size());
                auto new_data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto new_dist = reinterpret_cast<dist_t *>(
                    merged_index->get_dist_at_level(cur_c + offset, level));
                for (size_t i = 0; !top_candidates.empty(); ++i) {
                    new_data[i] = top_candidates.top().second;
                    new_dist[i] = top_candidates.top().first;
                    top_candidates.pop();
                }
            }
        }
    }

    for (int level = 0; level >= 0; level--) {
        int index_with_nodes = -1;
        int count_indices_with_nodes = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!layer_nodes_for_indices[idx][level].empty()) {
                index_with_nodes = static_cast<int>(idx);
                ++count_indices_with_nodes;
            }
        }
        if (count_indices_with_nodes == 1) {
            merged_index->deepCopyOneLayerOnIndex(
                indices[index_with_nodes],
                level, index_offsets[index_with_nodes],
                layer_nodes_for_indices[index_with_nodes]);
            continue;
        }
        if (count_indices_with_nodes == 0) continue;

        // 收集并预分配 mergedDataPoints
        size_t totalPoints = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &vec = mergedDataPointsFromIndices[idx];
            vec.reserve(vec.size() + layer_nodes_for_indices[idx][level].size());
            for (auto old_c : layer_nodes_for_indices[idx][level]) {
                vec.push_back(old_c);
            }
            totalPoints += vec.size();
        }

        // 清空且保留本地 map 空间
        for (int t = 0; t < numThreads; ++t) {
            localCandidateSets[t].clear();
        }
        std::unordered_map<
            std::pair<size_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            pair_hash>
            candidateSetsForNode;
        candidateSetsForNode.reserve(totalPoints);

        // 并行前向搜索：内层循环并行，chunk=1024
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
#pragma omp parallel for
            for (size_t xi = 0; xi < points.size(); ++xi) {
                int tid = omp_get_thread_num();
                auto &localMap = localCandidateSets[tid];
                tableint cur_c = points[xi];
                char *data_point = indices[idx]->getDataByInternalId(cur_c);
                size_t linklistCount = indices[idx]->getListCount(indices[idx]->get_linklist_at_level(cur_c, 0));
                int lowerLevel = (indices[idx]->element_levels_[cur_c] > 0 || linklistCount < int(0.75 * M)) ? 0 : 1;
                // int lowerLevel = (indices[idx]->element_levels_[cur_c] > 0) ? 0 : 1;

                for (size_t target_idx = idx + 1; target_idx < indices.size(); ++target_idx) {
                    auto &last_entry_point = entry_points_collect[idx][target_idx][cur_c];
                    int higherLevel = (last_entry_point == -1) ? indices[target_idx]->maxlevel_ : std::max(0, std::min(indices[target_idx]->maxlevel_, lowerLevel));
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        typename HierarchicalNSW<dist_t>::CompareByFirst>
                        top_candidates;

                    indices[target_idx]->search2Layer(
                        data_point,
                        last_entry_point,
                        higherLevel,
                        lowerLevel,
                        3,
                        &top_candidates,
                        index_offsets[target_idx]);

                    auto temp_q = top_candidates;
                    while (!temp_q.empty()) {
                        auto current = temp_q.top();
                        temp_q.pop();
                        tableint raw_target = current.second;
                        tableint target_node = raw_target - index_offsets[target_idx];
                        tableint source_node = cur_c + index_offsets[idx];
                        localMap[{idx, cur_c}].emplace_back(current);
                        localMap[{target_idx, target_node}]
                            .emplace_back(current.first, source_node);
                    }
                }
            }
        }

        // 并行合并本地到全局 map
        // #pragma omp parallel for schedule(static)
        for (int t = 0; t < numThreads; ++t) {
            for (auto &kv : localCandidateSets[t]) {
                auto &globalVec = candidateSetsForNode[kv.first];
                globalVec.insert(
                    globalVec.end(),
                    kv.second.begin(),
                    kv.second.end());
            }
        }

        // 第二阶段：链接合并，内层大循环并行，chunk=512
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
            size_t offset = index_offsets[idx];
#pragma omp parallel for
            for (size_t pi = 0; pi < points.size(); ++pi) {
                tableint cur_c = points[pi];
                std::priority_queue<
                    std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst>
                    top_candidates;

                auto ll_cur = indices[idx]->get_linklist_at_level(cur_c, level);
                size_t linklistCount = indices[idx]->getListCount(ll_cur);
                auto data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto dist = indices[idx]->get_dist_at_level(cur_c, level);

                for (size_t i = 0; i < linklistCount; ++i) {
                    top_candidates.emplace(dist[i], data[i] + offset);
                }

                auto key = std::make_pair(idx, cur_c);
                if (auto it = candidateSetsForNode.find(key); it != candidateSetsForNode.end()) {
                    for (auto &p : it->second) {
                        top_candidates.emplace(p.first, p.second);
                    }
                }

                size_t Mcurmax = level ? merged_index->maxM_ : merged_index->maxM0_;
                merged_index->getNeighborsByHeuristic2(
                    top_candidates, Mcurmax, true);

                ll_cur = merged_index->get_linklist_at_level(cur_c + offset, level);
                merged_index->setListCount(ll_cur, top_candidates.size());
                auto new_data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto new_dist = reinterpret_cast<dist_t *>(
                    merged_index->get_dist_at_level(cur_c + offset, level));
                for (size_t i = 0; !top_candidates.empty(); ++i) {
                    new_data[i] = top_candidates.top().second;
                    new_dist[i] = top_candidates.top().first;
                    top_candidates.pop();
                }
            }
        }
    }

    printf("time stage 2: %f\n", elapsed() - t0);
    printf("total time: %f\n", elapsed() - t00);
    return merged_index;
}

    template <typename dist_t>
HierarchicalNSW<dist_t> *HNSWRefinement(HierarchicalNSW<dist_t> *index1, L2Space *space, bool random_select = true) {
    // 1) 构建新索引，并复制基本参数
    size_t max_elements = index1->max_elements_;
    size_t M = index1->M_;
    size_t ef_construction = index1->ef_construction_;

    double t0 = elapsed();
    auto index2 = new HierarchicalNSW<dist_t>(space, max_elements, M, ef_construction);
    index2->setMaxLevel(index1->maxlevel_);
    index2->cur_element_count.store(index1->cur_element_count);
    index2->enterpoint_node_ = index1->enterpoint_node_;

    // 2) 复制数据存储 (level0)
    index2->data_level0_memory_ = (char *)malloc(max_elements * index2->size_data_per_element_);
    if (!index2->data_level0_memory_) {
        throw std::runtime_error("Refinement failed: cannot allocate data_level0_memory");
    }
    memcpy(index2->data_level0_memory_, index1->data_level0_memory_, index1->cur_element_count * index1->size_data_per_element_);

    index2->dist_level0_memory_ = (char *)malloc(max_elements * index2->size_dist_per_element_);
    if (!index2->dist_level0_memory_) {
        throw std::runtime_error("Refinement failed: cannot allocate dist_level0_memory");
    }
    memcpy(index2->dist_level0_memory_, index1->dist_level0_memory_, index1->cur_element_count * index1->size_dist_per_element_);

    index2->element_levels_ = index1->element_levels_;
    index2->label_lookup_ = index1->label_lookup_;

    for (int id = 0; id < index1->cur_element_count; id++) {
        if (index1->element_levels_[id] < 1)
            continue;
        int level = index1->element_levels_[id];
        index2->linkLists_[id] = (char *)malloc(index2->size_links_per_element_ * level + 1);
        if (index2->linkLists_[id] == nullptr) {
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        }
        memcpy(index2->linkLists_[id], index1->linkLists_[id], index2->size_links_per_element_ * level + 1);
        index2->dist_linkLists_[id] = (char *)malloc(index2->size_dist_links_per_element_ * level + 1);
        if (index2->dist_linkLists_[id] == nullptr) {
            throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linklist");
        }
        memcpy(index2->dist_linkLists_[id], index1->dist_linkLists_[id], index2->size_dist_links_per_element_ * level + 1);
    }
    printf("[%.3f s] step 1\n", elapsed() - t0);

    t0 = elapsed();

    // 放到外面或函数开头
    std::atomic<bool> stop{false};
    std::atomic<size_t> matchCount{0};
    const int max_fixed_count = 1000000;

    size_t N = index1->cur_element_count; // 总元素数
    size_t K = max_fixed_count;

    // std::vector<int> pool(N);
    // std::iota(pool.begin(), pool.end(), 0);

    // // 2. 随机打乱并截取前 K
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::shuffle(pool.begin(), pool.end(), gen);
    // pool.resize(K);

    // // 3. 构造一个只读的掩码数组：0/1 标记哪些 ID 在样本集中
    // // std::vector<char> mask(N, 0);
    // // for (int id : pool) {
    // //     mask[id] = 1;
    // // }
    // std::vector<int> marked_ids(K);
    // marked_ids = pool;

    std::vector<tableint> marked_ids;
    marked_ids.reserve(K);

    // 互斥锁保护对 marked_ids 的写入
    std::mutex mtx;
    std::atomic<bool> stop_flag{false};

#pragma omp parallel for schedule(dynamic, 64)
    for (tableint id = 0; id < (tableint)N; ++id) {
        // 如果已经超过 K，就跳过后续所有工作
        if (stop_flag.load(std::memory_order_relaxed))
            continue;

        // 检查当前点的 level
        // if (index2->element_levels_[id] <= 0)
        //     continue;
        auto *ll = index2->get_linklist_at_level(id, 0);
        size_t cnt = index2->getListCount(ll);
        if (cnt < 45 || cnt > 65) {
            continue;
        }

        // 先检查一次大小，避免不必要的锁竞争
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (marked_ids.size() > K) {
                stop_flag.store(true, std::memory_order_relaxed);
                continue;
            }
            marked_ids.push_back(id);
        }

        // 读取 level 0 上的所有邻居

        // auto *neighbors = reinterpret_cast<tableint *>(ll + 1);

        // for (size_t j = 0; j < cnt; ++j) {
        //     tableint nid = neighbors[j];

        //     // 再次检查全局大小上限
        //     {
        //         std::lock_guard<std::mutex> lock(mtx);
        //         if (marked_ids.size() > K) {
        //             stop_flag.store(true, std::memory_order_relaxed);
        //             break;
        //         }
        //         // 将 nid 加入结果向量
        //         marked_ids.push_back(nid);
        //     }
        //     if (stop_flag.load(std::memory_order_relaxed))
        //         break;
        // }
    }

    printf("[%.3f s] step 1.5\n", elapsed() - t0);
    printf("实际采样的点数 = %zu\n", marked_ids.size());
    t0 = elapsed();

#pragma omp parallel for schedule(dynamic)
    // for (tableint id = 0; id < index1->cur_element_count; ++id) {
    for (tableint id : marked_ids) {
        // 如果已经达到了阈值，就直接跳过
        // if (stop.load(std::memory_order_relaxed)) {
        //     continue;
        // }

        // 原先的参数设置
        int level_higher = 0, level_lower = 0, cnt = 128, offset = 0;
        // tableint ep = id;

        // 先检查 linklistCount，再决定是否计数和进入后续工作
        linklistsizeint *ll2 = index2->get_linklist_at_level(id, 0);
        size_t linklistCount = index2->getListCount(ll2);
        // if (!mask[id]) {
        //     continue;
        // }

        // 原子自增，并检查是否超过阈值
        // size_t prev = matchCount.fetch_add(1, std::memory_order_relaxed);
        // if (prev + 1 >= max_fixed_count) {
        //     // 只要有任意一个线程发现到达阈值，就把 stop 置为 true
        //     stop.store(true, std::memory_order_relaxed);
        // }

        // // 如果刚好超过阈值，可以选择不再做后续工作
        // if (stop.load(std::memory_order_relaxed) && prev + 1 > max_fixed_count) {
        //     continue;
        // }

        // ------------ 真正的 refinement 逻辑 ------------
        std::priority_queue<
            std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            typename HierarchicalNSW<dist_t>::CompareByFirst>
            top_candidates;

        index1->search2Layer(
            index1->getDataByInternalId(id),
            id,
            level_higher,
            level_lower,
            cnt,
            &top_candidates,
            offset);
        index2->getNeighborsByHeuristic2(
            top_candidates,
            index2->maxM0_,
            false,
            id);

        index2->setListCount(ll2, top_candidates.size());
        tableint *nbrs = (tableint *)(ll2 + 1);
        dist_t *dists = (dist_t *)(index2->get_dist_at_level(id, 0));

        size_t idx = 0;
        while (!top_candidates.empty()) {
            auto pr = top_candidates.top();
            top_candidates.pop();
            nbrs[idx] = pr.second;
            dists[idx] = pr.first;
            ++idx;
        }
        // -----------------------------------------------
    }

    // 循环结束后可以读取 matchCount
    printf("实际执行到阈值前的 matchCount = %zu\n", matchCount.load());
    printf("[%.3f s] step 2\n", elapsed() - t0);
    // printf("matchCount: %zu\n", matchCount);

    return index2;
}


    template <typename dist_t>
HierarchicalNSW<dist_t> *MultiIndexMerger(std::vector<HierarchicalNSW<dist_t> *> indices, L2Space *space, size_t M = -1, size_t ef_construction = -1) {
    if (indices.empty()) {
        throw std::runtime_error("No indices provided for merging");
    }

    if (indices.size() == 1) {
        return indices[0];
    }

    double t0 = elapsed();
    std::sort(indices.begin(), indices.end(),
              [](auto a, auto b) { return a->cur_element_count != b->cur_element_count ? a->cur_element_count < b->cur_element_count : a->maxlevel_ < b->maxlevel_; });

    size_t total_elements = 0;
    size_t maxLevel = 0;
    std::vector<size_t> index_offsets = {0};

    for (auto index : indices) {
        total_elements += index->getCurrentElementCount();
        maxLevel = std::max(maxLevel, (size_t)index->maxlevel_);
        index_offsets.push_back(total_elements);
    }

    M = (M == -1) ? (*std::max_element(indices.begin(), indices.end(), [](auto a, auto b) { return a->M_ < b->M_; }))->M_ : M;

    ef_construction = (ef_construction == -1) ? (*std::max_element(indices.begin(), indices.end(), [](auto a, auto b) { return a->ef_construction_ < b->ef_construction_; }))->ef_construction_ : ef_construction;

    size_t new_max_elements = std::accumulate(indices.begin(), indices.end(), 0, [](size_t sum, auto index) { return sum + index->max_elements_; });

    HierarchicalNSW<dist_t> *merged_index = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    merged_index->setMaxLevel(maxLevel);
    merged_index->cur_element_count.store(total_elements);

    auto allocateMemory = [&](char *&memory, const char *errorMsg, size_t size) {
        memory = (char *)malloc(size);
        if (memory == nullptr)
            throw std::runtime_error(errorMsg);
    };

    auto allocateMemory_layer = [&](hnswlib::HierarchicalNSW<dist_t> *index, int element_count_offset) {
#pragma omp parallel for schedule(dynamic)
        for (int id = 0; id < index->cur_element_count; id++) {
            if (index->element_levels_[id] < 1)
                continue;
            int level = index->element_levels_[id];
            tableint new_c = id + element_count_offset;
            allocateMemory(merged_index->linkLists_[new_c],
                           "Not enough memory: addPoint failed to allocate linklist",
                           merged_index->size_links_per_element_ * level + 1);
            allocateMemory(merged_index->dist_linkLists_[new_c],
                           "Not enough memory: addPoint failed to allocate dist_linklist",
                           merged_index->size_dist_links_per_element_ * level + 1);
        }
    };

    allocateMemory(merged_index->data_level0_memory_, "Not enough memory: MultiIndexMerger failed to allocate level0 data",
                   merged_index->max_elements_ * merged_index->size_data_per_element_);

    allocateMemory(merged_index->dist_level0_memory_, "Not enough memory: MultiIndexMerger failed to allocate level0 distance",
                   merged_index->max_elements_ * merged_index->size_dist_per_element_);

    std::vector<std::vector<std::vector<int>>> layer_nodes_for_indices(indices.size());
    std::vector<std::vector<std::vector<tableint>>> entry_points_collect(indices.size());
    for (size_t i = 0; i < indices.size(); i++) {
        layer_nodes_for_indices[i].resize(maxLevel + 2);
        indices[i]->searchNodeOnEachLayer(layer_nodes_for_indices[i]);

        entry_points_collect[i].resize(indices.size());
        size_t element_count = indices[i]->getCurrentElementCount();
        for (size_t j = 0; j < indices.size(); j++) {
            if (i != j) {
                entry_points_collect[i][j].resize(element_count, -1);
            }
        }

        allocateMemory_layer(indices[i], index_offsets[i]);

        memcpy(merged_index->element_levels_.data() + index_offsets[i],
               indices[i]->element_levels_.data(),
               sizeof(int) * indices[i]->getCurrentElementCount());
        memcpy(merged_index->data_level0_memory_ + index_offsets[i] * merged_index->size_data_per_element_,
               indices[i]->data_level0_memory_,
               indices[i]->getCurrentElementCount() * merged_index->size_data_per_element_);
        memcpy(merged_index->dist_level0_memory_ + index_offsets[i] * merged_index->size_dist_per_element_,
               indices[i]->dist_level0_memory_,
               indices[i]->getCurrentElementCount() * merged_index->size_dist_per_element_);
    }

    std::vector<std::vector<int>> mergedDataPointsFromIndices(indices.size());

    std::vector<int> newLayer;
    size_t total_top_layer_size =
        std::accumulate(layer_nodes_for_indices.begin(),
                        layer_nodes_for_indices.end(),
                        0,
                        [maxLevel](size_t sum, auto &nodes) {
                            return sum + nodes[maxLevel].size();
                        });

    if (total_top_layer_size > M) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        int cnt = 0;

        auto processTopLayerNode = [&](size_t idx, auto &layer_nodes, size_t offset) {
            auto it = layer_nodes.begin();
            while (it != layer_nodes.end()) {
                if (distribution(merged_index->level_generator_) < 1.0 / M || cnt == M) {
                    newLayer.push_back((*it) + offset);
                    cnt = 0;
                    tableint new_c = *it;

                    allocateMemory(merged_index->linkLists_[new_c + offset], "Not enough memory: addPoint failed to allocate linklist",
                                   merged_index->size_links_per_element_ * (maxLevel + 1) + 1);

                    allocateMemory(merged_index->dist_linkLists_[new_c + offset], "Not enough memory: addPoint failed to allocate dist_linkLists",
                                   merged_index->size_dist_links_per_element_ * (maxLevel + 1) + 1);

                    merged_index->element_levels_[new_c + offset] = (maxLevel + 1);

                    memcpy(merged_index->getDataByInternalId(new_c + offset), indices[idx]->getDataByInternalId(new_c), merged_index->data_size_);
                    merged_index->setExternalLabel(new_c + offset, indices[idx]->getExternalLabel(new_c));
                    mergedDataPointsFromIndices[idx].push_back(new_c);

                    it = layer_nodes.erase(it);
                } else {
                    cnt++;
                    ++it;
                }
            }
        };

        for (size_t idx = 0; idx < indices.size(); idx++) {
            processTopLayerNode(idx, layer_nodes_for_indices[idx][maxLevel], index_offsets[idx]);
        }

        if (!newLayer.empty()) {
            for (int newLayerIter = 0; newLayerIter < newLayer.size(); newLayerIter++) {
                linklistsizeint *ll_cur = merged_index->get_linklist_at_level(newLayer[newLayerIter], maxLevel + 1);
                merged_index->setListCount(ll_cur, newLayer.size() - 1);
                tableint *data = (tableint *)(ll_cur + 1);
                dist_t *dist = merged_index->get_dist_at_level(newLayer[newLayerIter], maxLevel + 1);

                for (size_t it = 0, idx = 0; it < newLayer.size(); it++) {
                    if (it == newLayerIter)
                        continue;
                    dist_t dist0 = merged_index->fstdistfunc_(merged_index->getDataByInternalId(newLayer[newLayerIter]), merged_index->getDataByInternalId(newLayer[it]),
                                                              merged_index->dist_func_param_);
                    data[idx] = newLayer[it];
                    dist[idx] = dist0;
                    idx++;
                }
            }

            merged_index->setMaxLevel(maxLevel + 1);
            merged_index->enterpoint_node_ = newLayer[0];
            maxLevel += 1;
        } else {
            merged_index->enterpoint_node_ = layer_nodes_for_indices.back()[maxLevel][0] + index_offsets[indices.size() - 1];
        }
    } else {
        merged_index->enterpoint_node_ = layer_nodes_for_indices.back()[maxLevel][0] + index_offsets[indices.size() - 1];
    }

    printf("time stage 1: %f\n", elapsed() - t0);
    t0 = elapsed();

    int numThreads = omp_get_max_threads();
    struct alignas(64) LocalMap :
        std::unordered_map<std::pair<size_t, tableint>,
                           std::vector<std::pair<dist_t, tableint>>,
                           pair_hash> {};
    std::vector<LocalMap> localCandidateSets(numThreads);
    
    for (int level = maxLevel; level >= 0; --level) {
        // 统计层中活跃索引
        int index_with_nodes = -1;
        int count_indices_with_nodes = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!layer_nodes_for_indices[idx][level].empty()) {
                index_with_nodes = static_cast<int>(idx);
                ++count_indices_with_nodes;
            }
        }
        if (count_indices_with_nodes == 1) {
            deepCopyOneLayerOnIndex(
                indices[index_with_nodes], merged_index,
                level, index_offsets[index_with_nodes],
                layer_nodes_for_indices[index_with_nodes]
            );
            continue;
        }
        if (count_indices_with_nodes == 0) continue;
    
        // 收集并预分配 mergedDataPoints
        size_t totalPoints = 0;
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &vec = mergedDataPointsFromIndices[idx];
            vec.reserve(vec.size()+layer_nodes_for_indices[idx][level].size());
            for (auto old_c : layer_nodes_for_indices[idx][level]) {
                vec.push_back(old_c);
            }
            totalPoints += vec.size();
        }
    
        // 清空且保留本地 map 空间
        for (int t = 0; t < numThreads; ++t) {
            localCandidateSets[t].clear();
        }
        std::unordered_map<
            std::pair<size_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            pair_hash
        > candidateSetsForNode;
        candidateSetsForNode.reserve(totalPoints);
    
        // 并行前向搜索：内层循环并行，chunk=1024
        for (size_t idx = 0; idx + 1 < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
            #pragma omp parallel for schedule(dynamic,1024)
            for (size_t xi = 0; xi < points.size(); ++xi) {
                int tid = omp_get_thread_num();
                auto &localMap = localCandidateSets[tid];
                tableint cur_c = points[xi];
                char *data_point = indices[idx]->getDataByInternalId(cur_c);
                for (size_t target_idx = idx + 1; target_idx < indices.size(); ++target_idx) {
                    auto &last_entry_point = entry_points_collect[idx][target_idx][cur_c];
                    int higherLevel = (last_entry_point == -1)
                        ? indices[target_idx]->maxlevel_
                        : level;
                    std::priority_queue<
                        std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        typename HierarchicalNSW<dist_t>::CompareByFirst
                    > top_candidates;
    
                    indices[target_idx]->search2Layer(
                        data_point,
                        last_entry_point,
                        higherLevel,
                        level,
                        13,
                        &top_candidates,
                        index_offsets[target_idx]
                    );
    
                    auto temp_q = top_candidates;
                    while (!temp_q.empty()) {
                        auto current = temp_q.top(); temp_q.pop();
                        tableint raw_target = current.second;
                        tableint target_node = raw_target - index_offsets[target_idx];
                        tableint source_node = cur_c + index_offsets[idx];
                        localMap[{idx, cur_c}].emplace_back(current);
                        localMap[{target_idx, target_node}]
                            .emplace_back(current.first, source_node);
                    }
                }
            }
        }
    
        // 并行合并本地到全局 map
        // #pragma omp parallel for schedule(static)
        for (int t = 0; t < numThreads; ++t) {
            for (auto &kv : localCandidateSets[t]) {
                auto &globalVec = candidateSetsForNode[kv.first];
                globalVec.insert(
                    globalVec.end(),
                    kv.second.begin(),
                    kv.second.end()
                );
            }
        }
    
        // 第二阶段：链接合并，内层大循环并行，chunk=512
        for (size_t idx = 0; idx < indices.size(); ++idx) {
            auto &points = mergedDataPointsFromIndices[idx];
            size_t offset = index_offsets[idx];
            #pragma omp parallel for schedule(dynamic,512)
            for (size_t pi = 0; pi < points.size(); ++pi) {
                tableint cur_c = points[pi];
                std::priority_queue<
                    std::pair<dist_t, tableint>,
                    std::vector<std::pair<dist_t, tableint>>,
                    typename HierarchicalNSW<dist_t>::CompareByFirst
                > top_candidates;
    
                auto ll_cur = indices[idx]->get_linklist_at_level(cur_c, level);
                size_t linklistCount = indices[idx]->getListCount(ll_cur);
                auto data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto dist = indices[idx]->get_dist_at_level(cur_c, level);
    
                for (size_t i = 0; i < linklistCount; ++i) {
                    top_candidates.emplace(dist[i], data[i] + offset);
                }
    
                auto key = std::make_pair(idx, cur_c);
                if (auto it = candidateSetsForNode.find(key); it != candidateSetsForNode.end()) {
                    for (auto &p : it->second) {
                        top_candidates.emplace(p.first, p.second);
                    }
                }
    
                size_t Mcurmax = level ? merged_index->maxM_ : merged_index->maxM0_;
                merged_index->getNeighborsByHeuristic2(
                    top_candidates, Mcurmax, true
                );
    
                ll_cur = merged_index->get_linklist_at_level(cur_c + offset, level);
                merged_index->setListCount(ll_cur, top_candidates.size());
                auto new_data = reinterpret_cast<tableint *>(ll_cur + 1);
                auto new_dist = reinterpret_cast<dist_t *>(
                    merged_index->get_dist_at_level(cur_c + offset, level)
                );
                for (size_t i = 0; !top_candidates.empty(); ++i) {
                    new_data[i] = top_candidates.top().second;
                    new_dist[i] = top_candidates.top().first;
                    top_candidates.pop();
                }
            }
        }
    }
    
    printf("time stage 2: %f\n", elapsed() - t0);

    // for (int level = maxLevel; level >= 0; level--) {
    //     int index_with_nodes = -1;
    //     int count_indices_with_nodes = 0;

    //     for (size_t idx = 0; idx < indices.size(); idx++) {
    //         if (!layer_nodes_for_indices[idx][level].empty()) {
    //             index_with_nodes = idx;
    //             count_indices_with_nodes++;
    //         }
    //     }

    //     if (count_indices_with_nodes == 1) {
    //         deepCopyOneLayerOnIndex(indices[index_with_nodes], merged_index, level, index_offsets[index_with_nodes], layer_nodes_for_indices[index_with_nodes]);
    //         continue;
    //     }

    //     if (count_indices_with_nodes == 0) {
    //         continue;
    //     }

    //     auto copyDataAndLabels = [&](size_t idx, size_t offset) {
    //         auto &layer_nodes = layer_nodes_for_indices[idx][level];

    //         for (int id = 0; id < layer_nodes.size(); id++) {
    //             tableint old_c = layer_nodes[id];
    //             mergedDataPointsFromIndices[idx].push_back(old_c);
    //         }
    //     };

    //     for (size_t idx = 0; idx < indices.size(); idx++) {
    //         copyDataAndLabels(idx, index_offsets[idx]);
    //     }

    //     // 为每个节点创建候选集，而不是为每个索引创建
    //     std::unordered_map<std::pair<size_t, tableint>, std::vector<std::pair<dist_t, tableint>>, pair_hash> candidateSetsForNode;

    //     // 首先处理所有节点的正向搜索（每个节点搜索后续索引中的邻居）
    //     #pragma omp parallel for schedule(dynamic)
    //     for (size_t idx = 0; idx < indices.size(); idx++) {
    //         auto &mergedDataPoints = mergedDataPointsFromIndices[idx];

    //         for (int iter = 0; iter < mergedDataPoints.size(); iter++) {
    //             tableint cur_c = mergedDataPoints[iter];
    //             char *data_point = indices[idx]->getDataByInternalId(cur_c);

    //             // 只在后续索引中搜索
    //             for (size_t target_idx = idx + 1; target_idx < indices.size(); target_idx++) {
    //                 tableint &last_entry_point = entry_points_collect[idx][target_idx][cur_c];
    //                 int higherLevel = level;
    //                 if (last_entry_point == -1) {
    //                     higherLevel = indices[target_idx]->maxlevel_;
    //                 }

    //                 std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;

    //                 int cnt = 6;
    //                 indices[target_idx]->search2Layer(data_point,
    //                                                   last_entry_point,
    //                                                   higherLevel,
    //                                                   level,
    //                                                   cnt,
    //                                                   &top_candidates,
    //                                                   index_offsets[target_idx]);

    //                 if (!top_candidates.empty()) {
    //                     auto best_candidate = top_candidates.top();
    //                     last_entry_point = best_candidate.second - index_offsets[target_idx];
    //                 }

    //                 // 保存正向搜索结果
    //                 auto temp_queue = top_candidates;
    //                 while (!temp_queue.empty()) {
    //                     std::pair<dist_t, tableint> current = temp_queue.top();
    //                     temp_queue.pop();

    //                     tableint target_node = current.second - index_offsets[target_idx];
    //                     tableint source_node = cur_c + index_offsets[idx];

    //                     // 正向边：当前节点 -> 目标节点
    //                     std::pair<size_t, tableint> node_key = {idx, cur_c};
    //                     candidateSetsForNode[node_key].push_back(std::make_pair(current.first, current.second)); // 保存目标节点的全局ID

    //                     // 反向边：目标节点 -> 当前节点
    //                     std::pair<size_t, tableint> target_key = {target_idx, target_node};
    //                     candidateSetsForNode[target_key].push_back(std::make_pair(current.first, source_node)); // 保存当前节点的全局ID
    //                 }
    //             }
    //         }
    //     }

    //     // 合并所有索引的链接，处理每个节点的连边
    //     #pragma omp parallel for schedule(dynamic)
    //     for (size_t idx = 0; idx < indices.size(); idx++) {
    //         auto &mergedDataPoints = mergedDataPointsFromIndices[idx];
    //         size_t offset = index_offsets[idx];

    //         for (int iter = 0; iter < mergedDataPoints.size(); iter++) {
    //             tableint cur_c = mergedDataPoints[iter];

    //             // 获取当前节点在原索引中的邻居
    //             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;

    //             // 1. 添加原索引中的邻居
    //             linklistsizeint *ll_cur = indices[idx]->get_linklist_at_level(cur_c, level);
    //             size_t linklistCount = indices[idx]->getListCount(ll_cur);
    //             tableint *data = (tableint *)(ll_cur + 1);
    //             dist_t *dist = indices[idx]->get_dist_at_level(cur_c, level);

    //             for (size_t i = 0; i < linklistCount; i++) {
    //                 tableint candidate_id = data[i] + offset;
    //                 dist_t dist1 = dist[i];
    //                 top_candidates.emplace(dist1, candidate_id);
    //             }

    //             // 2. 添加候选集中的节点（包括正向和反向边）
    //             std::pair<size_t, tableint> node_key = {idx, cur_c};
    //             if (candidateSetsForNode.find(node_key) != candidateSetsForNode.end()) {
    //                 const auto &candidates = candidateSetsForNode[node_key];
    //                 for (const auto &p : candidates) {
    //                     top_candidates.emplace(p.first, p.second);
    //                 }
    //             }

    //             // 使用启发式算法选择最佳邻居
    //             size_t Mcurmax = level ? merged_index->maxM_ : merged_index->maxM0_;
    //             merged_index->getNeighborsByHeuristic2(top_candidates, Mcurmax, false);

    //             // 更新合并后索引中的链接
    //             ll_cur = merged_index->get_linklist_at_level(cur_c + offset, level);
    //             merged_index->setListCount(ll_cur, top_candidates.size());
    //             data = (tableint *)(ll_cur + 1);
    //             dist_t *distData = (dist_t *)merged_index->get_dist_at_level(cur_c + offset, level);

    //             for (size_t i = 0; top_candidates.size() > 0; i++) {
    //                 data[i] = top_candidates.top().second;
    //                 distData[i] = top_candidates.top().first;
    //                 top_candidates.pop();
    //             }
    //         }
    //     }
    // }

    return merged_index;
}

          // void resizeIndex(size_t new_max_elements)
        // {
        //     if (new_max_elements < cur_element_count)
        //         throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        //     visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        //     element_levels_.resize(new_max_elements);

        //     std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        //     // Reallocate base layer
        //     char *data_level0_memory_new = (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        //     if (data_level0_memory_new == nullptr)
        //         throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        //     data_level0_memory_ = data_level0_memory_new;
        //     char *dist_level0_memory_new = (char *)realloc(dist_level0_memory_, new_max_elements * size_dist_per_element_);
        //     if (dist_level0_memory_new == nullptr)
        //         throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        //     data_level0_memory_ = dist_level0_memory_new;

        //     // Reallocate all other layers
        //     char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
        //     if (linkLists_new == nullptr)
        //         throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        //     linkLists_ = linkLists_new;
        //     char **dist_linkLists_new = (char **)realloc(dist_linkLists_, sizeof(void *) * new_max_elements);
        //     if (dist_linkLists_new == nullptr)
        //         throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        //     linkLists_ = dist_linkLists_new;

        //     max_elements_ = new_max_elements;
        // }

  template <typename dist_t>
void HierarchicalNSW<dist_t>::getNeighborsByHeuristic2(const size_t M,
                                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                                                       &top_candidates, bool collect_metrics, tableint cur_c, tableint static_upperbound, tableint static_lowerbound, int cnt) {
    if (collect_metrics && top_candidates.size() < M) {
        return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
        if (top_candidates.top().second != cur_c)
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }

    while (queue_closest.size()) {
        if (return_list.size() >= M)
            break;
        std::pair<dist_t, tableint> curent_pair = queue_closest.top();
        dist_t dist_to_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<dist_t, tableint> second_pair : return_list) {
            dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second), getDataByInternalId(curent_pair.second), dist_func_param_);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.push_back(curent_pair);
        }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
        top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
}

        /*
         *  This function, which is used in search2Layer, searching the graph from the top layer to the level_low layer.
         *  If the enterpoint_node is -1, it will be set to enterpoint_node_. Otherwise, the enterpoint_node will be used as the starting point, as it's the closest point to the query on layer level_low + 1.
         *  The function will return the closest point to the query on layer level_low.
         */
        void search2Layer(
            const void *query_data,
            tableint &enterpoint_node,
            int level_higher,
            int level_lower,
            int &cnt)
        {
            tableint currObj = enterpoint_node == -1 ? enterpoint_node_ : enterpoint_node;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
            // printf("method0\n");
            for (int level = level_higher; level >= level_lower; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data;
                    cnt++;
                    data = (unsigned int *)get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;

                    tableint *datal = (tableint *)(data + 1);

                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                        {
                            throw std::runtime_error("cand error");
                        }
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                            // printf("replace candidateSet: %d, %f\n", currObj, curdist);
                        }
                    }
                }
            }
            // printf("method0, level: %d, enterpoint_node: %d, currObj: %d\n", level_lower, enterpoint_node, currObj);
            enterpoint_node = currObj;
        }

template <typename dist_t>
void HierarchicalNSW<dist_t>::search2Layer(
    const void *query_data,
    tableint prevObj,
    int level,
    int cnt,
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    int offset)
/*
 * This function, which is used in search2Layer, searching the graph on layer `level`.
 * This funciton will search based on the prevObj's connections on layer `level`. If prevObj is -1, it will be set to enterpoint_node_.
 * The function will also return the `top_candidates` which contains the `cnt`-closest points to the query on layer level_low. `cnt` can always be set as ef_construction.
 *
 */
{
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    if (prevObj != -1) {
        int *data = (int *)get_linklist_at_level(prevObj, level);
        size_t size = getListCount((linklistsizeint *)data);
        tableint *datal = (tableint *)(data + 1);

        for (int i = 0; i < size; i++) {
            tableint currObj = datal[i];
            if (currObj < 0 || currObj > max_elements_)
                throw std::runtime_error("currObj error");

            if (!isMarkedDeleted(currObj)) {
                dist_t dist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
                top_candidates.emplace(dist, currObj + offset);
                // enterpoint_node = currObj;
                // lowerBound = dist;
                candidateSet.emplace(-dist, currObj);
            } else {
                // lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-std::numeric_limits<dist_t>::max(), currObj);
            }
            visited_array[currObj] = visited_array_tag;
        }
    } else {
        tableint currObj = enterpoint_node_;
        if (!isMarkedDeleted(currObj)) {
            dist_t dist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
            top_candidates.emplace(dist, currObj + offset);
            // enterpoint_node = currObj;
            // lowerBound = dist;
            candidateSet.emplace(-dist, currObj);
        } else {
            // lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-std::numeric_limits<dist_t>::max(), currObj);
        }
        visited_array[currObj] = visited_array_tag;
    }
    dist_t lowerBound = -candidateSet.top().first;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == cnt) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        // std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data = (int *)get_linklist_at_level(curNodeNum, level);
        size_t size = getListCount((linklistsizeint *)data);
        tableint *datal = (tableint *)(data + 1);
        // #ifdef USE_SSE
        //                 _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        //                 _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        //                 _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        //                 _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
        // #endif

        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = *(datal + j);
            // #ifdef USE_SSE
            //                     _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
            //                     _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
            // #endif
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;
            char *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
            if (top_candidates.size() < cnt || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
                // printf("replace candidateSet: %d, %f\n", candidate_id, dist1);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id)) {
                    top_candidates.emplace(dist1, candidate_id + offset);
                    // enterpoint_node = candidate_id;
                }

                if (top_candidates.size() > cnt)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
    // printf("cnt: %d, enterpoint_node: %d\n", cnt, enterpoint_node);
}

template <typename dist_t, typename Func>
void HierarchicalNSW<dist_t>::measureExecutionTime(const std::string &label, Func &&lambda) {
    auto start = std::chrono::high_resolution_clock::now();
    lambda(); // Execute the lambda
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    time_counter_ += elapsed.count();
}

// std::vector<std::set<std::pair<int, int>>> layeredStructure;

template <typename dist_t>
void HierarchicalNSW<dist_t>::mergeIndex1BasedOnIndex2Connection3(
    HierarchicalNSW<dist_t> *index1,
    HierarchicalNSW<dist_t> *index2,
    int offset_index1,
    int offset_index2,
    int level,
    int &last_entry_point /* in & out */
    )                     /* This function is for the strategy that "neighbors' neighbors can be new neighbors" */
{
    tableint cur_c = index1->enterpoint_node_;
    std::unordered_set<tableint> seen_index1;
    seen_index1.insert(cur_c);
    char *data_point = index1->getDataByInternalId(cur_c);
    int cnt = 0;
    tableint entry_point = index2->search2Layer(data_point,
                                                last_entry_point,
                                                last_entry_point == -1 ? index2->maxlevel_ : level,
                                                level,
                                                cnt);
    // forward_step_avg_per_layer[level] += cnt;
    last_entry_point = entry_point;
    // int sum=0,cnt=0;
    // time_counter_ += elapsed()-t0;
    // double t0 = elapsed();
    auto calculateCandidates = [&](std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                   linklistsizeint *ll_cur, size_t linklistCount, int offset, std::unordered_set<tableint> &seen_candidates, bool useDist, tableint entry_point = -1, tableint cur_c = -1) {
        auto processCandidate = [&](tableint candidate_id, bool debugFlag = false) {
            if (seen_candidates.find(candidate_id) != seen_candidates.end())
                return;
            seen_candidates.insert(candidate_id);

            dist_t dist1 = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);

            if (useDist) {
                if (top_candidates.size() < ef_construction_ || dist1 < top_candidates.top().first) {
                    top_candidates.emplace(dist1, candidate_id);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
            } else {
                top_candidates.emplace(dist1, candidate_id);
            }

            // if (debugFlag)
            // {
            //     layeredStructure[level].insert({cur_c + offset_index1, dist1});
            // }
        };

        // Process the entry point
        if (entry_point != -1)
            processCandidate(entry_point + offset, true);

        // Process neighbors
        tableint *data = (tableint *)(ll_cur + 1);
        for (size_t iter = 0; iter < linklistCount; iter++) {
            processCandidate(data[iter] + offset);
        }
    };

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::unordered_set<tableint> seen_candidates;
    linklistsizeint *ll_cur = index1->get_linklist_at_level(cur_c, level);
    size_t linklistCount = index1->getListCount(ll_cur);
    calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index1, seen_candidates, false);
    // measureExecutionTime("calculateCandidates", [&](){ calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index1, seen_candidates, false); });

    ll_cur = index2->get_linklist_at_level(entry_point, level);
    linklistCount = index2->getListCount(ll_cur);
    calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index2, seen_candidates, true, entry_point, cur_c);
    // measureExecutionTime("calculateCandidates", [&](){ calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index2, seen_candidates, true); });
    // cnt++;sum+=top_candidates.size();

    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
    ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
    setListCount(ll_cur, top_candidates.size());

    // std::vector<tableint> selectedNeighbors;
    tableint selectedNeighbors = entry_point + offset_index2;

    auto processNeighbors = [&](linklistsizeint *ll_cur, std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates) {
        auto check_index1_point = [&](tableint x) -> bool {
            return (offset_index1 == 0 && x < offset_index2) || (offset_index1 != 0 && x > offset_index1);
        };
        tableint *data = (tableint *)(ll_cur + 1);
        dist_t *distData = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
        for (size_t idx = 0; !top_candidates.empty(); idx++) {
            tableint neighbor = top_candidates.top().second;
            data[idx] = neighbor;
            distData[idx] = top_candidates.top().first;
            top_candidates.pop();
        }
    };

    processNeighbors(ll_cur, top_candidates);

    auto pushQueue = [&](tableint cur_c, tableint selectedNeighbors, std::queue<std::pair<tableint, tableint>> &q) {
        linklistsizeint *ll_cur = index1->get_linklist_at_level(cur_c, level);
        size_t linklistCount = index1->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        for (size_t iter = 0; iter < linklistCount; iter++) {
            tableint candidate_id = data[iter];
            if (seen_index1.find(candidate_id) == seen_index1.end())
                q.push({candidate_id, selectedNeighbors});
        }
    };

    std::queue<std::pair<tableint, tableint>> q;
    pushQueue(cur_c, selectedNeighbors, q);
    // time_counter_ += elapsed()-t0;
    while (!q.empty()) {
        auto [cur_c, selectedNeighbors_index2] = q.front();
        q.pop();

        if (seen_index1.find(cur_c) != seen_index1.end())
            continue;
        seen_index1.insert(cur_c);
        data_point = index1->getDataByInternalId(cur_c);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::unordered_set<tableint> seen_candidates;
        ll_cur = index1->get_linklist_at_level(cur_c, level);
        linklistCount = index1->getListCount(ll_cur);
        calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index1, seen_candidates, false);
        // measureExecutionTime("calculateCandidates", [&]() { calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index1, seen_candidates, false); });

        int cnt = 0;
        selectedNeighbors_index2 = index2->search2Layer(data_point,
                                                        selectedNeighbors_index2 - offset_index2,
                                                        level,
                                                        level,
                                                        cnt)
                                   + offset_index2;
        // forward_step_avg_per_layer[level] += cnt;
        ll_cur = index2->get_linklist_at_level(selectedNeighbors_index2 - offset_index2, level);
        linklistCount = index2->getListCount(ll_cur);
        calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index2, seen_candidates, true, selectedNeighbors_index2 - offset_index2, cur_c);
        // measureExecutionTime("calculateCandidates", [&]() { calculateCandidates(top_candidates, ll_cur, linklistCount, offset_index2, seen_candidates, true); });

        // cnt++;sum+=top_candidates.size();
        getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
        ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
        setListCount(ll_cur, top_candidates.size());

        // std::vector<tableint> selectedNeighbors;
        processNeighbors(ll_cur, top_candidates);

        pushQueue(cur_c, selectedNeighbors_index2, q);
    }
    // printf("level %d, offset %d, cnt %d, sum %d, avg %.4f\n", level, offset_index1, cnt, sum, sum*1.0/cnt);
}

// std::vector<float> forward_step_avg_per_layer;

template <typename dist_t>
void HierarchicalNSW<dist_t>::cluster_points(
    int &K,
    int l,
    int max_iterations,
    size_t dim,
    const std::vector<int> &valid_ids,
    float *centroids,
    std::vector<std::vector<tableint>> &clusters) {
    size_t num_points = valid_ids.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(0, num_points - 1);

    // 选取 K 个随机点作为初始 centroids
    std::unordered_set<tableint> selected_centroids;
    std::vector<tableint> centroid_indices;
    while (selected_centroids.size() < K) {
        tableint rand_id = valid_ids[distrib(gen)];
        if (selected_centroids.insert(rand_id).second) {
            centroid_indices.push_back(rand_id);
            memcpy(centroids + (selected_centroids.size() - 1) * dim, getDataByInternalId(rand_id), dim * sizeof(float));
        }
    }

    int avg_cluster_size = num_points / K;
    int balance_threshold = avg_cluster_size / 2; // 允许的最大偏差

    for (int iter = 0; iter < max_iterations; iter++) {
        // std::cout << "Iteration " << iter << std::endl;
        clusters.clear();
        clusters.resize(K);

        std::vector<int> cluster_sizes(K, 0);

        for (tableint i : valid_ids) {
            std::priority_queue<std::pair<float, int>> nearest_centroids;
            void *point_data = getDataByInternalId(i);

            for (int j = 0; j < K; j++) {
                float dist = fstdistfunc_((float *)(centroids + j * dim), point_data, dist_func_param_);
                nearest_centroids.push({-dist, j});
            }

            while (!nearest_centroids.empty()) {
                int cluster_id = nearest_centroids.top().second;
                nearest_centroids.pop();

                if (cluster_sizes[cluster_id] < avg_cluster_size + balance_threshold) {
                    clusters[cluster_id].push_back(i);
                    cluster_sizes[cluster_id]++;
                    break;
                }
            }
        }

        // 计算新的 centroids 并直接写入 `centroids`，避免修改无效
        std::vector<bool> non_empty_clusters(K, false);
        for (int j = 0; j < K; j++) {
            if (!clusters[j].empty()) {
                std::vector<float> centroid(dim, 0);
                for (auto point : clusters[j]) {
                    float *point_data = (float *)getDataByInternalId(point);
                    for (size_t d = 0; d < dim; d++) {
                        centroid[d] += point_data[d];
                    }
                }
                for (size_t d = 0; d < dim; d++) {
                    centroid[d] /= clusters[j].size();
                }
                memcpy(centroids + j * dim, centroid.data(), dim * sizeof(float));
                non_empty_clusters[j] = true;
            }
        }

        // std::cout << "Remaining clusters: " << K << std::endl;

        for (int j = 0; j < K; j++) {
            if (!non_empty_clusters[j]) {
                tableint rand_id = valid_ids[distrib(gen)];
                memcpy(centroids + j * dim, getDataByInternalId(rand_id), dim * sizeof(float));
                clusters[j].clear();
                clusters[j].push_back(rand_id);
            }
        }
    }
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::assign_points_to_clusters(
    int level,
    int K,
    int T,
    int l,
    int dim,
    float *centroids,
    std::vector<std::vector<tableint>> &clusters) {
    clusters.clear();
    clusters.resize(K);
    std::unordered_map<tableint, int> point_selection_count;

    std::vector<tableint> closest_points(K);
    for (int j = 0; j < K; j++) {
        float min_dist = std::numeric_limits<float>::max();
        tableint closest_point = -1;
        int cnt = 0;

        search2Layer(centroids + j * dim,
                     closest_point,
                     maxlevel_,
                     level,
                     cnt);

        closest_points[j] = closest_point;
    }

    for (int j = 0; j < K; j++) {
        std::queue<tableint> bfs_queue;
        std::unordered_set<tableint> visited;
        bfs_queue.push(closest_points[j]);
        visited.insert(closest_points[j]);

        while (!bfs_queue.empty() && clusters[j].size() < T) {
            tableint current = bfs_queue.front();
            bfs_queue.pop();

            clusters[j].push_back(current);

            linklistsizeint *ll_cur = get_linklist_at_level(current, level);
            size_t linklistCount = getListCount(ll_cur);
            tableint *data = (tableint *)(ll_cur + 1);
            for (size_t iter = 0; iter < linklistCount; iter++) {
                int neighbor = data[iter];
                if (visited.find(neighbor) == visited.end() && clusters[j].size() < T && point_selection_count[neighbor] < l) {
                    bfs_queue.push(neighbor);
                    visited.insert(neighbor);
                }
                if (clusters[j].size() >= T)
                    break;
            }
        }
    }
}

template <typename dist_t>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
HierarchicalNSW<dist_t>::searchBaseLayerSubgroup(tableint ep_id, const void *data_point, int layer, std::unordered_set<tableint> *cluster_set, bool fromTopLayer = false) {
    tableint currObj = enterpoint_node_;
    if (fromTopLayer) {
        dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;
                std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                data = get_linklist(currObj, level);
                int size = getListCount(data);

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        ep_id = currObj;
    }

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidateSet.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;
    int set = 0;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
        if (layer == 0) {
            data = (int *)get_linklist0(curNodeNum);
        } else {
            data = (int *)get_linklist(curNodeNum, layer);
            //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
        }
        size_t size = getListCount((linklistsizeint *)data);
        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = *(datal + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif

            if ((cluster_set && cluster_set->find(candidate_id) == cluster_set->end()) || visited_array[candidate_id] == visited_array_tag)
            // if (visited_array[candidate_id] == visited_array_tag)
            {
                // if (cluster_set.find(candidate_id) == cluster_set.end())
                //     printf("candidate_id: %d, visited_array[candidate_id]: %d, visited_array_tag: %d\n", candidate_id, visited_array[candidate_id], visited_array_tag);
                continue;
            }
            // if (set == 0)
            // {
            //     set++;
            //     printf("p");
            // }
            visited_array[candidate_id] = visited_array_tag;
            char *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
            if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::mutuallyConnectNewElementSubgroup(
    tableint cur_c,
    int offset,
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    int level,
    bool isUpdate) {
    size_t Mcurmax = level ? maxM_ : maxM0_;

    getNeighborsByHeuristic2(top_candidates, M_, false);

    if (top_candidates.size() > M_)
        throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    std::vector<dist_t> selectedNeighborsDist;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selectedNeighbors.push_back(top_candidates.top().second);
        selectedNeighborsDist.push_back(top_candidates.top().first);
        top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
        // lock only during the update
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
        if (isUpdate) {
            lock.lock();
        }
        linklistsizeint *ll_cur;
        if (level == 0)
            ll_cur = get_linklist0(cur_c);
        else
            ll_cur = get_linklist(cur_c, level);

        if (*ll_cur && !isUpdate) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        setListCount(ll_cur, selectedNeighbors.size());
        tableint *data = (tableint *)(ll_cur + 1);
        dist_t *distData = (dist_t *)get_dist_at_level(cur_c, level);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            if (data[idx] && !isUpdate)
                throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            data[idx] = selectedNeighbors[idx];
            distData[idx] = selectedNeighborsDist[idx];
        }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        if (selectedNeighbors[idx] >= offset)
            continue;
        std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

        linklistsizeint *ll_other;
        if (level == 0)
            ll_other = get_linklist0(selectedNeighbors[idx]);
        else
            ll_other = get_linklist(selectedNeighbors[idx], level);

        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbors[idx] == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbors[idx]])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        tableint *data = (tableint *)(ll_other + 1);
        dist_t *distData = (dist_t *)get_dist_at_level(selectedNeighbors[idx], level);

        bool is_cur_c_present = false;
        if (isUpdate) {
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
                }
            }
        }

        // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
        if (!is_cur_c_present) {
            if (sz_link_list_other < Mcurmax) {
                data[sz_link_list_other] = cur_c;
                distData[sz_link_list_other] = selectedNeighborsDist[idx];
                setListCount(ll_other, sz_link_list_other + 1);
            } else {
                // finding the "weakest" element to replace it with the new one
                dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                            dist_func_param_);
                // Heuristic:
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(
                        fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                     dist_func_param_),
                        data[j]);
                }

                getNeighborsByHeuristic2(candidates, Mcurmax);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    distData[indx] = candidates.top().first;
                    candidates.pop();
                    indx++;
                }

                setListCount(ll_other, indx);
                // Nearest K:
                /*int indx = -1;
                for (int j = 0; j < sz_link_list_other; j++) {
                    dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                    if (d > d_max) {
                        indx = j;
                        d_max = d;
                    }
                }
                if (indx >= 0) {
                    data[indx] = cur_c;
                } */
            }
        }
    }

    // return next_closest_entry_point;
}

//         template <typename dist_t>
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
//         HierarchicalNSW<dist_t>::searchBaseLayer(std::vector<tableint> ep_ids, const void *data_point, int layer, int count_bound = 5)
//         {
//             VisitedList *vl = visited_list_pool_->getFreeVisitedList();
//             vl_type *visited_array = vl->mass;
//             vl_type visited_array_tag = vl->curV;

//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

//             for (int i = 0; i < ep_ids.size(); i++)
//             {
//                 tableint ep_id = ep_ids[i];
//                 if (!isMarkedDeleted(ep_id))
//                 {
//                     dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
//                     top_candidates.emplace(dist, ep_id);
//                     candidateSet.emplace(-dist, ep_id);
//                 }
//                 else
//                 {
//                     candidateSet.emplace(-std::numeric_limits<dist_t>::max(), ep_id);
//                 }
//                 visited_array[ep_id] = visited_array_tag;
//             }
//             dist_t lowerBound = -candidateSet.top().first;

//             while (!candidateSet.empty())
//             {
//                 std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
//                 if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == count_bound)
//                 {
//                     break;
//                 }
//                 candidateSet.pop();

//                 tableint curNodeNum = curr_el_pair.second;

//                 std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

//                 int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
//                 if (layer == 0)
//                 {
//                     data = (int *)get_linklist0(curNodeNum);
//                 }
//                 else
//                 {
//                     data = (int *)get_linklist(curNodeNum, layer);
//                     //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
//                 }
//                 size_t size = getListCount((linklistsizeint *)data);
//                 tableint *datal = (tableint *)(data + 1);
// #ifdef USE_SSE
//                 _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
//                 _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
//                 _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
//                 _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
// #endif

//                 for (size_t j = 0; j < size; j++)
//                 {
//                     tableint candidate_id = *(datal + j);
// //                    if (candidate_id == 0) continue;
// #ifdef USE_SSE
//                     _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
//                     _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
// #endif
//                     if (visited_array[candidate_id] == visited_array_tag)
//                         continue;
//                     visited_array[candidate_id] = visited_array_tag;
//                     char *currObj1 = (getDataByInternalId(candidate_id));

//                     dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
//                     if (top_candidates.size() < count_bound || lowerBound > dist1)
//                     {
//                         candidateSet.emplace(-dist1, candidate_id);
// #ifdef USE_SSE
//                         _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
// #endif

//                         if (!isMarkedDeleted(candidate_id))
//                             top_candidates.emplace(dist1, candidate_id);

//                         if (top_candidates.size() > count_bound)
//                             top_candidates.pop();

//                         if (!top_candidates.empty())
//                             lowerBound = top_candidates.top().first;
//                     }
//                 }
//             }
//             visited_list_pool_->releaseVisitedList(vl);

//             return top_candidates;
//         }

// template <typename dist_t>
// void HierarchicalNSW<dist_t>:: MergeIndex1OnIndex2(
//     HierarchicalNSW<dist_t> *index1,
//     HierarchicalNSW<dist_t> *index2,
//     int upper_level,
//     int offset_index1,
//     int offset_index2)

// {
//     tableint headObj = index1->enterpoint_node_;
//     std::vector<tableint> ep_ids_entrypoint_index1;
//     ep_ids_entrypoint_index1.push_back(index2->enterpoint_node_);
//     for (int level = upper_level; level >= 0; level -= 1)
//     {

//         std::queue<std::pair<tableint, std::vector<tableint>>> q;
//         std::unordered_set<tableint> seen_candidates;
//         q.push({headObj, ep_ids_entrypoint_index1});
//         size_t Mcurmax = level ? maxM_ : maxM0_;

//         while (!q.empty())
//         {
//             auto [cur_c, eps] = q.front();
//             q.pop();
//             if (seen_candidates.find(cur_c) != seen_candidates.end())
//                 continue;
//             seen_candidates.insert(cur_c);

//             const void *data_point = index1->getDataByInternalId(cur_c);

//             // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = index2->searchBaseLayer(eps, data_point, level, 3);
//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

//             index2->search2Layer(data_point,
//                 eps[0],
//                 level,
//                 level,
//                 3,
//                 top_candidates,
//                 0);
//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> pq_copy = top_candidates;

//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candi;

//             std::vector<tableint> new_ep_ids;
//             while (!pq_copy.empty())
//             {
//                 new_ep_ids.push_back(pq_copy.top().second);
//                 candi.push({pq_copy.top().first, pq_copy.top().second + offset_index2});
//                 pq_copy.pop();
//             }
//             if (cur_c == headObj)
//             {
//                 ep_ids_entrypoint_index1.assign(new_ep_ids.end() - 1, new_ep_ids.end());
//             }
//             linklistsizeint *ll_cur = index1->get_linklist_at_level(cur_c, level);
//             size_t linklistCount = index1->getListCount(ll_cur);
//             tableint *data = (tableint *)(ll_cur + 1);
//             dist_t *dist = index1->get_dist_at_level(cur_c, level);
//             for (size_t i = 0; i < linklistCount; i++)
//             {
//                 q.push({data[i], new_ep_ids});
//             }
//             top_candidates = candi;
//             // if (epDeleted)
//             // {
//             //     top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
//             //     if (top_candidates.size() > ef_construction_)
//             //         top_candidates.pop();
//             // }
//             if (top_candidates.size() + index1->getListCount(index1->get_linklist_at_level(cur_c, level)) > Mcurmax)
//             {
//                 dist_t lowerBound = top_candidates.top().first;
//                 for (size_t iter = 0; iter < linklistCount; iter++)
//                 {
//                     tableint candidate_id = data[iter] + offset_index1;
//                     dist_t dist1 = dist[iter];
//                     if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
//                     {
//                         top_candidates.emplace(dist1, candidate_id);
//                         if (top_candidates.size() > ef_construction_)
//                             top_candidates.pop();
//                         if (!top_candidates.empty())
//                             lowerBound = top_candidates.top().first;
//                     }
//                 }
//                 getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
//                 ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
//                 setListCount(ll_cur, top_candidates.size());
//                 data = (tableint *)(ll_cur + 1);
//                 dist_t *distData = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
//                 for (size_t idx = 0; top_candidates.size() > 0; idx++)
//                 {
//                     data[idx] = top_candidates.top().second;
//                     distData[idx] = top_candidates.top().first;
//                     top_candidates.pop();
//                 }
//             }
//             else
//             {
//                 linklistsizeint *ll_cur_alg_hnsw = get_linklist_at_level(cur_c + offset_index1, level);
//                 setListCount(ll_cur_alg_hnsw, top_candidates.size() + linklistCount);
//                 tableint *data_alg_hnsw = (tableint *)(ll_cur_alg_hnsw + 1);
//                 dist_t *distData_alg_hnsw = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
//                 size_t offset = top_candidates.size();
//                 for (size_t idx = 0; top_candidates.size() > 0; idx++)
//                 {
//                     data_alg_hnsw[idx] = top_candidates.top().second;
//                     distData_alg_hnsw[idx] = top_candidates.top().first;
//                     top_candidates.pop();
//                 }
//                 for (size_t idx = 0; idx < linklistCount; idx++)
//                 {
//                     data_alg_hnsw[offset + idx] = data[idx] + offset_index1;
//                     distData_alg_hnsw[offset + idx] = dist[idx];
//                 }
//             }
//         }
//     }
// }

// template <typename dist_t>
// void HierarchicalNSW<dist_t>:: HNSWMerger(
//     HierarchicalNSW<dist_t> *index1,
//     HierarchicalNSW<dist_t> *index2,
//     int method_flag)
// {
//     if (index1->maxlevel_ < index2->maxlevel_ || (index1->maxlevel_ == index2->maxlevel_ && index1->cur_element_count < index2->cur_element_count))
//     {
//         std::swap(index1, index2);
//     }
//     size_t maxLevel = std::max(index1->maxlevel_, index2->maxlevel_);
//     setMaxLevel(maxLevel);
//     cur_element_count.store(index1->cur_element_count + index2->cur_element_count);
//     enterpoint_node_ = index1->enterpoint_node_;

//     size_t element_count_for_index1 = index1->getCurrentElementCount();
//     size_t element_count_for_index2 = index2->getCurrentElementCount();
//     std::vector<std::vector<int>> layer_node_for_index1(maxlevel_ + 2);
//     index1->searchNodeOnEachLayer(layer_node_for_index1, true);
//     // std::vector<std::vector<int>> layer_node_for_index2(maxlevel_ + 2);
//     // index2->searchNodeOnEachLayer(layer_node_for_index2, true);
//     int dim = data_size_ / sizeof(dist_t);

//     auto allocateMemory = [&](HierarchicalNSW<dist_t> *index, size_t element_count_for_index, size_t offset)
//     {
//         for (tableint old_c = 0; old_c < element_count_for_index; old_c++)
//         {
//             tableint new_c = old_c + offset;
//             int level = index->element_levels_[old_c];
//             memcpy(getDataByInternalId(new_c), index->getDataByInternalId(old_c), data_size_);
//             setExternalLabel(new_c, index->getExternalLabel(old_c));
//             element_levels_[new_c] = level;

//             if (level == 0)
//                 continue;
//             linkLists_[new_c] = (char *)malloc(size_links_per_element_ * level + 1);
//             if (linkLists_[new_c] == nullptr)
//             {
//                 throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
//             }
//             memset(linkLists_[new_c], 0, size_links_per_element_ * level + 1);
//             dist_linkLists_[new_c] = (char *)malloc(size_dist_links_per_element_ * level + 1);
//             if (dist_linkLists_[new_c] == nullptr)
//             {
//                 throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linklist");
//             }
//             memset(dist_linkLists_[new_c], 0, size_dist_links_per_element_ * level + 1);
//         }
//     };
//     allocateMemory(index1, element_count_for_index1, 0);
//     allocateMemory(index2, element_count_for_index2, element_count_for_index1);

//     // TODO: work for (1) do a fully-layer search for index1->entrypoint_node_ on index2, collect all candidate nodes; (2) on each layer, based on the thought of BFS, search index1->entrypoint_node_'s neighbor currObj based on those candidate nodes, then use its candidate nodes as new candidate set for currObj's neighbor; (3) do all layers searching on index2; (4) do same thing for index2->entrypoint_node_ on index1

//     // std::vector<std::vector<int>> candidate_node_for_entrypoint_index1(maxlevel_ + 2);
//     tableint currObj = index2->enterpoint_node_;
//     const void *data_point = index1->getDataByInternalId(index1->enterpoint_node_);
//     // TODO: deepcopy for one layer.
//     int upper_level = maxlevel_;
//     for (; upper_level >= 0; upper_level--)
//     {
//         if (upper_level > index2->maxlevel_)
//         {
//             deepCopyOneLayerOnIndex(index1, this, upper_level, 0, layer_node_for_index1);
//             layer_node_for_index1[upper_level - 1].insert(layer_node_for_index1[upper_level - 1].end(), layer_node_for_index1[upper_level].begin(), layer_node_for_index1[upper_level].end());
//         }
//         else
//             break;
//     }
//     MergeIndex1OnIndex2(index1, index2, upper_level, 0, element_count_for_index1);
//     MergeIndex1OnIndex2(index2, index1, upper_level, element_count_for_index1, 0);
// }

// template <typename dist_t>
// void HierarchicalNSW<dist_t>:: mergeIndex1BasedOnIndex2Connection(
//     HierarchicalNSW<dist_t> *index1,
//     HierarchicalNSW<dist_t> *index2,
//     tableint cur_c,
//     char *data_point,
//     int offset_index1,
//     int offset_index2,
//     int level,
//     tableint &last_entry_point,
//     bool debugFlag = false)
// {
//     int higherLevel = level;
//     if (last_entry_point == -1)
//     {
//         higherLevel = index2->maxlevel_;
//     }
//     std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

//     linklistsizeint *ll_cur;
//     size_t linklistCount;
//     tableint *data;
//     dist_t *dist;
//     tableint candidate_id;
//     dist_t dist1;
//     dist_t lowerBound;
//     size_t Mcurmax = level ? maxM_ : maxM0_;

//     int method_flag = 0b10;

//     tableint xx = last_entry_point;

//     if (method_flag % 2)
//     {
//         int cnt = 0;
//         // if(cur_c==4373 && level == 2){
//         //     cnt=1;
//         // }
//         index2->search2Layer(data_point,
//                              last_entry_point,
//                              higherLevel,
//                              level,
//                              cnt);
//         forward_step_avg_per_layer[level] += cnt;
//         candidate_id = last_entry_point + offset_index2;
//         dist1 = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
//         top_candidates.emplace(dist1, candidate_id);

//         /*
//             adding neighbors will increase recall less than 0.005
//         */

//         // lowerBound = dist1;
//         // if (debugFlag)
//         // {
//         //     layeredStructure[level].insert({cur_c + offset_index1, dist1});
//         // }

//         // ll_cur = index2->get_linklist_at_level(last_entry_point, level);
//         // linklistCount = index2->getListCount(ll_cur);
//         // data = (tableint *)(ll_cur + 1);

//         // for (size_t iter = 0; iter < linklistCount; iter++)
//         // {
//         //     candidate_id = data[iter] + offset_index2;
//         //     dist1 = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
//         //     if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
//         //     {
//         //         top_candidates.emplace(dist1, candidate_id);
//         //         if (top_candidates.size() > ef_construction_)
//         //             top_candidates.pop();
//         //         if (!top_candidates.empty())
//         //             lowerBound = top_candidates.top().first;
//         //     }
//         // }

//         /*
//             comment this may cause recall raise <0.01, and search time per 10000 queries raise 1.8s
//         */
//         // getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
//     }
//     if (((int)method_flag / 2) % 2)
//     // if (true)
//     {
//         int cnt = 2;
//         // if(cur_c==4373 && level == 2){
//         //     cnt=1;
//         // }
//         // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates2;

//         index2->search2Layer(data_point,
//                              xx,
//                              higherLevel,
//                              level,
//                              cnt,
//                              top_candidates,
//                              offset_index2);
//         // if (xx != last_entry_point)
//         // {
//         //     printf("level %d, cur_c %d, last_entry_point %d, xx %d\n", level, cur_c, last_entry_point, xx);
//         // }
//         /* comment this may cause recall raise <0.01, and search time per 10000 queries raise 1.8s */
//         // getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
//     }

//     if (top_candidates.size() + index1->getListCount(index1->get_linklist_at_level(cur_c, level)) > Mcurmax)
//     {
//         ll_cur = index1->get_linklist_at_level(cur_c, level);
//         linklistCount = index1->getListCount(ll_cur);
//         data = (tableint *)(ll_cur + 1);
//         dist = index1->get_dist_at_level(cur_c, level);
//         for (size_t iter = 0; iter < linklistCount; iter++)
//         {
//             candidate_id = data[iter] + offset_index1;
//             dist1 = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
//             if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
//             {
//                 top_candidates.emplace(dist1, candidate_id);
//                 if (top_candidates.size() > ef_construction_)
//                     top_candidates.pop();
//                 if (!top_candidates.empty())
//                     lowerBound = top_candidates.top().first;
//             }
//         }
//         getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
//         ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
//         setListCount(ll_cur, top_candidates.size());
//         data = (tableint *)(ll_cur + 1);
//         dist_t *distData = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
//         for (size_t idx = 0; top_candidates.size() > 0; idx++)
//         {
//             data[idx] = top_candidates.top().second;
//             distData[idx] = top_candidates.top().first;
//             top_candidates.pop();
//         }
//     }
//     else
//     {
//         linklistsizeint *ll_cur_index1 = index1->get_linklist_at_level(cur_c, level);
//         size_t linklistCount_index1 = index1->getListCount(ll_cur_index1);
//         tableint *data_index1 = (tableint *)(ll_cur_index1 + 1);
//         dist_t *dist_index1 = index1->get_dist_at_level(cur_c, level);

//         ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
//         setListCount(ll_cur, top_candidates.size() + linklistCount_index1);
//         data = (tableint *)(ll_cur + 1);
//         dist_t *distData = (dist_t *)get_dist_at_level(cur_c + offset_index1, level);
//         size_t offset = top_candidates.size();
//         for (size_t idx = 0; top_candidates.size() > 0; idx++)
//         {
//             data[idx] = top_candidates.top().second;
//             distData[idx] = top_candidates.top().first;
//             top_candidates.pop();
//         }
//         for (size_t idx = 0; idx < linklistCount_index1; idx++)
//         {
//             data[offset + idx] = data_index1[idx] + offset_index1;
//             distData[offset + idx] = dist_index1[idx];
//         }
//     }
// }

template <typename dist_t>
void HierarchicalNSW<dist_t>::findNeighborsLayerK(hnswlib::HierarchicalNSW<dist_t> *index, int currObj, int cur_c, int level, tableint &candidate_entrypoint) {
    const void *data_point = getDataByInternalId(cur_c);
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, hnswlib::HierarchicalNSW<dist_t>::CompareByFirst> top_candidates = index->searchBaseLayer(
        currObj, data_point, level);

    size_t Mcurmax = level ? maxM_ : maxM0_;
    getNeighborsByHeuristic2(top_candidates, Mcurmax, false, cur_c);

    // linklistsizeint *ll_cur = get_linklist(currObj, level);
    // tableint *data = (tableint *)(ll_cur + 1);
    // dist_t *distData = (dist_t *)get_dist_at_level(cur_c, level);
    int cnt = 0;
    for (size_t idx = 0; top_candidates.size() > 0; idx++) {
        if (top_candidates.top().second == cur_c) {
            // data[idx] = top_candidates.top().second;
            // distData[idx] = top_candidates.top().first;
            // cnt++;
            printf("distData[%d]: %f\n", idx, top_candidates.top().first);
        }
        top_candidates.pop();
    }
    // setListCount(ll_cur, cnt);
}


// template <typename dist_t>
// HierarchicalNSW<dist_t> *HNSWRefinement(HierarchicalNSW<dist_t> *index, L2Space *space) {
//     HierarchicalNSW<dist_t> *alg_hnsw = new HierarchicalNSW<dist_t>(space, index->max_elements_, index->M_, index->ef_construction_);
//     alg_hnsw->setMaxLevel(index->maxlevel_);
//     alg_hnsw->cur_element_count.store(index->cur_element_count);

//     size_t num_points = index->cur_element_count;
//     int K = index->M_;
//     int max_iter = 1;

//     std::vector<std::vector<tableint>> out_edges(num_points);
//     std::vector<std::vector<tableint>> in_edges(num_points);

//     double t0;
//     t0 = elapsed();
//     // Step 1: 初始化 layer 0 邻接表
//     for (tableint i = 0; i < num_points; i++) {
//         linklistsizeint *linklist = index->get_linklist_at_level(i, 0);
//         int link_count = index->getListCount(linklist);
//         tableint *data = (tableint *)(linklist + 1);
//         for (int j = 0; j < link_count; j++) {
//             tableint neighbor = data[j];
//             out_edges[i].push_back(neighbor);
//             in_edges[neighbor].push_back(i);
//         }
//     }
//     printf("[%.3f s] step 1\n", elapsed() - t0);
//     std::vector<char> updated(num_points);

//     // 假设你已经包含了 HierarchicalNSW 相关定义和 typedefs

//     std::vector<std::vector<tableint>> old_edges = out_edges;
//     std::vector<std::vector<tableint>> new_edges(num_points);
//     std::vector<std::mutex> locks(num_points); // 替代 omp_lock_t，线程安全 push_back

//     auto try_insert_pair = [&](tableint u, tableint v) {
//         if (u == v)
//             return;

//         const void *vec_u = index->getDataByInternalId(u);
//         const void *vec_v = index->getDataByInternalId(v);
//         dist_t d = index->fstdistfunc_(vec_u, vec_v, index->dist_func_param_);

//         auto try_update = [&](tableint src, tableint tgt, dist_t dval) {
//             std::lock_guard<std::mutex> guard(locks[src]);
//             auto &neigh = out_edges[src];
//             for (auto n : neigh)
//                 if (n == tgt)
//                     return;

//             if ((int)neigh.size() < K) {
//                 neigh.push_back(tgt);
//             } else {
//                 int worst_i = -1;
//                 dist_t worst_d = -1;
//                 for (int i = 0; i < (int)neigh.size(); i++) {
//                     const void *vec_w = index->getDataByInternalId(neigh[i]);
//                     dist_t dw = index->fstdistfunc_(vec_w, vec_u, index->dist_func_param_);
//                     if (dw > worst_d) {
//                         worst_d = dw;
//                         worst_i = i;
//                     }
//                 }
//                 if (dval < worst_d && worst_i >= 0) {
//                     neigh[worst_i] = tgt;
//                 }
//             }
//         };

//         try_update(u, v, d);
//         try_update(v, u, d);
//     };

//     for (int iter = 0; iter < max_iter; iter++) {
//         t0 = elapsed();
//         std::atomic<int> updates(0);

//         // swap new/old
//         old_edges = out_edges;
//         new_edges.clear();
//         new_edges.resize(num_points);

// // 标记哪些点是新邻居（KGraph中的 "new" 集合）
// #pragma omp parallel for schedule(dynamic)
//         for (tableint i = 0; i < num_points; i++) {
//             std::unordered_set<tableint> all;
//             for (auto u : out_edges[i])
//                 all.insert(u);
//             for (auto u : old_edges[i])
//                 all.insert(u);
//             std::vector<tableint> unique(all.begin(), all.end());
//             std::shuffle(unique.begin(), unique.end(), std::mt19937(i));

//             for (size_t a = 0; a < unique.size(); ++a) {
//                 for (size_t b = a + 1; b < unique.size(); ++b) {
//                     try_insert_pair(unique[a], unique[b]);
//                 }
//             }
//         }

//         printf("[%.3f s] NNDescent iter %d\n", elapsed() - t0, iter + 1);
//     }

//     // Step 3: 更新索引
//     t0 = elapsed();
//     alg_hnsw->data_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
//     if (alg_hnsw->data_level0_memory_ == nullptr)
//         throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

//     alg_hnsw->dist_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_dist_per_element_);
//     if (alg_hnsw->dist_level0_memory_ == nullptr)
//         throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

//     alg_hnsw->enterpoint_node_ = index->enterpoint_node_;

//     for (int cur_c = 0; cur_c < alg_hnsw->cur_element_count; cur_c++) {
//         auto level = alg_hnsw->element_levels_[cur_c] = index->element_levels_[cur_c];
//         if (level > 0) {
//             alg_hnsw->linkLists_[cur_c] = (char *)malloc(alg_hnsw->size_links_per_element_ * level + 1);
//             if (alg_hnsw->linkLists_[cur_c] == nullptr) {
//                 throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
//             }
//             memset(alg_hnsw->linkLists_[cur_c], 0, alg_hnsw->size_links_per_element_ * level + 1);

//             alg_hnsw->dist_linkLists_[cur_c] = (char *)malloc(alg_hnsw->size_dist_links_per_element_ * level + 1);
//             if (alg_hnsw->dist_linkLists_[cur_c] == nullptr) {
//                 throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linklist");
//             }
//             memset(alg_hnsw->dist_linkLists_[cur_c], 0, alg_hnsw->size_dist_links_per_element_ * level + 1);
//             alg_hnsw->element_levels_[cur_c] = level;
//         }
//         memcpy(alg_hnsw->getDataByInternalId(cur_c), index->getDataByInternalId(cur_c), alg_hnsw->data_size_);
//         alg_hnsw->setExternalLabel(cur_c, index->getExternalLabel(cur_c));
//     }
//     printf("[%.3f s] step 3-1\n", elapsed() - t0);

//     t0 = elapsed();
//     int maxM = alg_hnsw->maxM0_;
//     for (tableint i = 0; i < num_points; i++) {
//         // 1. 构建 max-heap 的 top_candidates（距离小的优先）
//         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;

//         const void *vec_i = index->getDataByInternalId(i);
//         for (tableint neighbor : out_edges[i]) {
//             const void *vec_n = index->getDataByInternalId(neighbor);
//             dist_t d = index->fstdistfunc_(vec_i, vec_n, index->dist_func_param_);
//             top_candidates.emplace(d, neighbor);
//         }

//         // 2. 执行剪枝：注意 cur_c = i
//         alg_hnsw->getNeighborsByHeuristic2(top_candidates, maxM);

//         // 3. 写入 linklist (neighbor + dist) 到 alg_hnsw 的 layer 0
//         linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(i, 0);
//         tableint *data = (tableint *)(ll_cur + 1);
//         dist_t *distData = (dist_t *)alg_hnsw->get_dist_at_level(i, 0);

//         size_t sz = top_candidates.size();
//         *ll_cur = (linklistsizeint)sz;

//         for (size_t idx = 0; top_candidates.size() > 0; idx++) {
//             data[idx] = top_candidates.top().second;
//             distData[idx] = top_candidates.top().first;
//             top_candidates.pop();
//         }
//     }
//     printf("[%.3f s] step 3-2\n", elapsed() - t0);

//     return alg_hnsw;
// }

// #pragma omp parallel for
//   for (int cur_c = 0; cur_c < alg_hnsw->cur_element_count; cur_c++) {
//     auto level = alg_hnsw->element_levels_[cur_c];
//     const void *data_point = index->getDataByInternalId(cur_c);
//     int currObj = index->enterpoint_node_;
//     for (int cur_level = level; cur_level >= 0; cur_level--) {
//       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
//                           hnswlib::HierarchicalNSW<float>::CompareByFirst>
//           top_candidates = index->searchBaseLayer(currObj, data_point, cur_level);

//       size_t Mcurmax = cur_level ? index->maxM_ : index->maxM0_;
//       index->getNeighborsByHeuristic2(top_candidates, Mcurmax, false, cur_c);

//       linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(cur_c, cur_level);
//       tableint *data = (tableint *)(ll_cur + 1);
//       dist_t *distData = (dist_t *)alg_hnsw->get_dist_at_level(cur_c, cur_level);
//       int cnt = 0;
//       for (size_t idx = 0; top_candidates.size() > 0; idx++) {
//         // printf("id: %d, level: %d", cur_c, cur_level);
//         if (top_candidates.top().second != cur_c) {
//           data[idx] = top_candidates.top().second;
//           distData[idx] = top_candidates.top().first;
//           cnt++;
//           currObj = top_candidates.top().second;
//           // printf("[neighbor: %d, dist: %f], ", data[idx], distData[idx]);
//         }
//         top_candidates.pop();
//       }
//       alg_hnsw->setListCount(ll_cur, cnt);
//     }
//   }

// for (int level = maxLevel; level >= 0; level -= 1)
// {
//     if (level > 0)
//     {
//         auto allocateMemory = [&](std::vector<std::vector<int>>
//         &layer_node_for_index)
//         {
//             for (int id = 0; id < layer_node_for_index[level].size(); id++)
//             {
//                 tableint new_c = layer_node_for_index[level][id];
//                 alg_hnsw->linkLists_[new_c] = (char
//                 *)malloc(alg_hnsw->size_links_per_element_ * level + 1); if
//                 (alg_hnsw->linkLists_[new_c] == nullptr)
//                 {
//                     throw std::runtime_error("Not enough memory: addPoint
//                     failed to allocate linklist");
//                 }
//                 memset(alg_hnsw->linkLists_[new_c], 0,
//                 alg_hnsw->size_links_per_element_ * level + 1);
//                 alg_hnsw->dist_linkLists_[new_c] = (char
//                 *)malloc(alg_hnsw->size_dist_links_per_element_ * level +
//                 1); if (alg_hnsw->dist_linkLists_[new_c] == nullptr)
//                 {
//                     throw std::runtime_error("Not enough memory: addPoint
//                     failed to allocate dist_linklist");
//                 }
//                 memset(alg_hnsw->dist_linkLists_[new_c], 0,
//                 alg_hnsw->size_dist_links_per_element_ * level + 1);
//                 alg_hnsw->element_levels_[new_c] = level;
//             }
//         };
//         allocateMemory(layer_node_for_index);
//     }

//     for (int id = 0; id < layer_node_for_index[level].size(); id++)
//     {
//         tableint new_c = layer_node_for_index[level][id];
//         memcpy(alg_hnsw->getDataByInternalId(new_c),
//         index1->getDataByInternalId(new_c), alg_hnsw->data_size_);
//         alg_hnsw->setExternalLabel(new_c, index->getExternalLabel(new_c));
//         mergedDataPointsFromTopIndex.push_back(new_c);
//     }

//     for (int iter = 0; iter < mergedDataPointsFromTopIndex.size(); iter++)
//     {
//         tableint cur_c = mergedDataPointsFromTopIndex1[iter];
//         char *data_point = index->getDataByInternalId(cur_c);
//         alg_hnsw->mergeIndex1BasedOnIndex2Connection(index1,
//                                                      index2,
//                                                      cur_c,
//                                                      data_point,
//                                                      0,
//                                                      element_count_for_index1,
//                                                      level,
//                                                      entry_point_collect_index1_on_index2[cur_c]);
//     }
// }

//   return alg_hnsw;
// }

} // namespace hnswlib