#include "hnswalg.h"
#include <mutex>
#include <omp.h>
#include <random>

using namespace std;
namespace hnswlib {

template <typename dist_t>
void HierarchicalNSW<dist_t>::searchNodeOnEachLayer(
    std::vector<std::vector<int>> &resultVector,
    bool combine) {
    size_t elementCount = getCurrentElementCount();
    for (size_t iter = 0; iter < elementCount; iter++) {
        if (!isMarkedDeleted(iter)) {
            resultVector[element_levels_[iter]].push_back(iter);
        }
    }
    if (combine) {
        for (int i = resultVector.size() - 2; i >= 0; --i) {
            resultVector[i].insert(resultVector[i].end(), resultVector[i + 1].begin(), resultVector[i + 1].end());
        }
    }
}

/*
 * This function, which is used in search2Layer, searching the graph from the
 * top layer to the level_low layer. If the enterpoint_node is -1, it will be
 * set to enterpoint_node_. Otherwise, the enterpoint_node will be used as the
 * starting point, as it's the closest point to the query on layer level_low
 * + 1. The function will also return the `top_candidates` which contains the
 * `cnt`-closest points to the query on layer level_low.
 */
template <typename dist_t>
void HierarchicalNSW<dist_t>::search2Layer(const void *query_data,
                                           tableint &enterpoint_node,
                                           int level_higher,
                                           int level_lower,
                                           int cnt,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> *top_candidates,
                                           int offset) {
    tableint currObj = enterpoint_node == -1 ? enterpoint_node_ : enterpoint_node;
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
    for (int level = level_higher; level > level_lower; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data = (unsigned int *)get_linklist_at_level(currObj, level);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);

            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }
    if (top_candidates == nullptr) {
        enterpoint_node = currObj;
        return;
    }
    // printf("method1, level %d, enterpoint_node: %d, currObj: %d\n",
    // level_lower, enterpoint_node, currObj);

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    // std::priority_queue<std::pair<dist_t, tableint>,
    // std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    dist_t lowerBound;
    dist_t entry_dist;
    if (!isMarkedDeleted(currObj)) {
        dist_t dist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        top_candidates->emplace(dist, currObj + offset);
        enterpoint_node = currObj;
        entry_dist = dist;
        lowerBound = dist;
        candidateSet.emplace(-dist, currObj);
    } else {
        entry_dist = std::numeric_limits<dist_t>::max();
        lowerBound = std::numeric_limits<dist_t>::max();
        candidateSet.emplace(-lowerBound, currObj);
    }
    visited_array[currObj] = visited_array_tag;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() == cnt) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        // std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data = (int *)get_linklist_at_level(curNodeNum, level_lower);
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
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;
            char *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
            if (top_candidates->size() < cnt || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id)) {
                    top_candidates->emplace(dist1, candidate_id + offset);
                    if (entry_dist > dist1) {
                        enterpoint_node = candidate_id;
                        entry_dist = dist1;
                    }
                }

                while (top_candidates->size() > cnt)
                    top_candidates->pop();

                if (!top_candidates->empty())
                    lowerBound = top_candidates->top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::search2Layer_MT(const void *query_data,
                                           tableint &enterpoint_node,
                                           int level_higher,
                                           int level_lower,
                                           int cnt,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> *top_candidates,
                                           int offset) {
    tableint currObj = enterpoint_node == -1 ? enterpoint_node_ : enterpoint_node;
    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
    for (int level = level_higher; level > level_lower; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data = (unsigned int *)get_linklist_at_level(currObj, level);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);

            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }
                dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }
    if (top_candidates == nullptr) {
        enterpoint_node = currObj;
        return;
    }
    // printf("method1, level %d, enterpoint_node: %d, currObj: %d\n",
    // level_lower, enterpoint_node, currObj);

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    // std::priority_queue<std::pair<dist_t, tableint>,
    // std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    dist_t lowerBound;
    dist_t entry_dist;
    if (!isMarkedDeleted(currObj)) {
        dist_t dist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        top_candidates->emplace(dist, currObj + offset);
        enterpoint_node = currObj;
        entry_dist = dist;
        lowerBound = dist;
        candidateSet.emplace(-dist, currObj);
    } else {
        entry_dist = std::numeric_limits<dist_t>::max();
        lowerBound = std::numeric_limits<dist_t>::max();
        candidateSet.emplace(-lowerBound, currObj);
    }
    visited_array[currObj] = visited_array_tag;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() == cnt) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        // std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data = (int *)get_linklist_at_level(curNodeNum, level_lower);
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
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;
            char *currObj1 = (getDataByInternalId(candidate_id));

            dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
            if (top_candidates->size() < cnt || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id)) {
                    top_candidates->emplace(dist1, candidate_id + offset);
                    if (entry_dist > dist1) {
                        enterpoint_node = candidate_id;
                        entry_dist = dist1;
                    }
                }

                while (top_candidates->size() > cnt)
                    top_candidates->pop();

                if (!top_candidates->empty())
                    lowerBound = top_candidates->top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
}

template <typename dist_t>
void HierarchicalNSW<dist_t>::mergeIndex1BasedOnIndex2Connection(HierarchicalNSW<dist_t> *index1,
                                                                 HierarchicalNSW<dist_t> *index2,
                                                                 tableint cur_c,
                                                                 char *data_point,
                                                                 int offset_index1,
                                                                 int offset_index2,
                                                                 int level,
                                                                 tableint &last_entry_point,
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

    int cnt = 6;
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
        getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
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

template <typename dist_t>
void HierarchicalNSW<dist_t>::deepCopyOneLayerOnIndex(HierarchicalNSW<dist_t> *index,
                                                      int level,
                                                      int offset,
                                                      std::vector<std::vector<int>> &layer_node_for_index) {
#pragma omp parallel for schedule(dynamic)
    for (int iter = 0; iter < layer_node_for_index[level].size(); iter++) {
        tableint cur_c = layer_node_for_index[level][iter];
        tableint new_c = cur_c + offset;

        memcpy(getExternalLabeLp(new_c), index->getExternalLabeLp(cur_c), sizeof(labeltype));
        memcpy(getDataByInternalId(new_c), index->getDataByInternalId(cur_c), data_size_);

        linklistsizeint *ll_cur = index->get_linklist(cur_c, level);
        linklistsizeint *ll_new = get_linklist(new_c, level);
        size_t linklistCount = index->getListCount(ll_cur);
        setListCount(ll_new, linklistCount);
        tableint *data_cur = (tableint *)(ll_cur + 1);
        tableint *data_new = (tableint *)(ll_new + 1);
        dist_t *ll_cur_dist = index->get_dist_at_level(cur_c, level);
        dist_t *ll_new_dist = get_dist_at_level(new_c, level);

        if (level == 0) {
            for (int i = 0; i < linklistCount; i++) {
                data_new[i] = data_cur[i] + offset;
            }
            memcpy(ll_new_dist, ll_cur_dist, index->size_dist_per_element_);
        } else {
            for (int i = 0; i < linklistCount; i++) {
                data_new[i] = data_cur[i] + offset;
            }
            memcpy(ll_new_dist, ll_cur_dist, index->size_dist_links_per_element_);
        }
    }
}

class StopW {
    std::chrono::steady_clock::time_point time_begin;

public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

template <typename dist_t>
HierarchicalNSW<dist_t> *HNSWMerger(HierarchicalNSW<dist_t> *index1, HierarchicalNSW<dist_t> *index2, L2Space *space, size_t M = -1, size_t ef_construction = -1) {
    // TODO: implement logic with deleted node in an index, by re-organize the
    // internal label of nodes in an index

    double s0 = elapsed();
    if (index1->cur_element_count > index2->cur_element_count || index1->maxlevel_ > index2->maxlevel_) {
        std::swap(index1, index2);
        printf("Swap index with more current element count to index1.\n");
    }

    size_t element_count_for_index1 = index1->getCurrentElementCount();
    size_t element_count_for_index2 = index2->getCurrentElementCount();
    size_t index1_offset = 0;
    size_t index2_offset = element_count_for_index1;

    M = M == -1 ? std::max(index1->M_, index2->M_) : M;
    ef_construction = ef_construction == -1 ? std::max(index1->ef_construction_, index2->ef_construction_) : ef_construction;

    // printf("M: %ld, ef_construction: %ld\n", M, ef_construction);
    size_t new_max_elements = index1->max_elements_ + index2->max_elements_;
    HierarchicalNSW<dist_t> *alg_hnsw = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    size_t maxLevel = std::max(index1->maxlevel_, index2->maxlevel_);
    alg_hnsw->setMaxLevel(maxLevel);
    alg_hnsw->cur_element_count.store(index1->cur_element_count + index2->cur_element_count);

    alg_hnsw->data_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
    if (alg_hnsw->data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

    alg_hnsw->dist_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_dist_per_element_);
    if (alg_hnsw->dist_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

    // First, view all elements in both indexes and find the element sets in each
    // layer.

    std::vector<std::vector<int>> layer_node_for_index1(maxLevel + 2);
    index1->searchNodeOnEachLayer(layer_node_for_index1);
    std::vector<std::vector<int>> layer_node_for_index2(maxLevel + 2);
    index2->searchNodeOnEachLayer(layer_node_for_index2);

    tableint *entry_point_collect_index1_on_index2 = new tableint[element_count_for_index1];
    // tableint *entry_point_collect_index2_on_index1 = new tableint[element_count_for_index2];
    memset(entry_point_collect_index1_on_index2, -1, element_count_for_index1 * sizeof(int));
    // memset(entry_point_collect_index2_on_index1, -1, element_count_for_index2 * sizeof(int));

    int entry_point_index1 = index2->enterpoint_node_;
    // int entry_point_index2 = index1->enterpoint_node_;

    std::vector<int> mergedDataPointsFromTopIndex1;
    std::vector<int> mergedDataPointsFromTopIndex2;

    /* create new ligher layer */
    if (layer_node_for_index1[maxLevel].size() + layer_node_for_index2[maxLevel].size() > M) {
        std::vector<int> newLayer;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        int cnt = 0;
        auto filter_and_remove = [&](std::vector<int> &layer_nodes, HierarchicalNSW<dist_t> *index, std::vector<int> &mergedDataPointsFromTopIndex, int offset) {
            auto it = layer_nodes.begin();
            while (it != layer_nodes.end()) {
                if (distribution(alg_hnsw->level_generator_) < 1.0 / M || cnt == M) {
                    newLayer.push_back((*it) + offset);
                    cnt = 0;
                    tableint new_c = *it;
                    alg_hnsw->linkLists_[new_c + offset] = (char *)malloc(alg_hnsw->size_links_per_element_ * (maxLevel + 1) + 1);
                    if (alg_hnsw->linkLists_[new_c + offset] == nullptr)
                        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                    memset(alg_hnsw->linkLists_[new_c + offset], 0, alg_hnsw->size_links_per_element_ * (maxLevel + 1) + 1);
                    alg_hnsw->element_levels_[new_c + offset] = (maxLevel + 1);

                    alg_hnsw->dist_linkLists_[new_c + offset] = (char *)malloc(alg_hnsw->size_dist_links_per_element_ * (maxLevel + 1) + 1);
                    if (alg_hnsw->dist_linkLists_[new_c + offset] == nullptr)
                        throw std::runtime_error("Not enough memory: addPoint failed to "
                                                 "allocate dist_linkLists");
                    memset(alg_hnsw->dist_linkLists_[new_c + offset], 0, alg_hnsw->size_dist_links_per_element_ * (maxLevel + 1) + 1);

                    memcpy(alg_hnsw->getDataByInternalId(new_c + offset), index->getDataByInternalId(new_c), alg_hnsw->data_size_);
                    alg_hnsw->setExternalLabel(new_c + offset, index->getExternalLabel(new_c));
                    mergedDataPointsFromTopIndex.push_back(new_c);

                    it = layer_nodes.erase(it);
                } else {
                    cnt++;
                    ++it;
                }
            }
        };
        filter_and_remove(layer_node_for_index1[maxLevel], index1, mergedDataPointsFromTopIndex1, index1_offset);
        filter_and_remove(layer_node_for_index2[maxLevel], index2, mergedDataPointsFromTopIndex2, index2_offset);
        if (newLayer.size() > 0) { /* construct graph for new layer */
            for (int newLayerIter = 0; newLayerIter < newLayer.size(); newLayerIter++) {
                linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(newLayer[newLayerIter], maxLevel + 1);
                alg_hnsw->setListCount(ll_cur, newLayer.size() - 1);
                tableint *data = (tableint *)(ll_cur + 1);
                dist_t *dist = alg_hnsw->get_dist_at_level(newLayer[newLayerIter], maxLevel + 1);
                for (size_t it = 0, idx = 0; it < newLayer.size(); it++) {
                    if (it == newLayerIter)
                        continue;
                    dist_t dist0 =
                        alg_hnsw->fstdistfunc_(alg_hnsw->getDataByInternalId(newLayer[newLayerIter]), alg_hnsw->getDataByInternalId(newLayer[it]), alg_hnsw->dist_func_param_);
                    data[idx] = newLayer[it];
                    dist[idx] = dist0;
                    idx++;
                }
            }
            alg_hnsw->setMaxLevel(maxLevel + 1);
            alg_hnsw->enterpoint_node_ = newLayer[0];
        } else {
            alg_hnsw->enterpoint_node_ = layer_node_for_index2[maxLevel][0] + index2_offset;
        }
        // printf("new maxLevel: %d\n", maxLevel);
    } /* end of new code */
    else {
        alg_hnsw->enterpoint_node_ = layer_node_for_index2[maxLevel][0] + index2_offset;
    }

    // element_levels_ cannot use memcpy because of the order of element compared with external label.
    // memcpy(alg_hnsw->element_levels_.data(), index1->element_levels_.data(), sizeof(int) * element_count_for_index1);
    // memcpy(alg_hnsw->element_levels_.data() + sizeof(int) * element_count_for_index1, index2->element_levels_.data(), sizeof(int) * element_count_for_index2);

    alg_hnsw->label_lookup_.clear();
    alg_hnsw->label_lookup_.reserve(alg_hnsw->max_elements_);

    for (int id = 0; id < element_count_for_index1; id++) {
        alg_hnsw->label_lookup_[index1->getExternalLabel(id)] = id + index1_offset;
    }
    for (int id = 0; id < element_count_for_index2; id++) {
        alg_hnsw->label_lookup_[index2->getExternalLabel(id)] = id + index2_offset;
    }

    // TODO: fix increased-layer nodes
    auto allocateMemory = [&](hnswlib::HierarchicalNSW<dist_t> *index, int element_count_offset) {
        // #pragma omp parallel for schedule(dynamic)
        for (int id = 0; id < index->cur_element_count; id++) {
            if (index->element_levels_[id] < 1)
                continue;
            int level = index->element_levels_[id];
            alg_hnsw->element_levels_[id + element_count_offset] = level;
            tableint new_c = id + element_count_offset;
            alg_hnsw->linkLists_[new_c] = (char *)malloc(alg_hnsw->size_links_per_element_ * level + 1);
            if (alg_hnsw->linkLists_[new_c] == nullptr) {
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            }
            memset(alg_hnsw->linkLists_[new_c], 0, alg_hnsw->size_links_per_element_ * level + 1);
            alg_hnsw->dist_linkLists_[new_c] = (char *)malloc(alg_hnsw->size_dist_links_per_element_ * level + 1);
            if (alg_hnsw->dist_linkLists_[new_c] == nullptr) {
                throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linklist");
            }
            memset(alg_hnsw->dist_linkLists_[new_c], 0, alg_hnsw->size_dist_links_per_element_ * level + 1);
        }
    };
    allocateMemory(index1, index1_offset);
    allocateMemory(index2, index2_offset);

    memcpy(alg_hnsw->data_level0_memory_,
           index1->data_level0_memory_,
           element_count_for_index1 * alg_hnsw->size_data_per_element_);
    memcpy(alg_hnsw->data_level0_memory_ + element_count_for_index1 * alg_hnsw->size_data_per_element_,
           index2->data_level0_memory_,
           (element_count_for_index2)*alg_hnsw->size_data_per_element_);
    memcpy(alg_hnsw->dist_level0_memory_,
           index1->dist_level0_memory_,
           element_count_for_index1 * alg_hnsw->size_dist_per_element_);
    memcpy(alg_hnsw->dist_level0_memory_ + element_count_for_index1 * alg_hnsw->size_dist_per_element_,
           index2->dist_level0_memory_,
           (element_count_for_index2)*alg_hnsw->size_dist_per_element_);
    printf("time for new layer: %f\n", elapsed() - s0);

    StopW sw2 = StopW();
    StopW sw3 = StopW();
    StopW sw1 = StopW();

    double sum0 = 0.0, sum1 = 0.0;
    double sum2 = 0.0;
    for (int level = maxLevel; level >= 0; level -= 1) {
        // printf("level: %d, index1 size: %ld, index2 size: %ld\n", level, layer_node_for_index1[level].size(), layer_node_for_index2[level].size());
        if (layer_node_for_index1[level].size() > 0 && layer_node_for_index2[level].size() == 0) // copy all data for index 1 on this layer to new index
        {
            alg_hnsw->deepCopyOneLayerOnIndex(index1,
                                              level, index1_offset, layer_node_for_index1);
            continue;
        }
        if (layer_node_for_index1[level].size() == 0 && layer_node_for_index2[level].size() > 0) // copy all data for index 2 on this layer to new index
        {
            alg_hnsw->deepCopyOneLayerOnIndex(index2,
                                              level, index2_offset, layer_node_for_index2);
            continue;
        }
        // If in this layer, index1 and index2 both have data points, we need to
        // merge these 2 graphs. For data points having existed in level-1 layer from index1
        sw2.reset();
        sw3.reset();

        std::unordered_map<tableint, std::vector<std::pair<dist_t, tableint>>> candidateSetIndex2;
        candidateSetIndex2.reserve(100000);
        constexpr int NUM_BUCKETS = 128;
        std::vector<std::unordered_map<tableint, std::vector<std::pair<dist_t, tableint>>>> buckets(NUM_BUCKETS);
        std::vector<std::mutex> bucket_mutexes(NUM_BUCKETS);

        for (int cur_level = maxLevel; cur_level >= level; cur_level--) {
#pragma omp parallel for schedule(dynamic)
            for (int iter = 0; iter < layer_node_for_index1[cur_level].size(); iter++) {
                tableint cur_c = layer_node_for_index1[cur_level][iter];
                char *data_point = index1->getDataByInternalId(cur_c);

                std::unordered_map<tableint, std::vector<std::pair<dist_t, tableint>>> local_map;
                alg_hnsw->mergeIndex1BasedOnIndex2Connection(index1,
                                                             index2,
                                                             cur_c,
                                                             data_point,
                                                             index1_offset,
                                                             index2_offset,
                                                             level,
                                                             entry_point_collect_index1_on_index2[cur_c],
                                                             &local_map);
                //  &candidateSetIndex2);

                for (const auto &[key, vec] : local_map) {
                    size_t bucket_id = std::hash<tableint>{}(key) % NUM_BUCKETS;
                    std::lock_guard<std::mutex> lock(bucket_mutexes[bucket_id]);
                    auto &target = buckets[bucket_id][key];
                    target.insert(target.end(), vec.begin(), vec.end());
                }
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < NUM_BUCKETS; ++b) {
            for (const auto &[key, vec] : buckets[b]) {
#pragma omp critical
                candidateSetIndex2[key].insert(candidateSetIndex2[key].end(), vec.begin(), vec.end());
            }
        }

        sum0 += sw2.getElapsedTimeMicro();
        sw2.reset();

        for (int cur_level = maxLevel; cur_level >= level; cur_level--) {
#pragma omp parallel for schedule(dynamic)
            for (int iter = 0; iter < layer_node_for_index2[cur_level].size(); iter++) {
                tableint cur_c = layer_node_for_index2[cur_level][iter];
                char *data_point = index2->getDataByInternalId(cur_c);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;
                auto it = candidateSetIndex2.find(cur_c);
                if (it != candidateSetIndex2.end()) {
                    const std::vector<std::pair<dist_t, tableint>> &candidates = it->second;
                    int candidate_size = candidates.size();
                    linklistsizeint *ll_cur;
                    size_t linklistCount;
                    tableint *data;
                    dist_t *dist;
                    tableint candidate_id;
                    dist_t dist1;
                    dist_t lowerBound;
                    size_t Mcurmax = level ? alg_hnsw->maxM_ : alg_hnsw->maxM0_;

                    if (candidate_size + index2->getListCount(index2->get_linklist_at_level(cur_c, level)) > Mcurmax) {
                        for (const auto &p : candidates) {
                            tableint neighbor_id = p.second;
                            dist_t distance = p.first;
                            top_candidates.emplace(distance, neighbor_id);
                        }

                        ll_cur = index2->get_linklist_at_level(cur_c, level);
                        linklistCount = index2->getListCount(ll_cur);
                        data = (tableint *)(ll_cur + 1);
                        dist = index2->get_dist_at_level(cur_c, level);
                        for (size_t iter = 0; iter < linklistCount; iter++) {
                            candidate_id = data[iter] + index2_offset;
                            dist1 = dist[iter];
                            top_candidates.emplace(dist1, candidate_id);
                        }
                        alg_hnsw->getNeighborsByHeuristic2(top_candidates, Mcurmax, false);
                        ll_cur = alg_hnsw->get_linklist_at_level(cur_c + index2_offset, level);
                        alg_hnsw->setListCount(ll_cur, top_candidates.size());
                        data = (tableint *)(ll_cur + 1);
                        dist_t *distData = (dist_t *)alg_hnsw->get_dist_at_level(cur_c + index2_offset, level);
                        for (size_t idx = 0; top_candidates.size() > 0; idx++) {
                            data[idx] = top_candidates.top().second;
                            distData[idx] = top_candidates.top().first;
                            top_candidates.pop();
                        }
                    } else {
                        linklistsizeint *ll_cur_index2 = index2->get_linklist_at_level(cur_c, level);
                        size_t linklistCount_index2 = index2->getListCount(ll_cur_index2);
                        tableint *data_index2 = (tableint *)(ll_cur_index2 + 1);
                        dist_t *dist_index2 = index2->get_dist_at_level(cur_c, level);

                        ll_cur = alg_hnsw->get_linklist_at_level(cur_c + index2_offset, level);
                        alg_hnsw->setListCount(ll_cur, candidate_size + linklistCount_index2);
                        data = (tableint *)(ll_cur + 1);
                        dist_t *distData = (dist_t *)alg_hnsw->get_dist_at_level(cur_c + index2_offset, level);
                        size_t offset = candidate_size;
                        for (size_t idx = 0; idx < candidate_size; idx++) {
                            data[idx] = candidates[idx].second;
                            distData[idx] = candidates[idx].first;
                        }
                        for (size_t idx = 0; idx < linklistCount_index2; idx++) {
                            data[offset + idx] = data_index2[idx] + index2_offset;
                            distData[offset + idx] = dist_index2[idx];
                        }
                    }
                } else {
                    linklistsizeint *ll_cur_index2 = index2->get_linklist_at_level(cur_c, level);
                    size_t linklistCount_index2 = index2->getListCount(ll_cur_index2);
                    tableint *data_index2 = (tableint *)(ll_cur_index2 + 1);
                    dist_t *dist_index2 = index2->get_dist_at_level(cur_c, level);

                    linklistsizeint *ll_cur;
                    size_t linklistCount;
                    tableint *data;
                    dist_t *dist;
                    tableint candidate_id;
                    dist_t dist1;
                    dist_t lowerBound;
                    size_t Mcurmax = level ? alg_hnsw->maxM_ : alg_hnsw->maxM0_;

                    ll_cur = alg_hnsw->get_linklist_at_level(cur_c + index2_offset, level);
                    alg_hnsw->setListCount(ll_cur, linklistCount_index2);
                    data = (tableint *)(ll_cur + 1);
                    dist_t *distData = (dist_t *)alg_hnsw->get_dist_at_level(cur_c + index2_offset, level);

                    for (size_t idx = 0; idx < linklistCount_index2; idx++) {
                        data[idx] = data_index2[idx] + index2_offset;
                        distData[idx] = dist_index2[idx];
                    }
                }
            }
        }

        sum1 += sw2.getElapsedTimeMicro();
        sum2 += sw3.getElapsedTimeMicro();
    }
    printf("mergeIndex1BasedOnIndex2Connection sum0: %f\n", sum0 / 1000000);
    printf("mergeIndex1BasedOnIndex2Connection sum1: %f\n", sum1 / 1000000);
    printf("mergeIndex1BasedOnIndex2Connection sum2: %f\n", sum2 / 1000000);
    return alg_hnsw;
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
            id);
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

} // namespace hnswlib