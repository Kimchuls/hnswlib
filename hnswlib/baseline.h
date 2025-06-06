  #include "hnswalg.h"
#include <mutex>
#include <omp.h>

namespace hnswlib {

inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

inline long encode(int value1, int value2) {
    return (static_cast<long>(-value1) << 32) | (static_cast<long>(value2) & 0xFFFFFFFFL);
}

inline int decodeValue1(long encoded) {
    return static_cast<int>(-(encoded >> 32));
}

inline int decodeValue2(long encoded) {
    return static_cast<int>(encoded & 0xFFFFFFFFL);
}

template <typename dist_t>
std::unordered_set<int> computeJoinSet(HierarchicalNSW<dist_t> *index) {
    int N = index->cur_element_count;
    std::priority_queue<long, std::vector<long>, std::greater<long>> heap;

    std::unordered_set<int> joinSet;
    std::vector<bool> stale(N, false);
    std::vector<short> counts(N, 0);

    long gExit = 0L;
    for (int v = 0; v < N; ++v) {
        linklistsizeint *ll_cur = index->get_linklist_at_level(v, 0);
        size_t degree = index->getListCount(ll_cur);
        int k = (degree < 9 ? 2 : ceilDiv(degree, 4));
        gExit += k;
        int gain = k + degree;
        heap.push(encode(gain, v));
    }

    long gTot = 0L;
    while (gTot < gExit && !heap.empty()) {
        long topEncoded = heap.top();
        heap.pop();

        int gain = decodeValue1(topEncoded);
        int v = decodeValue2(topEncoded);

        linklistsizeint *ll_cur = index->get_linklist_at_level(v, 0);
        size_t degree = index->getListCount(ll_cur);

        std::vector<int> ns;
        ns.reserve(degree);
        tableint *data = (tableint *)(ll_cur + 1);
        for (size_t iter = 0; iter < degree; iter++) {
            ns.push_back(data[iter]);
        }

        int k = (degree < 9 ? 2 : ceilDiv(degree, 4));

        if (stale[v]) {
            int newGain = std::max(0, k - counts[v]);
            for (int i = 0; i < ns.size(); i++) {
                int u = ns[i];
                if (counts[u] < k && joinSet.find(u) == joinSet.end()) {
                    newGain += 1;
                }
            }
            if (newGain > 0) {
                heap.push(encode(newGain, v));
                stale[v] = false;
            }
        } else {
            joinSet.insert(v);
            gTot += gain;

            bool markNeighboursStale = (counts[v] < k);
            for (int i = 0; i < ns.size(); i++) {
                int u = ns[i];
                if (markNeighboursStale) {
                    stale[u] = true;
                }
                if (counts[u] < (k - 1)) {
                    linklistsizeint *ll_cur = index->get_linklist_at_level(u, 0);
                    size_t degree = index->getListCount(ll_cur);
                    tableint *data = (tableint *)(ll_cur + 1);
                    for (size_t iter = 0; iter < degree; iter++) {
                        ns.push_back(data[iter]);
                    }
                }
                counts[u] += 1;
            }
        }
    }

    return joinSet;
}

template <typename dist_t>
std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst>
HierarchicalNSW<dist_t>::ExtendSearchBaseLayer(const void *query_data,
                                               //  tableint &entry_point,
                                               int level,
                                               std::unordered_set<tableint> *eps) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    dist_t lowerBound = std::numeric_limits<dist_t>::max();

    for (const tableint &val : *eps) {
        dist_t dist = fstdistfunc_(query_data, getDataByInternalId(val), dist_func_param_);
        top_candidates.emplace(dist, val);
        lowerBound = std::min(dist, lowerBound);
        candidateSet.emplace(-dist, val);
        visited_array[val] = visited_array_tag;
    }

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data = (int *)get_linklist_at_level(curNodeNum, level);
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
            if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist1, candidate_id);

                while (top_candidates.size() > ef_construction_)
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
tableint HierarchicalNSW<dist_t>::addPoint(labeltype label,
                                           const void *data_point,
                                           std::unordered_set<tableint> *eps0) {
    tableint cur_c = 0;
    {
        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search != label_lookup_.end()) {
            tableint existingInternalId = search->second;
            if (allow_replace_deleted_) {
                if (isMarkedDeleted(existingInternalId)) {
                    throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                }
            }
            lock_table.unlock();

            if (isMarkedDeleted(existingInternalId)) {
                unmarkDeletedInternal(existingInternalId);
            }
            updatePoint(data_point, existingInternalId, 1.0);

            return existingInternalId;
        }

        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }

        cur_c = cur_element_count;
        cur_element_count++;
        label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = getRandomLevel(mult_);

    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
        templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
    memset(dist_level0_memory_ + cur_c * size_dist_per_element_ + offsetLevel0_, 0, size_dist_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if (curlevel) {
        linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        dist_linkLists_[cur_c] = (char *)malloc(size_dist_links_per_element_ * curlevel + 1);
        if (dist_linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linkLists_");
        memset(dist_linkLists_[cur_c], 0, size_dist_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
        if (curlevel < maxlevelcopy) {
            dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxlevelcopy; level > curlevel; level--) {
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
        }
        //    beamCandidates0 = new GraphBuilderKnnCollector(Math.min(beamWidth / 2, M * 3));
        bool epDeleted = isMarkedDeleted(enterpoint_copy);
        std::unordered_set<tableint> eps;
        eps.insert(currObj);
        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            if (level > maxlevelcopy || level < 0) // possible?
                throw std::runtime_error("Level error");
            if (level == 0) eps = *eps0;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            top_candidates = ExtendSearchBaseLayer(data_point,
                                                   level,
                                                   &eps);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand(top_candidates);
            eps.clear();
            while (!cand.empty()) {
                eps.insert(cand.top().second);
                cand.pop();
            }

            if (epDeleted) {
                top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();
            }
            currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
        }
    } else {
        // Do nothing for the first element
        enterpoint_node_ = 0;
        maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    return cur_c;
}

template <typename dist_t>
HierarchicalNSW<dist_t> *HNSWMerger_ES(HierarchicalNSW<dist_t> *index1, HierarchicalNSW<dist_t> *index2, L2Space *space, size_t M = -1, size_t ef_construction = -1) {
    // TODO: implement logic with deleted node in an index, by re-organize the
    // internal label of nodes in an index

    double s0 = elapsed();
    // make sure the larger index is index1
    if (index1->cur_element_count < index2->cur_element_count || index1->maxlevel_ < index2->maxlevel_) {
        std::swap(index1, index2);
        printf("Swap index with more current element count to index1.\n");
    }

    size_t element_count_for_index1 = index1->getCurrentElementCount();
    size_t index2_offset = element_count_for_index1;
    M = M == -1 ? std::max(index1->M_, index2->M_) : M;
    ef_construction = ef_construction == -1 ? std::max(index1->ef_construction_, index2->ef_construction_) : ef_construction;
    size_t new_max_elements = index1->max_elements_ + index2->max_elements_;
    HierarchicalNSW<dist_t> *alg_hnsw = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    size_t maxLevel = index1->maxlevel_;
    alg_hnsw->setMaxLevel(maxLevel);
    alg_hnsw->cur_element_count.store(index1->cur_element_count + index2->cur_element_count);

    alg_hnsw->data_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
    if (alg_hnsw->data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

    alg_hnsw->dist_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_dist_per_element_);
    if (alg_hnsw->dist_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

    alg_hnsw->enterpoint_node_ = index1->enterpoint_node_;

    alg_hnsw->label_lookup_.clear();
    alg_hnsw->label_lookup_.reserve(alg_hnsw->max_elements_);

    for (int id = 0; id < element_count_for_index1; id++) {
        alg_hnsw->label_lookup_[index1->getExternalLabel(id)] = id;
    }
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
    allocateMemory(index1, 0);
    memcpy(alg_hnsw->data_level0_memory_,
           index1->data_level0_memory_,
           element_count_for_index1 * alg_hnsw->size_data_per_element_);
    memcpy(alg_hnsw->dist_level0_memory_,
           index1->dist_level0_memory_,
           element_count_for_index1 * alg_hnsw->size_dist_per_element_);

    printf("time for new layer: %f\n", elapsed() - s0);

    // ElasticSearch merging algorithm
    int size = index2->cur_element_count;
    std::unordered_set<int> j = computeJoinSet(index2);
    std::unordered_map<int, int> ordMapS;
    ordMapS.reserve(size);

    for (int node : j) {
        tableint newId = alg_hnsw->addPoint(index2->getDataByInternalId(node), index2->getExternalLabel(node), -1);
        ordMapS[node] = newId;
    }

    for (int u = 0; u < size; ++u) {
        if (j.find(u) != j.end()) {
            continue;
        }
        std::unordered_set<tableint> eps;
        linklistsizeint *ll_cur = index2->get_linklist_at_level(u, 0);
        size_t degree = index2->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        for (size_t iter = 0; iter < degree; iter++) {
            int v = data[iter];
            if (v < u || j.find(v) != j.end()) {
                tableint newv = ordMapS[v];
                eps.insert(newv);

                linklistsizeint *ll_cur2 = index2->get_linklist_at_level(u, 0);
                size_t degree2 = index2->getListCount(ll_cur2);
                tableint *data2 = (tableint *)(ll_cur2 + 1);
                for (size_t iter2 = 0; iter2 < degree2; iter2++) {
                    tableint friendOrd = data2[iter2];
                    eps.insert(friendOrd);
                }
            }
        }
        tableint newId = alg_hnsw->addPoint(index2->getExternalLabel(u), index2->getDataByInternalId(u), &eps);
        ordMapS[u] = newId;
    }
    return alg_hnsw;
}

} // namespace hnswlib