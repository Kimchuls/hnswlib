#pragma once
#include "hnswalg.h"
#include <mutex>
#include <omp.h>
#include <random>
#include <iostream>
#include <vector>
#include <cstring>

using namespace std;

extern float global_counter;

namespace hnswlib {

class VarArray {
public:
    VarArray() :
        data_size_(0), size_(0), capacity_(0) {
    }
    VarArray(size_t data_size) :
        data_size_(data_size), size_(0), capacity_(0) {
    }

    void push_back(const char *src) {
        if (size_ == capacity_) reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        std::memcpy(data_.data() + size_ * data_size_, src, data_size_);
        ++size_;
    }

    char *operator[](size_t idx) {
        return data_.data() + idx * data_size_;
    }

    const char *operator[](size_t idx) const {
        return data_.data() + idx * data_size_;
    }

    size_t size() const {
        return size_;
    }

    void clear() {
        size_ = 0;
    }

    void release() {
        std::vector<char>().swap(data_);
        size_ = 0;
        capacity_ = 0;
    }

private:
    void reserve(size_t new_capacity) {
        std::vector<char> new_data(new_capacity * data_size_);
        if (!data_.empty()) {
            std::memcpy(new_data.data(), data_.data(), size_ * data_size_);
        }
        data_.swap(new_data);
        capacity_ = new_capacity;
    }

    size_t data_size_;
    size_t size_;
    size_t capacity_;
    std::vector<char> data_;
};

int read_from_disk(const std::string &location,
                   size_t offset,
                   char *dest,
                   size_t data_size) {
    try {
        std::ifstream input(location, std::ios::binary);
        input.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        input.seekg(offset, std::ios::beg);
        input.read(dest, data_size);

        return 0;
    } catch (const std::ios_base::failure &e) {
        return 1;
    }
}

int write_to_disk(const std::string &location,
                  size_t offset,
                  const char *dest,
                  size_t data_size) {
    try {
        std::fstream output(location, std::ios::in | std::ios::out | std::ios::binary);
        output.exceptions(std::fstream::failbit | std::fstream::badbit);

        output.seekp(offset, std::ios::beg);
        output.write(dest, data_size);

        return 0;
    } catch (const std::ios_base::failure &e) {
        return 1;
    }
}

template <typename dist_t>
class HierarchicalNSW_ME : public AlgorithmInterface<dist_t> {
public:
    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};
    size_t size_data_per_element_{0};
    size_t size_dist_per_element_{0};
    size_t size_links_per_element_{0};
    size_t size_dist_links_per_element_{0};
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    char *dist_level0_memory_{nullptr};
    char **dist_linkLists_{nullptr};
    std::vector<int> element_levels_; 
    std::vector<int> id_data_map_; 
    VarArray data_;

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    std::unordered_map<labeltype, tableint> label_lookup_;

    size_t index_offset_list0{0};
    size_t index_offset_dist0{0};
    size_t index_offset_linklist{0};
    size_t index_offset_distlist{0};
    std::vector<size_t> index_linklist_offset;
    std::vector<size_t> index_distlist_offset;
    std::string file_path;

    HierarchicalNSW_ME(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200
        ) :
        element_levels_(max_elements),
        id_data_map_(max_elements, -1),
        index_linklist_offset(max_elements + 1),
        index_distlist_offset(max_elements + 1)
    {
        max_elements_ = max_elements;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if (M <= 10000) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        size_dist_per_element_ = maxM0_ * sizeof(dist_t);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;
        data_ = VarArray(data_size_);

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        enterpoint_node_ = -1;
        maxlevel_ = -1;
        linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        dist_linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
        if (dist_linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate dist_linkLists_");

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_dist_links_per_element_ = maxM_ * sizeof(dist_t);
    }

    void allocate_level0() {
        data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
        dist_level0_memory_ = (char *)malloc(max_elements_ * size_dist_per_element_);
        if (dist_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
    }

    void clear_level0() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        free(dist_level0_memory_);
        dist_level0_memory_ = nullptr;
    }

    void clear_linklist() {
        for (tableint i = 0; i < cur_element_count; i++) {
            if (linkLists_[i] != nullptr) {
                free(linkLists_[i]);
                free(dist_linkLists_[i]);
                linkLists_[i] = nullptr;
                dist_linkLists_[i] = nullptr;
            }
        }
    }

    void clear() {
        clear_level0();
        clear_linklist();

        free(linkLists_);
        free(dist_linkLists_);
        linkLists_ = nullptr;
        dist_linkLists_ = nullptr;

        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }

    ~HierarchicalNSW_ME() {
        clear();
    }

    void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) override {
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void *, size_t, BaseFilterFunctor *isIdAllowed = nullptr) const override {
        return {};
    }

    std::vector<std::pair<dist_t, labeltype>>
    searchKnnCloserFirst(const void *query_data,
                         size_t k,
                         BaseFilterFunctor *isIdAllowed = nullptr) const override {
        return {};
    }

    void saveIndex(const std::string &location) override {
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t loadMetaData(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);
        size_t data_size = 0;
        if (!input.is_open())
            throw std::runtime_error("Cannot open file");
        file_path = location;
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        data_size += sizeof(offsetLevel0_);

        readBinaryPOD(input, max_elements_);
        data_size += sizeof(max_elements_);
        readBinaryPOD(input, cur_element_count);
        data_size += sizeof(cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        data_size += sizeof(size_data_per_element_);
        readBinaryPOD(input, size_dist_per_element_);
        data_size += sizeof(size_dist_per_element_);
        readBinaryPOD(input, label_offset_);
        data_size += sizeof(label_offset_);
        readBinaryPOD(input, offsetData_);
        data_size += sizeof(offsetData_);
        readBinaryPOD(input, maxlevel_);
        data_size += sizeof(maxlevel_);
        readBinaryPOD(input, enterpoint_node_);
        data_size += sizeof(enterpoint_node_);

        readBinaryPOD(input, maxM_);
        data_size += sizeof(maxM_);
        readBinaryPOD(input, maxM0_);
        data_size += sizeof(maxM0_);
        readBinaryPOD(input, M_);
        data_size += sizeof(M_);
        readBinaryPOD(input, mult_);
        data_size += sizeof(mult_);
        readBinaryPOD(input, ef_construction_);
        data_size += sizeof(ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        input.seekg(cur_element_count * size_dist_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }

            unsigned int distLinkListSize;
            readBinaryPOD(input, distLinkListSize);
            if (distLinkListSize != 0) {
                input.seekg(distLinkListSize, input.cur);
            }
        }

        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();

        input.seekg(pos, input.beg);
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        input.seekg(cur_element_count * size_dist_per_element_, input.cur);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_dist_links_per_element_ = maxM_ * sizeof(dist_t);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        element_levels_ = std::vector<int>(max_elements);
        size_t offset = 0;
        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize;

            readBinaryPOD(input, linkListSize);
            offset += linkListSize + sizeof(linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                input.seekg(linkListSize, input.cur);
            }
            linkLists_[i] = nullptr;
            index_distlist_offset[i] = offset;

            unsigned int distLinkListSize;
            readBinaryPOD(input, distLinkListSize);
            offset += distLinkListSize + sizeof(distLinkListSize);
            if (distLinkListSize > 0) {
                input.seekg(distLinkListSize, input.cur);
            }
            dist_linkLists_[i] = nullptr;
            index_linklist_offset[i + 1] = offset;
        }

        input.close();
        index_offset_list0 = data_size;
        index_offset_dist0 = data_size + cur_element_count * size_data_per_element_;
        index_offset_linklist = index_offset_dist0 + cur_element_count * size_dist_per_element_;
        index_offset_distlist = index_offset_linklist;
        return 0;
    }

    size_t get_data_position(tableint internalId) const {
        return index_offset_list0 + internalId * size_data_per_element_ + offsetData_;
    }

    size_t get_linklist_level_position(tableint internalId, int layer) const {
        return index_offset_linklist + index_linklist_offset[internalId] + sizeof(unsigned int) + (layer - 1) * size_links_per_element_;
    }

    size_t get_link0_position(tableint internalId) const {
        return index_offset_list0 + internalId * size_data_per_element_;
    }

    size_t get_link_position(tableint internal_id, int level) const {
        return level == 0 ? get_link0_position(internal_id) : get_linklist_level_position(internal_id, level);
    }

    size_t get_disklink_level_position(tableint internalId, int layer) const {
        return index_offset_distlist + index_distlist_offset[internalId] + sizeof(unsigned int) + (layer - 1) * size_dist_links_per_element_;
    }

    size_t get_dist0_position(tableint internalId) const {
        return index_offset_dist0 + internalId * size_dist_per_element_;
    }

    size_t get_dist_position(tableint internal_id, int level) const {
        return level == 0 ? get_dist0_position(internal_id) : get_disklink_level_position(internal_id, level);
    }

    inline char *getDataByInternalId_from_memory(tableint internal_id, int layer) {
        if (layer == 0) {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        } else {
            if (internal_id >= id_data_map_.size())
                return nullptr;
            return data_[id_data_map_[internal_id]];
        }
    }

    void getDataByInternalId_from_disk(tableint internal_id, char *dest) const {
        size_t offset = get_data_position(internal_id);
        read_from_disk(file_path, offset, dest, data_size_);
    }

    void load_graph0() {
        data_.release();
        data_level0_memory_ = (char *)malloc(cur_element_count * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
        dist_level0_memory_ = (char *)malloc(cur_element_count * size_dist_per_element_);
        if (dist_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
        read_from_disk(file_path, index_offset_list0, data_level0_memory_, cur_element_count * size_data_per_element_);
        read_from_disk(file_path, index_offset_dist0, dist_level0_memory_, cur_element_count * size_dist_per_element_);
    }

    void load_graph(int level, bool with_data = true) {
        clear_linklist();
        if (level == 0) {
            load_graph0();
            return;
        }
        for (int i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] < level)
                continue;
            linkLists_[i] = (char *)malloc(size_links_per_element_);
            dist_linkLists_[i] = (char *)malloc(size_dist_links_per_element_);
            read_from_disk(file_path, get_link_position(i, level), linkLists_[i], size_links_per_element_);
            read_from_disk(file_path, get_dist_position(i, level), dist_linkLists_[i], size_dist_links_per_element_);
            if (with_data && id_data_map_[i] == -1) {
                char *data = (char *)malloc(sizeof(char) * data_size_);
                getDataByInternalId_from_disk(i, data);
                data_.push_back(data);
                id_data_map_[i] = data_.size() - 1;
            }
        }
    }

    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_); // offsetLevel0_ always be 0 here
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *)(linkLists_[internal_id]);
    }

    linklistsizeint *get_linklist_at_level_from_memory(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    void get_linklist_at_level_from_disk(tableint internal_id, int level, char *dest, size_t size) const {
        size_t offset = get_link_position(internal_id, level);
        read_from_disk(file_path, offset, dest, size);
    }

    float *get_dist0(tableint internal_id) const {
        return (float *)(dist_level0_memory_ + internal_id * size_dist_per_element_);
    }

    float *get_dist(tableint internal_id, int level) const {
        return (float *)(dist_linkLists_[internal_id]);
    }

    float *get_dist_at_level_from_memory(tableint internal_id, int level) const {
        return level == 0 ? get_dist0(internal_id) : get_dist(internal_id, level);
    }

    void get_dist_at_level_from_disk(tableint internal_id, int level, char *dest, size_t size) const {
        size_t offset = get_dist_position(internal_id, level);
        read_from_disk(file_path, offset, dest, size);
    }

    unsigned short int getListCount(linklistsizeint *ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint *ptr, unsigned short int size) const {
        *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
    }

    void setLinkList(tableint internal_id, int level, linklistsizeint *source, size_t size) {
        size_t offset = get_link_position(internal_id, level);
        write_to_disk(file_path, offset, reinterpret_cast<char *>(source), size);
    }

    void setDistList(tableint internal_id, int level, dist_t *source, size_t size) {
        size_t offset = get_dist_position(internal_id, level);
        write_to_disk(file_path, offset, reinterpret_cast<char *>(source), size);
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                  std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M,
        bool collect_metrics = true,
        tableint cur_c = -1,
        float alpha = 1.0) {
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
                dist_t curdist =
                    fstdistfunc_(getDataByInternalId_from_memory(second_pair.second, 1),
                                 getDataByInternalId_from_memory(curent_pair.second, 1),
                                 dist_func_param_);
                if (curdist * alpha < dist_to_query) {
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

    void searchNodeOnEachLayer(
        std::vector<std::vector<int>> &resultVector,
        bool combine = false);

    void search2Layer(
        const void *query_data,
        tableint &enterpoint_node,
        int level_higher,
        int level_lower,
        int lambda,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> *top_candidates = nullptr,
        int offset = 0);
};

template <typename dist_t>
void HierarchicalNSW_ME<dist_t>::searchNodeOnEachLayer(
    std::vector<std::vector<int>> &resultVector,
    bool combine) {
    size_t elementCount = getCurrentElementCount();
    for (size_t iter = 0; iter < elementCount; iter++) {
        resultVector[element_levels_[iter]].push_back(iter);
    }
    if (combine) {
        for (int i = resultVector.size() - 2; i >= 0; --i) {
            resultVector[i].insert(resultVector[i].end(), resultVector[i + 1].begin(), resultVector[i + 1].end());
        }
    }
}

template <typename dist_t>
void deepCopyOneLayerOnIndex(HierarchicalNSW_ME<dist_t> *index,
                             HierarchicalNSW_ME<dist_t> *alg_hnsw,
                             int level,
                             int offset,
                             std::vector<std::vector<int>> &layer_node_for_index) {
    index->load_graph(level, false);
    int link_size = (level == 0) ? alg_hnsw->size_links_level0_ : alg_hnsw->size_links_per_element_;
    int dist_size = (level == 0) ? alg_hnsw->size_dist_per_element_ : alg_hnsw->size_dist_links_per_element_;
    for (int iter = 0; iter < layer_node_for_index[level].size(); iter++) {
        tableint cur_c = layer_node_for_index[level][iter];
        tableint new_c = cur_c + offset;

        linklistsizeint *ll_cur = index->get_linklist_at_level_from_memory(cur_c, level);

        int linklistCount = index->getListCount(ll_cur);
        tableint *data_cur = (tableint *)(ll_cur + 1);
        dist_t *ll_cur_dist = index->get_dist_at_level_from_memory(cur_c, level);
        for (int i = 0; i < linklistCount; i++) {
            data_cur[i] = data_cur[i] + offset;
        }
        alg_hnsw->setLinkList(new_c, level, data_cur, link_size);
        alg_hnsw->setDistList(new_c, level, ll_cur_dist, dist_size);
    }
    index->clear_linklist();
}
/*
 * This is the specific function for ME version of algorithm.
 * 
 * This function, which is used in search2Layer, searching the graph from the
 * top layer to the level_low layer. If the enterpoint_node is -1, it will be
 * set to enterpoint_node_. Otherwise, the enterpoint_node will be used as the
 * starting point, as it's the closest point to the query on layer level_low
 * + 1. The function will also return the `top_candidates` which contains the
 * `lambda`-closest points to the query on layer level_low.
 */
template <typename dist_t>
void HierarchicalNSW_ME<dist_t>::search2Layer(const void *query_data,
                                              tableint &enterpoint_node,
                                              int level_higher,
                                              int level_lower,
                                              int lambda,
                                              std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> *top_candidates,
                                              int offset) {
    tableint currObj = enterpoint_node == -1 ? enterpoint_node_ : enterpoint_node;

    dist_t curdist = fstdistfunc_(query_data, getDataByInternalId_from_memory(currObj, level_higher), dist_func_param_);
    for (int level = level_higher; level > level_lower; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int *data = (unsigned int *)get_linklist_at_level_from_memory(currObj, level);
            int size = getListCount(data);

            tableint *datal = (tableint *)(data + 1);

            for (int i = 0; i < size; i++) {
                tableint cand = datal[i];
                if (cand < 0 || cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }
                dist_t d = fstdistfunc_(query_data, getDataByInternalId_from_memory(cand, level_higher), dist_func_param_);

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

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    dist_t lowerBound;
    dist_t entry_dist;
    dist_t dist = fstdistfunc_(query_data, getDataByInternalId_from_memory(currObj, level_higher), dist_func_param_);
    top_candidates->emplace(dist, currObj + offset);
    enterpoint_node = currObj;
    entry_dist = dist;
    lowerBound = dist;
    candidateSet.emplace(-dist, currObj);
    visited_array[currObj] = visited_array_tag;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound && top_candidates->size() == lambda) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;


        int *data = (int *)get_linklist_at_level_from_memory(curNodeNum, level_higher);
        size_t size = getListCount((linklistsizeint *)data);
        tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId_from_memory(*datal, level_higher), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId_from_memory(*(datal + 1), level_higher), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = *(datal + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId_from_memory(*(datal + j + 1), level_higher), _MM_HINT_T0);
#endif
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;

            char *currObj1 = (getDataByInternalId_from_memory(candidate_id, level_higher));

            dist_t dist1 = fstdistfunc_(query_data, currObj1, dist_func_param_);
            if (top_candidates->size() < lambda || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId_from_memory(candidateSet.top().second, level_higher), _MM_HINT_T0);
#endif

                top_candidates->emplace(dist1, candidate_id + offset);
                if (entry_dist > dist1) {
                    enterpoint_node = candidate_id;
                    entry_dist = dist1;
                }

                while (top_candidates->size() > lambda)
                    top_candidates->pop();

                if (!top_candidates->empty())
                    lowerBound = top_candidates->top().first;
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
}

template <typename dist_t>
void HNSWMerger_ME(
    const std::string &location_index1,
    const std::string &location_index2,
    const std::string &location_merged,
    L2Space *space,
    size_t max_elements,
    size_t M = -1,
    size_t ef_construction = -1,
    int lambda = 4,
    float alpha = 1.05) {
    double s0 = elapsed();
    HierarchicalNSW_ME<dist_t> *index1 = new hnswlib::HierarchicalNSW_ME<float>(space, max_elements, M, ef_construction);
    index1->loadMetaData(location_index1, space);
    HierarchicalNSW_ME<dist_t> *index2 = new hnswlib::HierarchicalNSW_ME<float>(space, max_elements, M, ef_construction);
    index2->loadMetaData(location_index2, space);

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
    size_t new_max_elements = index1->max_elements_ + index2->max_elements_;

    tableint *entry_point_collect_index1_on_index2 = new tableint[element_count_for_index1];
    memset(entry_point_collect_index1_on_index2, -1, element_count_for_index1 * sizeof(int));
    int entry_point_index1 = index2->enterpoint_node_;

    size_t maxLevel = std::max(index1->maxlevel_, index2->maxlevel_);
    std::vector<std::vector<int>> layer_node_for_index1(maxLevel + 2);
    index1->searchNodeOnEachLayer(layer_node_for_index1, true);
    std::vector<std::vector<int>> layer_node_for_index2(maxLevel + 2);
    index2->searchNodeOnEachLayer(layer_node_for_index2, true);

    size_t alg_hnsw_maxM_ = M;
    size_t alg_hnsw_maxM0_ = M * 2;

    {
        HierarchicalNSW<dist_t> *alg_hnsw = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
        alg_hnsw->cur_element_count.store(index1->cur_element_count + index2->cur_element_count);
        alg_hnsw->data_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
        if (alg_hnsw->data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        alg_hnsw->dist_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_dist_per_element_);
        if (alg_hnsw->dist_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

        int ret;
        ret = read_from_disk(location_index1, index1->index_offset_list0, alg_hnsw->data_level0_memory_, element_count_for_index1 * alg_hnsw->size_data_per_element_);
        if (ret)
            throw std::runtime_error("Failed to load index1 data level0");
        ret = read_from_disk(location_index2, index2->index_offset_list0, alg_hnsw->data_level0_memory_ + element_count_for_index1 * alg_hnsw->size_data_per_element_, element_count_for_index2 * alg_hnsw->size_data_per_element_);
        if (ret)
            throw std::runtime_error("Failed to load index2 data level0");

        ret = read_from_disk(location_index1, index1->index_offset_dist0, alg_hnsw->dist_level0_memory_, element_count_for_index1 * alg_hnsw->size_dist_per_element_);
        if (ret)
            throw std::runtime_error("Failed to load index1 data level0");
        ret = read_from_disk(location_index2, index2->index_offset_dist0, alg_hnsw->dist_level0_memory_ + element_count_for_index1 * alg_hnsw->size_dist_per_element_, element_count_for_index2 * alg_hnsw->size_dist_per_element_);
        if (ret)
            throw std::runtime_error("Failed to load index1 data level0");

        int layer_max_min = std::min(index1->maxlevel_, index2->maxlevel_);
        int bound = layer_node_for_index1[layer_max_min + 1].size() + layer_node_for_index2[layer_max_min + 1].size();
        if (bound == 0) {
            bound = 1;
        }
        std::unordered_set<int> newLayerSet;
        if (layer_node_for_index1[layer_max_min].size() + layer_node_for_index2[layer_max_min].size() > M * bound) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            int lambda = 0;
            auto filter_and_remove = [&](std::vector<int> &layer_nodes, std::vector<int> &upper_layer_nodes, int offset) {
                std::vector<int> newLayer;
                auto it = layer_nodes.begin();
                while (it != layer_nodes.end()) {
                    if (distribution(alg_hnsw->level_generator_) < 1.0 / M || lambda == M) {
                        tableint old_c = *it;
                        tableint new_c = old_c + offset;
                        newLayer.push_back(new_c);
                        upper_layer_nodes.push_back(old_c);
                        newLayerSet.insert(new_c);
                        lambda = 0;

                        alg_hnsw->linkLists_[new_c] = (char *)malloc(alg_hnsw->size_links_per_element_ * (layer_max_min + 1) + 1);
                        if (alg_hnsw->linkLists_[new_c] == nullptr)
                            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                        memset(alg_hnsw->linkLists_[new_c], 0, alg_hnsw->size_links_per_element_ * (layer_max_min + 1) + 1);
                        alg_hnsw->element_levels_[new_c] = (layer_max_min + 1);

                        alg_hnsw->dist_linkLists_[new_c] = (char *)malloc(alg_hnsw->size_dist_links_per_element_ * (layer_max_min + 1) + 1);
                        if (alg_hnsw->dist_linkLists_[new_c] == nullptr)
                            throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linkLists");
                        memset(alg_hnsw->dist_linkLists_[new_c], 0, alg_hnsw->size_dist_links_per_element_ * (layer_max_min + 1) + 1);
                    } else {
                        lambda++;
                        ++it;
                    }
                }
                if (newLayer.size() > 0) { 
                    if (newLayer.size() > M) {
                        throw std::logic_error("too many points on new-top here, debug for less neighbors");
                    }
                    for (int newLayerIter = 0; newLayerIter < newLayer.size(); newLayerIter++) {
                        linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(newLayer[newLayerIter], layer_max_min + 1);
                        alg_hnsw->setListCount(ll_cur, newLayer.size() - 1);
                        tableint *data = (tableint *)(ll_cur + 1);
                        dist_t *dist = alg_hnsw->get_dist_at_level(newLayer[newLayerIter], layer_max_min + 1);
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
                }
            };
            if (layer_node_for_index1[layer_max_min + 1].size() == 0 && layer_node_for_index1[layer_max_min].size() > M) {
                filter_and_remove(layer_node_for_index1[layer_max_min], layer_node_for_index1[layer_max_min + 1], index1_offset);
                index1->maxlevel_++;
                index1->enterpoint_node_ = layer_node_for_index1[layer_max_min + 1][0];
            }
            if (layer_node_for_index2[layer_max_min + 1].size() == 0 && layer_node_for_index2[layer_max_min].size() > M) {
                filter_and_remove(layer_node_for_index2[layer_max_min], layer_node_for_index2[layer_max_min + 1], index2_offset);
                index2->maxlevel_++;
                index2->enterpoint_node_ = layer_node_for_index2[layer_max_min + 1][0];
            }
        } 
        if (index2->maxlevel_ < index1->maxlevel_) {
            throw std::logic_error("index1 is higher than index2, wrong case");
        }
        maxLevel = index2->maxlevel_;
        alg_hnsw->setMaxLevel(maxLevel);
        alg_hnsw->enterpoint_node_ = index2->enterpoint_node_ + index2_offset;

        alg_hnsw->label_lookup_.clear();
        alg_hnsw->label_lookup_.reserve(alg_hnsw->max_elements_);

        for (int id = 0; id < alg_hnsw->getCurrentElementCount(); id++) {
            alg_hnsw->label_lookup_[alg_hnsw->getExternalLabel(id)] = id;
        }

        auto allocateMemory = [&](hnswlib::HierarchicalNSW_ME<dist_t> *index, int element_count_offset) {
            for (int id = 0; id < index->cur_element_count; id++) {
                tableint new_c = id + element_count_offset;
                if (index->element_levels_[id] < 1 || newLayerSet.find(new_c) != newLayerSet.end())
                    continue;
                int level = index->element_levels_[id];
                alg_hnsw->element_levels_[new_c] = level;

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

        // printf("time for new layer: %f\n", elapsed() - s0);
        alg_hnsw->saveIndex(location_merged);
        // printf("total time for meta data and index setting: %f\n", elapsed() - s0);
        delete alg_hnsw;
    }

    HierarchicalNSW_ME<dist_t> *alg_hnsw = new hnswlib::HierarchicalNSW_ME<float>(space, max_elements, M, ef_construction);
    alg_hnsw->loadMetaData(location_merged, space);

    StopW sw2 = StopW();
    StopW sw3 = StopW();
    StopW sw1 = StopW();

    double sum0 = 0.0, sum1 = 0.0;
    double sum2 = 0.0;
    char *data_point = (char *)malloc(sizeof(char) * index1->data_size_);
    char *temp_data = (char *)malloc(sizeof(char) * index1->data_size_);
    for (int level = maxLevel; level >= 0; level -= 1) {
        index2->load_graph(level, true);

        if (layer_node_for_index1[level].size() > 0 && layer_node_for_index2[level].size() == 0) {
            throw std::logic_error("index1 is higher than index2, wrong case");
        }
        if (layer_node_for_index1[level].size() == 0 && layer_node_for_index2[level].size() > 0) {
            deepCopyOneLayerOnIndex(index2, alg_hnsw, level, index2_offset, layer_node_for_index2);
            continue;
        }

        sw2.reset();
        sw3.reset();

        size_t Mcurmax = level ? alg_hnsw_maxM_ : alg_hnsw_maxM0_;
        std::vector<std::vector<std::pair<dist_t, tableint>>> candidateSetIndex2(element_count_for_index2);
        std::vector<std::mutex> mtx(element_count_for_index2);

        size_t ll_cur_size = (level == 0) ? alg_hnsw->size_links_level0_ : alg_hnsw->size_links_per_element_;
        linklistsizeint *ll_cur = (linklistsizeint *)malloc(ll_cur_size);

        size_t dist_size = (level == 0) ? alg_hnsw->size_dist_per_element_ : alg_hnsw->size_dist_links_per_element_;
        dist_t *dist = (dist_t *)malloc(dist_size);

#pragma omp parallel for
        for (int i = 0; i < index1->cur_element_count; i++) {
            tableint cur_c = i;
            tableint new_c = cur_c + index1_offset;

            index1->getDataByInternalId_from_disk(cur_c, data_point);

            tableint &last_entry_point = entry_point_collect_index1_on_index2[cur_c];
            if (index1->element_levels_[i] < level) {
                index2->search2Layer(data_point,
                                     last_entry_point,
                                     level,
                                     level-1,
                                     lambda,
                                     nullptr,
                                     index2_offset);
                continue;
            }

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW_ME<dist_t>::CompareByFirst> top_candidates;

            index2->search2Layer(data_point,
                                 last_entry_point,
                                 level,
                                 level,
                                 lambda,
                                 &top_candidates,
                                 index2_offset);

            auto temp_queue = top_candidates;
            while (!temp_queue.empty()) {
                std::pair<dist_t, tableint> current = temp_queue.top();
                temp_queue.pop();

                alg_hnsw->data_.push_back(index2->getDataByInternalId_from_memory(current.second - index2_offset, level));
                alg_hnsw->id_data_map_[current.second] = alg_hnsw->data_.size() - 1;

                std::lock_guard<std::mutex> lk(mtx[current.second - index2_offset]);
                candidateSetIndex2[current.second - index2_offset].push_back(std::make_pair(current.first, cur_c + index1_offset));
            }

            index1->get_linklist_at_level_from_disk(cur_c, level, reinterpret_cast<char *>(ll_cur), ll_cur_size);
            int linklistCount = index1->getListCount(ll_cur);
            tableint *data = (tableint *)(ll_cur + 1);

            index1->get_dist_at_level_from_disk(cur_c, level, reinterpret_cast<char *>(dist), dist_size);

            if (top_candidates.size() + linklistCount > Mcurmax) {
                for (size_t iter = 0; iter < linklistCount; iter++) {
                    top_candidates.emplace(dist[iter], data[iter] + index1_offset);
                    index1->getDataByInternalId_from_disk(data[iter], temp_data);
                    alg_hnsw->data_.push_back(temp_data);
                    alg_hnsw->id_data_map_[data[iter] + index1_offset] = alg_hnsw->data_.size() - 1;
                }
                alg_hnsw->getNeighborsByHeuristic2(top_candidates, Mcurmax, false, -1, alpha);

                alg_hnsw->setListCount(ll_cur, top_candidates.size());
                for (size_t idx = 0; top_candidates.size() > 0; idx++) {
                    data[idx] = top_candidates.top().second;
                    dist[idx] = top_candidates.top().first;
                    top_candidates.pop();
                }
            } else {
                int length = top_candidates.size() + linklistCount;
                alg_hnsw->setListCount(ll_cur, length);

                for (size_t idx = 0; idx < linklistCount; idx++) {
                    data[idx] = data[idx] + index1_offset;
                }
                size_t offset = linklistCount;
                for (size_t idx = 0; top_candidates.size() > 0; idx++) {
                    data[idx + offset] = top_candidates.top().second;
                    dist[idx + offset] = top_candidates.top().first;
                    top_candidates.pop();
                }
            }
            alg_hnsw->data_.release();
            alg_hnsw->setLinkList(new_c, level, ll_cur, ll_cur_size);
            alg_hnsw->setDistList(new_c, level, dist, dist_size);
        }
        sum0 += sw2.getElapsedTimeMicro();
        sw2.reset();
        

        int batch_size = 64;
#pragma omp parallel for
        for (int b = 0; b < layer_node_for_index2[level].size(); b += batch_size) {
            for (int i = b; i < b + batch_size && i < layer_node_for_index2[level].size(); ++i) {
                tableint cur_c = layer_node_for_index2[level][i];
                tableint new_c = cur_c + index2_offset;
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW_ME<dist_t>::CompareByFirst> top_candidates;

                index2->get_linklist_at_level_from_disk(cur_c, level, reinterpret_cast<char *>(ll_cur), ll_cur_size);
                int linklistCount = index2->getListCount(ll_cur);
                tableint *data = (tableint *)(ll_cur + 1);
                index2->get_dist_at_level_from_disk(cur_c, level, reinterpret_cast<char *>(dist), dist_size);

                if (candidateSetIndex2[cur_c].size() > 0) {
                    const std::vector<std::pair<dist_t, tableint>> &candidates = candidateSetIndex2[cur_c];
                    int candidate_size = candidates.size();

                    if (candidate_size + linklistCount > Mcurmax) {
                        for (const auto &p : candidates) {
                            tableint neighbor_id = p.second;
                            dist_t distance = p.first;

                            index1->getDataByInternalId_from_disk(neighbor_id - index1_offset, temp_data);
                            alg_hnsw->data_.push_back(temp_data);
                            alg_hnsw->id_data_map_[neighbor_id] = alg_hnsw->data_.size() - 1;

                            top_candidates.emplace(distance, neighbor_id);
                        }

                        for (size_t iter = 0; iter < linklistCount; iter++) {
                            top_candidates.emplace(dist[iter], data[iter] + index2_offset);

                            index2->getDataByInternalId_from_disk(data[iter], temp_data);
                            alg_hnsw->data_.push_back(temp_data);
                            alg_hnsw->id_data_map_[data[iter] + index2_offset] = alg_hnsw->data_.size() - 1;
                        }
                        alg_hnsw->getNeighborsByHeuristic2(top_candidates, Mcurmax, false, -1, alpha);

                        alg_hnsw->setListCount(ll_cur, top_candidates.size());
                        for (size_t idx = 0; top_candidates.size() > 0; idx++) {
                            data[idx] = top_candidates.top().second;
                            dist[idx] = top_candidates.top().first;
                            top_candidates.pop();
                        }
                        alg_hnsw->data_.release();
                    } else {
                        int length = candidates.size() + linklistCount;
                        alg_hnsw->setListCount(ll_cur, length);

                        for (size_t idx = 0; idx < linklistCount; idx++) {
                            data[idx] = data[idx] + index2_offset;
                        }
                        size_t offset = linklistCount;
                        for (size_t idx = 0; idx < candidate_size; idx++) {
                            data[idx + offset] = candidates[idx].second;
                            dist[idx + offset] = candidates[idx].first;
                        }
                    }
                } else {
                    for (size_t idx = 0; idx < linklistCount; idx++) {
                        data[idx] = data[idx] + index2_offset;
                    }
                }
                alg_hnsw->setLinkList(new_c, level, ll_cur, ll_cur_size);
                alg_hnsw->setDistList(new_c, level, dist, dist_size);
            }
        }
        delete ll_cur;
        delete dist;
        index2->clear_linklist();
        sum1 += sw2.getElapsedTimeMicro();
        sum2 += sw3.getElapsedTimeMicro();
    }
    delete data_point;
    delete temp_data;
    delete alg_hnsw;
    delete index1;
    delete index2;

    // printf("mergeIndex1BasedOnIndex2Connection sum0: %f\n", sum0 / 1000000);
    // printf("mergeIndex1BasedOnIndex2Connection sum1: %f\n", sum1 / 1000000);
    // printf("mergeIndex1BasedOnIndex2Connection sum2: %f\n", sum2 / 1000000);
    // global_counter += sum2 / 1000000;
}

} // namespace hnswlib