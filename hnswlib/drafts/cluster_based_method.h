#include "hnswalg.h"

namespace hnswlib{
  template <typename dist_t>
  void HierarchicalNSW<dist_t>::HNSWMerger(
      HierarchicalNSW<dist_t> *index1,
      HierarchicalNSW<dist_t> *index2)
  {
    size_t maxLevel = std::max(index1->maxlevel_, index2->maxlevel_);
    setMaxLevel(maxLevel);
    cur_element_count.store(index1->cur_element_count + index2->cur_element_count);
    enterpoint_node_ = index1->enterpoint_node_;

    size_t element_count_for_index1 = index1->getCurrentElementCount();
    size_t element_count_for_index2 = index2->getCurrentElementCount();
    std::vector<std::vector<int>> layer_node_for_index1(maxlevel_ + 2);
    index1->searchNodeOnEachLayer(layer_node_for_index1, true);
    std::vector<std::vector<int>> layer_node_for_index2(maxlevel_ + 2);
    index2->searchNodeOnEachLayer(layer_node_for_index2, true);
    int dim = data_size_ / sizeof(dist_t);

    for (tableint old_c = 0; old_c < element_count_for_index1; old_c++)
    {
      tableint new_c = old_c;
      int level = index1->element_levels_[old_c];
      element_levels_[new_c] = level;

      if (level == 0)
        continue;
      linkLists_[new_c] = (char *)malloc(size_links_per_element_ * level + 1);
      if (linkLists_[new_c] == nullptr)
      {
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      }
      memcpy(linkLists_[new_c], index1->linkLists_[old_c], size_links_per_element_ * level + 1);
      dist_linkLists_[new_c] = (char *)malloc(size_dist_links_per_element_ * level + 1);
      if (dist_linkLists_[new_c] == nullptr)
      {
        throw std::runtime_error("Not enough memory: addPoint failed to allocate dist_linklist");
      }
      memcpy(dist_linkLists_[new_c], index1->dist_linkLists_[old_c], size_dist_links_per_element_ * level + 1);
    }
    memcpy(data_level0_memory_, index1->data_level0_memory_, element_count_for_index1 * size_data_per_element_);
    memcpy(dist_level0_memory_, index1->dist_level0_memory_, element_count_for_index1 * size_dist_per_element_);

    // TODO: approach 1 - clustering index2 into K clusters and assign each cluster to index1
    {
      // for (tableint old_c = 0; old_c < element_count_for_index2; old_c++)
      // {
      //     tableint new_c = old_c + element_count_for_index1;
      //     memcpy(getDataByInternalId(new_c), index2->getDataByInternalId(old_c), data_size_);
      //     setExternalLabel(new_c, index2->getExternalLabel(old_c));
      //     int level = index2->element_levels_[old_c];
      //     element_levels_[new_c] = 0;
      //     // TODO: Now we assume inserted index2 put all its nodes on index1 level 0; We may test on other levels
      // }

      // // TODO: add a logic for layer increment

      // linklistsizeint *ll_cur;
      // size_t size;
      // tableint *data;
      // dist_t *distData;
      // // for (int level = maxlevel_; level >= 0; level -= 1)
      // for (int level = 0; level >= 0; level -= 1)
      // {
      //     // TODO: check the performance of using faiss kmeans instead // complie faiss library and use the -lfaiss flag when compiling the code
      //     int T = 5000;
      //     int train_size = index2->cur_element_count;
      //     int K = train_size / T;
      //     float *centroid = (float *)malloc(K * data_size_);
      //     std::vector<std::vector<tableint>> clusters_index2;
      //     std::vector<std::vector<tableint>> clusters;

      //     auto t0 = elapsed();

      //     index2->cluster_points(K, 2, 3, dim, layer_node_for_index2[level], centroid, clusters_index2);
      //     // TODO: assign

      //     index1->assign_points_to_clusters(0, K, 5000, 1, dim, centroid, clusters);
      //     printf("[%.3f s] build cluster\n", elapsed() - t0);

      //     t0 = elapsed();
      //     for (int i = 0; i < K; i++)
      //     {
      //         // auto t0 = elapsed(), t1 = 0.0;
      //         tableint currObj = clusters[i][0];
      //         std::unordered_set<tableint> cluster_set(clusters[i].begin(), clusters[i].end());
      //         for (int j = 0; j < clusters_index2[i].size(); j++)
      //         {
      //             tableint old_c = clusters_index2[i][j];
      //             tableint new_c = old_c + element_count_for_index1;
      //             const void *data_point = getDataByInternalId(new_c);
      //             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = index1->searchBaseLayerSubgroup(currObj, data_point, level, &cluster_set, true);
      //             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> search_candidates;
      //             std::unordered_set<tableint> seen_candidates;
      //             ll_cur = get_linklist_at_level(old_c, level);
      //             size = getListCount(ll_cur);
      //             data = (tableint *)(ll_cur + 1);
      //             distData = (dist_t *)get_dist_at_level(old_c, level);
      //             for (size_t i = 0; i < size; i++)
      //             {
      //                 top_candidates.emplace(distData[i], data[i]);
      //                 seen_candidates.insert(data[i]);
      //             }
      //             // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = index1->searchBaseLayer(currObj, data_point, level);
      //             ll_cur = index2->get_linklist_at_level(old_c, level);
      //             size = index2->getListCount(ll_cur);
      //             data = (tableint *)(ll_cur + 1);
      //             distData = (dist_t *)index2->get_dist_at_level(old_c, level);
      //             for (size_t i = 0; i < size; i++)
      //             {
      //                 if (seen_candidates.find(data[i] + element_count_for_index1) != seen_candidates.end())
      //                     continue;
      //                 top_candidates.emplace(distData[i], data[i] + element_count_for_index1);
      //                 seen_candidates.insert(data[i] + element_count_for_index1);
      //             }
      //             while (!top_candidates.empty())
      //             {
      //                 if (seen_candidates.find(top_candidates.top().second) == seen_candidates.end())
      //                 {
      //                     search_candidates.push(top_candidates.top());
      //                     seen_candidates.insert(top_candidates.top().second);
      //                 }
      //                 top_candidates.pop();
      //             }

      //             // if (epDeleted)
      //             // {
      //             //     top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
      //             //     if (top_candidates.size() > ef_construction_)
      //             //         top_candidates.pop();
      //             // }
      //             mutuallyConnectNewElementSubgroup(new_c, element_count_for_index1, search_candidates, level, false);
      //         }
      //         // printf("[%.3f s] insert cluster %d\n", elapsed() - t0, i);
      //     }
      //     printf("[%.3f s] insert\n", elapsed() - t0);
      // }
    }

    // TODO: approach 2 : randomly select K points from index2 and assign each point to index1
    // TODO: bug: I directly copy the edge data into new index, but thei internal id is incorrect
    {
      // memcpy(data_level0_memory_ + element_count_for_index1 * size_data_per_element_, index2->data_level0_memory_, element_count_for_index2 * size_data_per_element_);
      // memcpy(dist_level0_memory_ + element_count_for_index1 * size_dist_per_element_, index2->dist_level0_memory_, element_count_for_index2 * size_dist_per_element_);

      // // for (int level = maxlevel_; level >= 0; level -= 1)
      // for (int level = 0; level >= 0; level -= 1)
      // {
      //     // TODO: check the performance of using faiss kmeans instead

      //     int selected_num = element_count_for_index2 /2 ;
      //     auto t0 = elapsed();

      //     // auto selectRandomPoints = [](int L, int R, int K)
      //     // {
      //     //     std::vector<int> random_selected;
      //     //     std::vector<int> numbers(R - L);
      //     //     std::generate(numbers.begin(), numbers.end(), [n = L]() mutable
      //     //                   { return n++; });
      //     //     std::mt19937 gen(std::random_device{}());
      //     //     random_selected.resize(K);
      //     //     std::sample(numbers.begin(), numbers.end(), random_selected.begin(), K, gen);

      //     //     return random_selected;
      //     // };
      //     // auto selected = selectRandomPoints(element_count_for_index1, element_count_for_index1 + element_count_for_index2, selected_num);

      //     FILE* f2 = fopen(
      //         "/home/jin467/github_download/VecDB/faiss/labels.bin", "rb");
      //     std::vector<int> selected = std::vector<int>(selected_num);
      //     fread(selected.data(), sizeof(int), selected_num, f2);

      //     printf("[%.3f s] build cluster\n", elapsed() - t0);

      //     t0 = elapsed();
      //     for (auto new_c : selected)
      //     {
      //         auto old_c = new_c - element_count_for_index1;
      //         const void *data_point = getDataByInternalId(new_c);
      //         memset(get_linklist0(new_c), 0, size_links_level0_);
      //         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = index1->searchBaseLayerSubgroup(enterpoint_node_, data_point, level, nullptr, true);

      //         linklistsizeint *ll_cur = index2->get_linklist_at_level(old_c, level);
      //         size_t size = index2->getListCount(ll_cur);
      //         tableint *data = (tableint *)(ll_cur + 1);
      //         dist_t *distData = (dist_t *)index2->get_dist_at_level(old_c, level);
      //         for (size_t i = 0; i < size; i++)
      //         {
      //             top_candidates.emplace(distData[i], data[i] + element_count_for_index1);
      //         }
      //         // if (epDeleted)
      //         // {
      //         //     top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
      //         //     if (top_candidates.size() > ef_construction_)
      //         //         top_candidates.pop();
      //         // }
      //         mutuallyConnectNewElementSubgroup(new_c, element_count_for_index1, top_candidates, level, false);
      //     }
      //     printf("[%.3f s] insert cluster\n", elapsed() - t0);
      // }
    }
    // TODO: approach 3 -

    {
      for (tableint old_c = 0; old_c < element_count_for_index2; old_c++)
      {
        tableint new_c = old_c + element_count_for_index1;
        memcpy(getDataByInternalId(new_c), index2->getDataByInternalId(old_c), data_size_);
        setExternalLabel(new_c, index2->getExternalLabel(old_c));
        int level = index2->element_levels_[old_c];
        element_levels_[new_c] = 0;
        // TODO: Now we assume inserted index2 put all its nodes on index1 level 0; We may test on other levels
      }
      // memcpy(data_level0_memory_ + element_count_for_index1 * size_data_per_element_, index2->data_level0_memory_, element_count_for_index2 * size_data_per_element_);
      // memcpy(dist_level0_memory_ + element_count_for_index1 * size_dist_per_element_, index2->dist_level0_memory_, element_count_for_index2 * size_dist_per_element_);

      // TODO: add a logic for layer increment

      linklistsizeint *ll_cur;
      size_t size;
      tableint *data;
      dist_t *distData;
      // for (int level = maxlevel_; level >= 0; level -= 1)
      for (int level = 0; level >= 0; level -= 1)
      {
        // TODO: check the performance of using faiss kmeans instead // complie faiss library and use the -lfaiss flag when compiling the code
        int T = 5;
        int train_size = index2->cur_element_count;
        int K = train_size / T;
        float *centroid = (float *)malloc(K * data_size_);
        std::vector<std::vector<tableint>> clusters;

        auto t0 = elapsed();
        std::vector<float> trainvecs(train_size * dim);
        for (int i = 0; i < train_size; i++)
        {
          memcpy(trainvecs.data() + i * dim, index2->getDataByInternalId(i), dim * sizeof(dist_t));
        }
        faiss::IndexFlatL2 quantizer(dim); // the other index
        // faiss::IndexFlatIP quantizer(d); // the other index
        faiss::IndexIVFFlat index(&quantizer, dim, K);
        index.metric_type = faiss::METRIC_L2;
        faiss::IndexHNSWFlat cluster_trainer(dim, 16);
        cluster_trainer.hnsw.efConstruction = 100;
        cluster_trainer.hnsw.efSearch = 200;
        index.clustering_index = &cluster_trainer;
        assert(!index.is_trained);
        index.train(train_size, trainvecs.data());
        assert(index.is_trained);
        printf("[%.3f s] finished clustering\n", elapsed() - t0);

        FILE *f = fopen(
            "/home/jin467/github_download/faiss/centroids.bin", "rb");
        size_t nr = fread(centroid, sizeof(float), K * dim, f);
        fclose(f);
        // index1->assign_points_to_clusters(0, K, 5000, 1, dim, centroid, clusters);
        for (int i = 0; i < K; i++)
        {
          std::priority_queue<std::pair<dist_t, labeltype>> result = index1->searchKnn(centroid + i * dim, 5 * T);
          clusters.push_back(std::vector<tableint>(result.size()));
          int j = 0;
          while (result.size())
          {
            clusters[i][j++] = result.top().second;
            result.pop();
          }
        }

        // index2->cluster_points(K, 2, 3, dim, layer_node_for_index2[level], centroid, clusters_index2);

        // file format 1: arrange by centroid, in each line, collect the pointID in the cluster
        {
          //     std::vector<std::vector<tableint>> clusters_index2;
          //     f = fopen(
          //         "/home/jin467/github_download/faiss/assign.bin", "rb");
          //     for (int i = 0; i < K; i++)
          //     {
          //         int length;
          //         fread(&length, sizeof(int), 1, f);
          //         clusters_index2.push_back(std::vector<tableint>(length));
          //         fread(clusters_index2[i].data(), sizeof(tableint), length, f);
          //     }
          //     fclose(f);
          //     printf("[%.3f s] build cluster\n", elapsed() - t0);

          //     t0 = elapsed();
          //     int valid_add = 5;
          //     int offset_index2 = index1->cur_element_count;
          //     // FILE* fw = fopen("/home/jin467/github_download/hnswlib/hnswlib/catch.bin", "wb");
          //     for (int i = 0; i < K; i++)
          //     {
          //         // auto t0 = elapsed(), t1 = 0.0;
          //         // tableint currObj = clusters[i][0];
          //         // std::unordered_set<tableint> cluster_set(clusters[i].begin(), clusters[i].end());
          //         int cnt_flag = 0;
          //         for (int j = 0; j < clusters_index2[i].size(); j++)
          //         {
          //             tableint old_c = clusters_index2[i][j];
          //             tableint new_c = old_c + element_count_for_index1;
          //             const void *data_point = getDataByInternalId(new_c);
          //             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
          //             // std::unordered_set<tableint> seen_candidates;

          //             for (int k = 0; k < clusters[i].size(); k++)
          //             {
          //                 top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(clusters[i][k]), dist_func_param_), clusters[i][k]);
          //                 if (top_candidates.size() > valid_add)
          //                 {
          //                     top_candidates.pop();
          //                 }
          //             };
          //             // TODO: change from cover link info to update link info
          //             ll_cur = index2->get_linklist_at_level(old_c, level);
          //             size = index2->getListCount(ll_cur);
          //             data = (tableint *)(ll_cur + 1);
          //             distData = (dist_t *)index2->get_dist_at_level(old_c, level);
          //             for (size_t i = 0; i < size; i++)
          //             {
          //                 top_candidates.emplace(distData[i], data[i] + offset_index2);
          //                 // seen_candidates.insert(data[i]);
          //             }

          //             mutuallyConnectNewElementSubgroup(new_c, element_count_for_index1, top_candidates, level, false);
          //         }
        }

        // file format 2: arrange by point. tell the #replica first, and for each point collect its clusterIDs
        {
          std::vector<std::vector<tableint>> clusters_index2;
          f = fopen(
              "/home/jin467/github_download/faiss/assign.bin", "rb");
          int length;
          fread(&length, sizeof(int), 1, f);
          for (int i = 0; i < index2->cur_element_count; i++)
          {
            clusters_index2.push_back(std::vector<tableint>(length));
            fread(clusters_index2[i].data(), sizeof(tableint), length, f);
          }
          fclose(f);
          printf("[%.3f s] build cluster\n", elapsed() - t0);

          t0 = elapsed();
          int valid_add = 5;
          int offset_index2 = index1->cur_element_count;
          std::unordered_map<tableint, std::vector<std::pair<tableint, dist_t>>> insertion_list;
          for (int old_c = 0; old_c < index2->cur_element_count; old_c++)
          {
            std::unordered_set<tableint> cluster_set;
            // tableint old_c = clusters_index2[i][j];
            tableint new_c = old_c + element_count_for_index1;
            const void *data_point = getDataByInternalId(new_c);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            for (tableint cluster_id : clusters_index2[old_c])
            {
              for (tableint index1_id : clusters[cluster_id])
              {
                if (cluster_set.find(index1_id) != cluster_set.end())
                  continue;
                cluster_set.insert(index1_id);
                top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(index1_id), dist_func_param_), index1_id);
                if (top_candidates.size() > valid_add)
                {
                  top_candidates.pop();
                }
              }
            }
            // TODO: change from cover link info to update link info
            ll_cur = index2->get_linklist_at_level(old_c, level);
            size = index2->getListCount(ll_cur);
            data = (tableint *)(ll_cur + 1);
            distData = (dist_t *)index2->get_dist_at_level(old_c, level);
            for (size_t i = 0; i < size; i++)
            {
              top_candidates.emplace(distData[i], data[i] + offset_index2);
            }

            // mutuallyConnectNewElementSubgroup(new_c, element_count_for_index1, top_candidates, level, false);
            // TODO: update for parallelism

            size_t Mcurmax = level ? maxM_ : maxM0_;

            getNeighborsByHeuristic2(top_candidates, M_, false);

            std::vector<tableint> selectedNeighbors;
            std::vector<dist_t> selectedNeighborsDist;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0)
            {
              selectedNeighbors.push_back(top_candidates.top().second);
              selectedNeighborsDist.push_back(top_candidates.top().first);
              top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
              ll_cur = get_linklist_at_level(new_c, level);
              setListCount(ll_cur, selectedNeighbors.size());
              tableint *data = (tableint *)(ll_cur + 1);
              dist_t *distData = (dist_t *)get_dist_at_level(new_c, level);

              for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
              {
                data[idx] = selectedNeighbors[idx];
                distData[idx] = selectedNeighborsDist[idx];
                if (selectedNeighbors[idx] < element_count_for_index1)
                  insertion_list[selectedNeighbors[idx]].emplace_back(new_c, selectedNeighborsDist[idx]);
              }
            }
          }
          for (const auto &kv : insertion_list)
          {
            tableint key = kv.first;
            const std::vector<std::pair<tableint, dist_t>> &vec = kv.second;

            for (const auto &p : vec)
            {
              tableint inner_key = p.first;
              dist_t dist = p.second;
            }
          }
        }

        printf("[%.3f s] insert\n", elapsed() - t0);
      }
    }

    // TODO: method 1: merge only on last layer
  }

}