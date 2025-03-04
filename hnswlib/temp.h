template <typename dist_t>
bool compareIndex(HierarchicalNSW<dist_t> *a, HierarchicalNSW<dist_t> *b)
{
  if (a->cur_element_count != b->cur_element_count)
  {
    return a->cur_element_count > b->cur_element_count;
  }
  else
  {
    return a->maxlevel_ > b->maxlevel_;
  }
}

void assign_points_to_clusters(
  const std::vector<int> &layer_nodes,
  int K,
  int T,
  int l,
  size_t dim,
  float *centroids,
  std::vector<std::vector<tableint>> &clusters)/* kmeans */
{
  clusters.clear();
  clusters.resize(K);

  for (int point : layer_nodes)
  {
      std::vector<std::pair<float, int>> distances;

      float *point_data = (float *)(getDataByInternalId(point));

      for (int j = 0; j < K; j++)
      {
          float dist = fstdistfunc_((float *)(centroids + j * dim), point_data, dist_func_param_);
          distances.push_back({dist, j});
      }

      std::sort(distances.begin(), distances.end());

      float min_dist = distances.front().first;
      int assigned_count = 0;
      int full_clusters = 0;
      for (const auto &[dist, cluster_id] : distances)
      {
          if (dist > 2 * min_dist) 
              break;

          if (clusters[cluster_id].size() < T) 
          {
              clusters[cluster_id].push_back(point);
              assigned_count++;
              if (assigned_count >= l)
                  break;
          }
          else{
              full_clusters++;
          }
      }
      if (full_clusters == K)
      {
          std::cout << "Warning: Point " << point << " could not be assigned (all clusters full)" << std::endl;
          return;
      }
  }
}

const double PI = 3.14159265358979323846;
double gaussianPDF(const std::vector<double> &x, const std::vector<double> &mean, const std::vector<std::vector<double>> &cov)
{
  int dim = x.size();
  double det = 1.0;
  for (int i = 0; i < dim; ++i)
  {
    det *= cov[i][i];
  }
  double norm_const = 1.0 / (std::pow(2 * PI, dim / 2.0) * std::sqrt(det));
  std::vector<double> diff(dim);
  for (int i = 0; i < dim; ++i)
  {
    diff[i] = x[i] - mean[i];
  }
  double exponent = 0.0;
  for (int i = 0; i < dim; ++i)
  {
    exponent += (diff[i] * diff[i]) / cov[i][i];
  }
  return norm_const * std::exp(-0.5 * exponent);
}

void gaussianMixtureModel(
    const std::vector<int> &data_indices,             // 每个点的索引
    HierarchicalNSW<float> *index,                    // HNSW 索引对象，用于访问数据点
    int numClusters,                                  // 聚类数
    int maxIterations,                                // 最大迭代次数
    std::vector<std::vector<int>> &clusterAssignments // 每个点的聚类分配
)
{
  int numPoints = data_indices.size();

  // 特殊情况处理：如果只有一个点，直接返回这个点属于一个聚类
  if (numPoints == 1)
  {
    clusterAssignments.resize(1);
    clusterAssignments[0].push_back(data_indices[0]);
    return;
  }

  // 根据聚类数量确定每个点最多所属的聚类数
  int L = 4; // 默认 4 个聚类
  if (numClusters < 8)
  {
    L = 2;
  }
  else if (numClusters < 27)
  {
    L = 3;
  }

  int dim = sizeof(index->getDataByInternalId(data_indices[0])) / sizeof(double);

  // 初始化 GMM 参数
  std::vector<std::vector<double>> means(numClusters, std::vector<double>(dim));
  std::vector<std::vector<std::vector<double>>> covariances(numClusters, std::vector<std::vector<double>>(dim, std::vector<double>(dim)));
  std::vector<double> weights(numClusters, 1.0 / numClusters);                                         // 初始均匀权重
  std::vector<std::vector<double>> responsibilities(numPoints, std::vector<double>(numClusters, 0.0)); // 责任矩阵

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, numPoints - 1);

  // 初始化均值（随机选择数据点作为初始均值）
  for (int c = 0; c < numClusters; ++c)
  {
    int random_idx = dis(gen);
    char *data_point = index->getDataByInternalId(data_indices[random_idx]);
    means[c].assign(data_point, data_point + dim); // 使用索引获取特征数据
  }

  // GMM 迭代过程
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    // E 步骤：计算每个点属于每个聚类的概率（责任）
    for (int i = 0; i < numPoints; ++i)
    {
      double sumResponsibility = 0.0;
      char *data_point = index->getDataByInternalId(data_indices[i]);
      for (int c = 0; c < numClusters; ++c)
      {
        responsibilities[i][c] = weights[c] * gaussianPDF(data_point, means[c], covariances[c]);
        sumResponsibility += responsibilities[i][c];
      }

      // 归一化责任
      for (int c = 0; c < numClusters; ++c)
      {
        responsibilities[i][c] /= sumResponsibility;
      }
    }

    // M 步骤：更新 GMM 参数
    for (int c = 0; c < numClusters; ++c)
    {
      double sumResponsibility = 0.0;
      std::vector<double> newMean(dim, 0.0);
      std::vector<std::vector<double>> newCovariance(dim, std::vector<double>(dim, 0.0));

      // 更新均值和协方差
      for (int i = 0; i < numPoints; ++i)
      {
        double responsibility = responsibilities[i][c];
        sumResponsibility += responsibility;

        char *data_point = index->getDataByInternalId(data_indices[i]);
        for (int d = 0; d < dim; ++d)
        {
          newMean[d] += responsibility * data_point[d];
        }
      }

      // 计算新的均值
      for (int d = 0; d < dim; ++d)
      {
        newMean[d] /= sumResponsibility;
      }

      // 计算新的协方差矩阵
      for (int i = 0; i < numPoints; ++i)
      {
        std::vector<double> diff(dim);
        char *data_point = index->getDataByInternalId(data_indices[i]);
        for (int d = 0; d < dim; ++d)
        {
          diff[d] = data_point[d] - newMean[d];
        }

        for (int d = 0; d < dim; ++d)
        {
          newCovariance[d][d] += responsibilities[i][c] * diff[d] * diff[d];
        }
      }

      for (int d = 0; d < dim; ++d)
      {
        newCovariance[d][d] /= sumResponsibility;
      }

      means[c] = newMean;
      covariances[c] = newCovariance;
      weights[c] = sumResponsibility / numPoints;
    }
  }

  // 聚类分配：根据责任值进行分配
  clusterAssignments.resize(numClusters);
  for (int i = 0; i < numPoints; ++i)
  {
    std::vector<std::pair<double, int>> membershipPairs;
    for (int c = 0; c < numClusters; ++c)
    {
      membershipPairs.push_back(std::make_pair(responsibilities[i][c], c));
    }
    std::sort(membershipPairs.begin(), membershipPairs.end(), std::greater<std::pair<double, int>>());

    // 每个点分配给 L 个概率最高的聚类
    for (int j = 0; j < L; ++j)
    {
      clusterAssignments[membershipPairs[j].second].push_back(data_indices[i]);
    }
  }
}

void fuzzyCMeans(
    const std::vector<int> &data_indices,
    HierarchicalNSW<float> *index,
    int layer,
    int maxIterations,
    int numClusters,
    int cluster_cnt,
    std::vector<std::vector<int>> &clusterAssignments)
{
  int numPoints = data_indices.size();
  int dim = *(int *)(index->dist_func_param_);

  std::vector<std::vector<double>> memberships(numPoints, std::vector<double>(numClusters, 0.0));
  std::vector<std::vector<double>> centroids(numClusters, std::vector<double>(dim));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  for (int i = 0; i < numPoints; ++i)
  {
    double sum = 0.0;
    for (int j = 0; j < numClusters; ++j)
    {
      memberships[i][j] = dis(gen);
      sum += memberships[i][j];
    }
    for (int j = 0; j < numClusters; ++j)
    {
      memberships[i][j] /= sum;
    }
  }

  for (int iter = 0; iter < maxIterations; ++iter)
  {
    for (int c = 0; c < numClusters; ++c)
    {
      for (int dim = 0; dim < centroids[0].size(); ++dim)
      {
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = 0; i < numPoints; ++i)
        {
          double weight = std::pow(memberships[i][c], 2);
          char *data_point = index->getDataByInternalId(data_indices[i]);
          numerator += weight * data_point[dim];
          denominator += weight;
        }
        centroids[c][dim] = numerator / denominator;
      }
    }

    for (int i = 0; i < numPoints; ++i)
    {
      std::vector<std::pair<double, int>> distToClusters(numClusters);
      char *data_point = index->getDataByInternalId(data_indices[i]);
      for (int c = 0; c < numClusters; ++c)
      {
        double dist = 0.0;
        for (int dim = 0; dim < centroids[0].size(); ++dim)
        {
          dist += std::pow(data_point[dim] - centroids[c][dim], 2);
        }
        dist = std::sqrt(dist);
        distToClusters[c] = std::make_pair(dist, c);
      }
      std::sort(distToClusters.begin(), distToClusters.end());

      for (int j = 0; j < numClusters; ++j)
      {
        memberships[i][distToClusters[j].second] = (j < cluster_cnt) ? 1.0 / distToClusters[j].first : 0.0;
      }
    }
  }

  clusterAssignments.resize(numClusters);
  for (int i = 0; i < numPoints; ++i)
  {
    std::vector<std::pair<double, int>> membershipPairs;
    for (int c = 0; c < numClusters; ++c)
    {
      membershipPairs.push_back(std::make_pair(memberships[i][c], c));
    }
    std::sort(membershipPairs.begin(), membershipPairs.end(), std::greater<std::pair<double, int>>());

    for (int j = 0; j < cluster_cnt; ++j)
    {
      clusterAssignments[membershipPairs[j].second].push_back(data_indices[i]);
    }
  }
}

std::unordered_map<int, std::unordered_map<int, std::vector<int>>> clusterPointsInLayer(
    HierarchicalNSW<float> **indexList,
    int indexList_size,
    int layer,
    std::vector<std::vector<int>> &mergedDataPoints,
    int cluster_cnt = 4)
{
  std::unordered_map<int, std::unordered_map<int, std::vector<int>>> clusteringResult;
  for (int idx = 0; idx < indexList_size; ++idx)
  {
    std::vector<int> &pointsInLayer = mergedDataPoints[idx];
    int numPointsInLayer = mergedDataPoints[idx].size();
    int numClusters = std::min(int(numPointsInLayer / 30), static_cast<int>(std::sqrt(numPointsInLayer)));
    if (numClusters == 1)
    {
      clusteringResult[idx][0] = mergedDataPoints[idx];
    }
    else
    {
      std::vector<std::vector<int>> clusterAssignments;
      if (numClusters < 8)
      {
        cluster_cnt = 2;
      }
      else if (numClusters < 27)
      {
        cluster_cnt = 3;
      }
      fuzzyCMeans(mergedDataPoints[idx], indexList[idx], layer, 25, numClusters, cluster_cnt, clusterAssignments);
      for (int c = 0; c < numClusters; ++c)
      {
        clusteringResult[idx][c] = clusterAssignments[c];
        // std::cout << "Cluster for idx " << idx << ", cluster " << c << ": ";
        // for (int val : clusterAssignments[c])
        // {
        //     std::cout << val << " ";
        // }
        // std::cout << std::endl;
      }
    }
  }
  return clusteringResult;
}

template <typename dist_t>
HierarchicalNSW<dist_t> *HNSWMergerOnMultiple(
    HierarchicalNSW<dist_t> **indexList,
    int indexList_size,
    L2Space *space,
    size_t M = -1,
    size_t ef_construction = -1)
{
  // TODO: implement logic with deleted node in an index, by re-organize the internal label of nodes in an index

  // Initialize a new HNSW index
  // When calling this function, make sure index[0] element count is the largest one, and has the max-layer.
  std::sort(indexList, indexList + indexList_size, compareIndex<dist_t>);
  size_t new_M = 0;
  size_t new_ef_construction = 0;
  size_t max_elements = 0;
  int maxLevel = 0;
  size_t cur_element_count = 0;

  for (size_t i = 0; i < indexList_size; ++i)
  {
    HierarchicalNSW<dist_t> *currentIndex = indexList[i];
    new_M = std::max(currentIndex->M_, new_M);
    new_ef_construction = std::max(currentIndex->ef_construction_, new_ef_construction);
    max_elements += currentIndex->max_elements_;
    maxLevel = std::max(currentIndex->maxlevel_, maxLevel);
    cur_element_count += currentIndex->cur_element_count.load();
  }
  M = M == -1 ? new_M : M;
  ef_construction = ef_construction == -1 ? new_ef_construction : ef_construction;
  HierarchicalNSW<dist_t> *alg_hnsw = new HierarchicalNSW<dist_t>(space, max_elements, M, ef_construction);
  alg_hnsw->setMaxLevel(maxLevel);
  alg_hnsw->cur_element_count.store(cur_element_count);

  alg_hnsw->data_level0_memory_ = (char *)malloc(alg_hnsw->max_elements_ * alg_hnsw->size_data_per_element_);
  if (alg_hnsw->data_level0_memory_ == nullptr)
    throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");

  size_t max_element_count = 0;
  std::vector<size_t> element_counts(indexList_size);
  std::vector<size_t> element_counts_offset(indexList_size);
  std::vector<std::vector<std::vector<int>>> layer_nodes(indexList_size, std::vector<std::vector<int>>(maxLevel + 2));
  for (int i = 0; i < indexList_size; ++i)
  {
    element_counts[i] = indexList[i]->getCurrentElementCount();
    if (i == 0)
      element_counts_offset[i] = 0;
    else
      element_counts_offset[i] = element_counts[i] + element_counts_offset[i - 1];
    searchNodeOnEachLayer(indexList[i], layer_nodes[i]);
    max_element_count = std::max(max_element_count, element_counts[i]);
  }
  std::vector<std::vector<int>> entry_point_collect(max_element_count, std::vector<int>(indexList_size, -1));
  std::vector<std::vector<int>> mergedDataPoints(indexList_size);
  if (std::accumulate(layer_nodes.begin(), layer_nodes.end(), 0,
                      [&](size_t sum, const std::vector<std::vector<int>> &layers)
                      {
                        return sum + layers[maxLevel].size();
                      }) > M)
  {
    std::vector<int> newLayer;
    auto filter_and_remove = [&](std::vector<int> &layer_nodes, HierarchicalNSW<dist_t> *index,
                                 std::vector<int> &mergedDataPoints, int offset)
    {
      auto it = layer_nodes.begin();
      std::uniform_real_distribution<double> distribution(0.0, 1.0);
      while (it != layer_nodes.end())
      {
        if (distribution(alg_hnsw->level_generator_) < 1.0 / M)
        {
          newLayer.push_back((*it) + offset);

          tableint new_c = *it;
          alg_hnsw->linkLists_[new_c + offset] = (char *)malloc(alg_hnsw->size_links_per_element_ * (maxLevel + 1) + 1);
          if (alg_hnsw->linkLists_[new_c + offset] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
          memset(alg_hnsw->linkLists_[new_c + offset], 0, alg_hnsw->size_links_per_element_ * (maxLevel + 1) + 1);
          alg_hnsw->element_levels_[new_c + offset] = (maxLevel + 1);

          memcpy(alg_hnsw->getDataByInternalId(new_c + offset), index->getDataByInternalId(new_c), alg_hnsw->data_size_);
          alg_hnsw->setExternalLabel(new_c + offset, index->getExternalLabel(new_c));
          mergedDataPoints.push_back(new_c);

          it = layer_nodes.erase(it);
        }
        else
        {
          ++it;
        }
      }
    };
    for (int i = 0; i < indexList_size; ++i)
      filter_and_remove(layer_nodes[i][maxLevel], indexList[i], mergedDataPoints[i], element_counts_offset[i]);
    if (!newLayer.empty())
    {
      for (int newLayerIter = 0; newLayerIter < newLayer.size(); ++newLayerIter)
      {
        linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(newLayer[newLayerIter], maxLevel + 1);
        alg_hnsw->setListCount(ll_cur, newLayer.size() - 1);
        tableint *data = (tableint *)(ll_cur + 1);
        for (size_t it = 0, idx = 0; it < newLayer.size(); ++it)
        {
          if (it == newLayerIter)
            continue;
          data[idx++] = newLayer[it];
        }
      }
      alg_hnsw->setMaxLevel(maxLevel + 1);
      alg_hnsw->enterpoint_node_ = newLayer[0];
    }
    else
    {
      alg_hnsw->enterpoint_node_ = layer_nodes[0][maxLevel][0];
    }
  }
  else
  {
    alg_hnsw->enterpoint_node_ = layer_nodes[0][maxLevel][0];
  }
  double t_total = 0.0, t_level, t_cnt;
  for (int level = maxLevel; level >= 0; level -= 1)
  {
    t_level = 0.0;
    if (level > 0)
    {
      for (int i = 0; i < indexList_size; ++i)
      {
        for (int id = 0; id < layer_nodes[i][level].size(); ++id)
        {
          tableint new_c = layer_nodes[i][level][id] + element_counts_offset[i];
          alg_hnsw->linkLists_[new_c] = (char *)malloc(alg_hnsw->size_links_per_element_ * level + 1);
          if (alg_hnsw->linkLists_[new_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
          memset(alg_hnsw->linkLists_[new_c], 0, alg_hnsw->size_links_per_element_ * level + 1);
          alg_hnsw->element_levels_[new_c] = level;
        }
      }
    }

    for (int i = 0; i < indexList_size; ++i)
    {
      for (int id = 0; id < layer_nodes[i][level].size(); ++id)
      {
        tableint new_c = layer_nodes[i][level][id];
        char *data_point = indexList[i]->getDataByInternalId(new_c);
        memcpy(alg_hnsw->getDataByInternalId(new_c + element_counts_offset[i]), data_point, alg_hnsw->data_size_);
        alg_hnsw->setExternalLabel(new_c + element_counts_offset[i], indexList[i]->getExternalLabel(new_c));
        mergedDataPoints[i].push_back(new_c);
      }
    }

    int all_other_empty = 0;
    for (int i = 0; i < indexList_size; ++i)
    {
      if (layer_nodes[i][level].size() == 0)
        continue;
      all_other_empty++;
    }
    if (all_other_empty == 1)
      for (int i = 0; i < indexList_size; ++i)
        if (layer_nodes[i][level].size() > 0)
        {
          deepCopyOneLayerOnIndex(indexList[i], alg_hnsw, level, element_counts_offset[i], layer_nodes[i]);
          break;
        }
    // clusterPointsInLayer(indexList, indexList_size, level, mergedDataPoints, 4);
    int cnt = 0;
    for (int i = 0; i < indexList_size; ++i)
    {
      for (int iter = 0; iter < mergedDataPoints[i].size(); ++iter, ++cnt)
      {
        tableint cur_c = mergedDataPoints[i][iter];
        char *data_point = indexList[i]->getDataByInternalId(cur_c);
        t_cnt = elapsed();
        alg_hnsw->mergeIndexBasedOnAllOtherIndexConnection(indexList,
                                                           i,
                                                           indexList_size,
                                                           cur_c,
                                                           data_point,
                                                           element_counts_offset,
                                                           level,
                                                           entry_point_collect[cur_c]);
        t_level += elapsed() - t_cnt;
      }
    }
    printf("level %d, cnt %d, time %.3f, avg %.6f\n", level, cnt, t_level, t_level / cnt);
    t_total += t_level;
    printf("total cost %.3f\n", t_total);
  }
  std::cout << "函数累计总耗时: " << total_duration.count() << " 秒" << std::endl;
  return alg_hnsw;
} // end of HNSWMergerOnMultiple

void HierarchicalNSW::mergeIndexBasedOnAllOtherIndexConnection(
    HierarchicalNSW<dist_t> **indexList,
    int cur_index,
    int indexList_size,
    tableint cur_c,
    char *data_point,
    std::vector<size_t> &element_counts_offset,
    int level,
    std::vector<int> &last_entry_points /* in & out */
)
{
  HierarchicalNSW<dist_t> *mainIndex = indexList[cur_index];
  std::vector<tableint> Candidates;
  for (int i = 0; i < indexList_size; i++)
  {
    if (i == cur_index)
      continue;
    HierarchicalNSW<dist_t> *currentIndex = indexList[i];
    int higherLevel = (last_entry_points[i] == -1) ? currentIndex->maxlevel_ : level;

    tableint entry_point = currentIndex->search2Layer(data_point, last_entry_points[i], higherLevel, level);
    // tableint entry_point = MEASURE_FUNCTION_TIME([&]()
    //                                              { return currentIndex->search2Layer(data_point, last_entry_points[i], higherLevel, level); });
    Candidates.push_back(entry_point + element_counts_offset[i]);

    linklistsizeint *ll_cur = currentIndex->get_linklist_at_level(entry_point, level);
    size_t linklistCount = currentIndex->getListCount(ll_cur);
    tableint *data = (tableint *)(ll_cur + 1);
    for (size_t iter = 0; iter < linklistCount; iter++)
    {
      Candidates.push_back(data[iter] + element_counts_offset[i]);
    }

    ll_cur = mainIndex->get_linklist_at_level(cur_c, level);
    linklistCount = mainIndex->getListCount(ll_cur);
    data = (tableint *)(ll_cur + 1);
    for (size_t iter = 0; iter < linklistCount; iter++)
    {
      Candidates.push_back(data[iter] + element_counts_offset[cur_index]);
    }

    if (level != element_levels_[cur_c + element_counts_offset[cur_index]])
    {
      ll_cur = get_linklist_at_level(cur_c + element_counts_offset[cur_index], level + 1);
      linklistCount = getListCount(ll_cur);
      Candidates.insert(Candidates.end(), (tableint *)(ll_cur + 1), (tableint *)(ll_cur + 1) + linklistCount);
    }
    last_entry_points[i] = entry_point;
  }

  std::unordered_set<tableint> seen;
  std::vector<tableint> result;
  for (const auto &val : Candidates)
  {
    if (seen.find(val) == seen.end() && val != cur_c + element_counts_offset[cur_index])
    {
      seen.insert(val);
      result.push_back(val);
    }
  }

  // get all distance data and then
  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
  dist_t dist = fstdistfunc_(data_point, getDataByInternalId(result[0]), dist_func_param_);
  // dist_t dist = MEASURE_FUNCTION_TIME(fstdistfunc_,data_point, getDataByInternalId(result[0]), dist_func_param_);
#ifdef USE_SSE
  _mm_prefetch(result.data(), _MM_HINT_T0);
  _mm_prefetch(result.data() + 1, _MM_HINT_T0);
#endif
  top_candidates.emplace(dist, result[0]);
  dist_t lowerBound = dist;
  candidateSet.emplace(-dist, result[0]);

  for (size_t j = 1; j < result.size(); j++)
  {
    tableint candidate_id = result[j];
#ifdef USE_SSE
    _mm_prefetch(result.data() + j + 1, _MM_HINT_T0);
#endif
    char *currObj1 = getDataByInternalId(candidate_id);
    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
    // dist_t dist1 = MEASURE_FUNCTION_TIME(fstdistfunc_,data_point, currObj1, dist_func_param_);
    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
    {
      candidateSet.emplace(-dist1, candidate_id);
      top_candidates.emplace(dist1, candidate_id);
      if (top_candidates.size() > ef_construction_)
        top_candidates.pop();
      if (!top_candidates.empty())
        lowerBound = top_candidates.top().first;
    }
  }
  size_t Mcurmax = level ? maxM_ : maxM0_;
  // auto start = std::chrono::high_resolution_clock::now();
  getNeighborsByHeuristic2(top_candidates, Mcurmax);
  // auto end = std::chrono::high_resolution_clock::now();
  // total_duration += end - start;

  linklistsizeint *ll_cur = get_linklist_at_level(cur_c + element_counts_offset[cur_index], level);
  setListCount(ll_cur, top_candidates.size());
  tableint *data = (tableint *)(ll_cur + 1);
  for (size_t idx = 0; top_candidates.size() > 0; idx++)
  {
    data[idx] = top_candidates.top().second;
    top_candidates.pop();
  }
}

std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
searchBaseLayerForMerge(std::vector<tableint> &ep_ids, const void *data_point, int layer, int offset, tableint cur_c)
{
  VisitedList *vl = visited_list_pool_->getFreeVisitedList();
  vl_type *visited_array = vl->mass;
  vl_type visited_array_tag = vl->curV;

  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

  dist_t lowerBound = std::numeric_limits<dist_t>::min();
  for (int iter = 0; iter < ep_ids.size(); iter++)
  {
    tableint ep_id = ep_ids[iter];
    if (!isMarkedDeleted(ep_id))
    {
      dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
      top_candidates.emplace(dist, ep_id + offset);
      lowerBound = std::max(dist, lowerBound);
      candidateSet.emplace(-dist, ep_id + offset);
    }
    visited_array[ep_id] = visited_array_tag;
  }
  if (lowerBound == std::numeric_limits<dist_t>::min())
  {
    lowerBound = std::numeric_limits<dist_t>::max();
    candidateSet.emplace(-lowerBound, ep_ids[0] + offset);
  }

  while (!candidateSet.empty())
  {
    std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
    if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_)
    {
      break;
    }
    candidateSet.pop();

    tableint curNodeNum = curr_el_pair.second - offset;

    std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

    int *data; // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
    if (layer == 0)
    {
      data = (int *)get_linklist0(curNodeNum);
    }
    else
    {
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

    for (size_t j = 0; j < size; j++)
    {
      tableint candidate_id = *(datal + j);
#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
      if (visited_array[candidate_id] == visited_array_tag)
        continue;
      visited_array[candidate_id] = visited_array_tag;
      char *currObj1 = (getDataByInternalId(candidate_id));

      dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
      if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
      {
        candidateSet.emplace(-dist1, candidate_id + offset);
#ifdef USE_SSE
        _mm_prefetch(getDataByInternalId(candidateSet.top().second - offset), _MM_HINT_T0);
#endif

        if (!isMarkedDeleted(candidate_id))
          top_candidates.emplace(dist1, candidate_id + offset);

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

void mergeLevel0(
    HierarchicalNSW<dist_t> *index1,
    HierarchicalNSW<dist_t> *index2,
    tableint cur_c,
    char *data_point,
    int offset_index1,
    int offset_index2,
    int &last_entry_point /* in & out */
)
{
  tableint entry_point;
  int higherLevel = 0;
  if (last_entry_point == -1)
  {
    higherLevel = index2->maxlevel_;
  }
  entry_point = index2->search2Layer(data_point,
                                     last_entry_point,
                                     higherLevel,
                                     0);
  std::vector<tableint> epids;
  epids.push_back(entry_point);
  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidate = index2->searchBaseLayerForMerge(epids, data_point, 0, offset_index2, cur_c + offset_index1);

  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
  while (!top_candidate.empty())
  {
    top_candidates.emplace(top_candidate.top().first, top_candidate.top().second);
    top_candidate.pop();
  }

  linklistsizeint *ll_cur = index1->get_linklist0(cur_c);
  size_t linklistCount = index1->getListCount(ll_cur);
  tableint *data = (tableint *)(ll_cur + 1);
  for (size_t iter = 0; iter < linklistCount; iter++)
  {
    char *currObj1 = (getDataByInternalId(data[iter] + offset_index1));
    dist_t dist1 = index1->fstdistfunc_(data_point, currObj1, index1->dist_func_param_);

    top_candidates.emplace(-dist1, data[iter] + offset_index1);
  }

  size_t Mcurmax = maxM0_;
  getNeighborsByHeuristic2(top_candidates, M_);
  if (top_candidates.size() > Mcurmax)
    throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");
  std::vector<tableint> selectedNeighbors;
  selectedNeighbors.reserve(Mcurmax);
  while (top_candidates.size() > 0)
  {
    selectedNeighbors.push_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::unique_lock<std::mutex> lock(link_list_locks_[cur_c + offset_index1], std::defer_lock);

  ll_cur = get_linklist0(cur_c + offset_index1);
  setListCount(ll_cur, selectedNeighbors.size());
  data = (tableint *)(ll_cur + 1);
  for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
  {
    data[idx] = selectedNeighbors[idx];
  }
}

void mergeIndex1BasedOnIndex2Connection2(
    HierarchicalNSW<dist_t> *index1,
    HierarchicalNSW<dist_t> *index2,
    tableint cur_c,
    char *data_point,
    int offset_index1,
    int offset_index2,
    int level,
    int &last_entry_point /* in & out */
)
{

  tableint entry_point;
  int higherLevel = level;
  if (last_entry_point == -1)
  {
    higherLevel = index2->maxlevel_;
  }
  entry_point = index2->search2Layer(data_point,
                                     last_entry_point,
                                     higherLevel,
                                     level);

  // Find ONE entry point with its neighbors
  // TODO: 1. find more than one neighbors (x neighbors), use following code to replace search2Layer
  // endless loop?
  // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> Candidates = searchBaseLayer(
  // currObj, dataPoint, level);
  // TODO: 2. find one-hot neighbors and two-hot neighbors

  // Find one index1-node's (or ef_construction_ nodes) candidate neighbors in index2, and add its neighbors in index2, index1-node's neighbor in index1 ,and its neighbors from last layer in new index
  std::vector<tableint> Candidates;
  Candidates.push_back(entry_point + offset_index2);

  linklistsizeint *ll_cur = index2->get_linklist_at_level(entry_point, level);
  size_t linklistCount = index2->getListCount(ll_cur);
  tableint *data = (tableint *)(ll_cur + 1);
  for (size_t iter = 0; iter < linklistCount; iter++)
  {
    Candidates.push_back(data[iter] + offset_index2);
  }

  ll_cur = index1->get_linklist_at_level(cur_c, level);
  linklistCount = index1->getListCount(ll_cur);
  data = (tableint *)(ll_cur + 1);
  for (size_t iter = 0; iter < linklistCount; iter++)
  {
    Candidates.push_back(data[iter] + offset_index1);
  }

  // if (level != maxlevel_ && last_entry_point != -1)
  // {
  //     ll_cur = get_linklist_at_level(cur_c + offset_index1, level + 1);
  //     linklistCount = getListCount(ll_cur);
  //     Candidates.insert(Candidates.end(), (tableint *)(ll_cur + 1), (tableint *)(ll_cur + 1) + linklistCount);
  // }

  // std::unordered_set<tableint> seen;
  std::vector<tableint> result = Candidates;
  // for (const auto &val : Candidates)
  // {
  //     if (seen.find(val) == seen.end() && val != cur_c + offset_index1)
  //     {
  //         seen.insert(val);
  //         result.push_back(val);
  //     }
  // }

  // double t0 = elapsed();
  // get all distance data and then
  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
  // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
  dist_t dist = fstdistfunc_(data_point, getDataByInternalId(result[0]), dist_func_param_);
#ifdef USE_SSE
  _mm_prefetch(result.data(), _MM_HINT_T0);
  _mm_prefetch(result.data() + 1, _MM_HINT_T0);
#endif
  top_candidates.emplace(dist, result[0]);
  dist_t lowerBound = dist;
  // candidateSet.emplace(-dist, result[0]);
  time_counter_ += result.size();
  for (size_t j = 1; j < result.size(); j++)
  {
    tableint candidate_id = result[j];
#ifdef USE_SSE
    _mm_prefetch(result.data() + j + 1, _MM_HINT_T0);
#endif
    char *currObj1 = getDataByInternalId(candidate_id);
    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
    if (top_candidates.size() < ef_construction_ || lowerBound > dist1)
    {
      // candidateSet.emplace(-dist1, candidate_id);
      top_candidates.emplace(dist1, candidate_id);
      if (top_candidates.size() > ef_construction_)
        top_candidates.pop();
      if (!top_candidates.empty())
        lowerBound = top_candidates.top().first;
    }
  }
  // time_counter_+=elapsed()-t0;
  size_t Mcurmax = level ? maxM_ : maxM0_;
  getNeighborsByHeuristic2(top_candidates, Mcurmax);
  ll_cur = get_linklist_at_level(cur_c + offset_index1, level);
  setListCount(ll_cur, top_candidates.size());
  data = (tableint *)(ll_cur + 1);
  dist_t *distData = (dist_t *)get_dist_at_level(cur_c, level);
  for (size_t idx = 0; top_candidates.size() > 0; idx++)
  {
    data[idx] = top_candidates.top().second;
    distData[idx] = top_candidates.top().first;
    top_candidates.pop();
  }
  last_entry_point = entry_point;
}