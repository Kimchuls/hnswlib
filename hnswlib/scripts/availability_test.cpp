#include "hnswlib.h"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <cstdint> // For int64_t
#include <sstream> // For stringstream
using namespace std;
// double elapsed()
// {
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   return tv.tv_sec + tv.tv_usec * 1e-6;
// }

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
  FILE *f = fopen(fname, "r");
  if (!f)
  {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d = *d_out;
  size_t n = *n_out;
  float *x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  assert(nr == n * (d + 1) || !"could not read whole file");
  for (size_t i = 0; i < n; i++)
    memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

  fclose(f);
  return x;
}
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
  return (int *)fvecs_read(fname, d_out, n_out);
}
float *bvecs_read(const char *input_file, int num_vectors_to_read, size_t *d_out, size_t *n_out)
{
  float *result = nullptr;
  unsigned char *tmp = nullptr;
  FILE *file = fopen(input_file, "r"); // Open the file in binary read mode
  if (!file)
  {
    std::cerr << "Error: Unable to open file " << input_file << std::endl;
    return result;
  }
  int32_t d;
  int32_t n;
  fread(&d, sizeof(int32_t), 1, file);
  *d_out = d;
  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  long total_n = file_size / (d + 4);
  num_vectors_to_read = std::min(num_vectors_to_read, (int)total_n);
  *n_out = total_n;
  n = num_vectors_to_read;
  int64_t tmp_elements = (int64_t)n * (d + 4);
  int64_t num_elements = (int64_t)num_vectors_to_read * d;
  result = new float[num_elements];
  tmp = new unsigned char[tmp_elements];

  fseek(file, 0, SEEK_SET);
  fread(tmp, n * (d + 4), 1, file);
  for (int64_t i = 0; i < num_vectors_to_read; ++i)
  {
    for (int j = 0; j < d; ++j)
    {
      result[i * d + j] = static_cast<float>(tmp[i * (d + 4) + 4 + j]);
    }
  }
  fclose(file);
  delete[] tmp;
  return result;
}

void workload() // test neighbor's neighbor
{
  int dim = 128;              // Dimension of the elements
  int max_elements = 1000000; // Maximum number of elements, should be known beforehand
  int nb = 1000000;
  int M = 32;               // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
  int ef_construction = 40; // Controls index search speed/build speed tradeoff
  int k = 100;

  hnswlib::L2Space space(dim);
  // hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
  // alg_hnsw0->setEf(200);

  hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
  alg_hnsw1->setEf(200);
  hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
  alg_hnsw2->setEf(200);

  char *base_filepath = "/ssd_root/dataset/sift1m/sift_base.fvecs";
  // char *base_filepath = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";
  float *xb = new float[dim * nb];
  printf("loading dataset of vectors \n");
  size_t dd2 = dim; // dimension
  size_t nt2 = nb;  // the number of query
  xb = fvecs_read(base_filepath, &dd2, &nt2);

  double t0;

  alg_hnsw1->loadIndex("index1.hnsw", &space);
  alg_hnsw2->loadIndex("index2.hnsw", &space);

  const int NUM_SAMPLES = 100; // Number of random points to select
  const int K = 10;            // Number of nearest neighbors
  std::vector<int> selected_points;
  std::unordered_set<int> selected_indices;
  std::mt19937 rng(static_cast<unsigned>(time(0)));
  std::uniform_int_distribution<int> dist(0, 500000 - 1);

  while (selected_points.size() < NUM_SAMPLES)
  {
    int random_index = dist(rng);
    if (selected_indices.find(random_index) == selected_indices.end())
    {
      selected_points.push_back(random_index);
      selected_indices.insert(random_index);
    }
  }

  int sum = K * NUM_SAMPLES;
  int cnt = 0;
  for (int p : selected_points)
  {
    auto neighbors_in_alg_hnsw2 = alg_hnsw2->searchKnn(&xb[p * dim], K);
    // std::unordered_set<int> valid_neighbors;
    // while (!neighbors_in_alg_hnsw2.empty())
    // {
    //   valid_neighbors.insert(neighbors_in_alg_hnsw2.top().second);
    //   neighbors_in_alg_hnsw2.pop();
    // }

    std::unordered_set<int> candidate_knn, candidate_drop;
    candidate_drop.insert(p);
    int DEEP = 4;
    for (int i = 1; i < DEEP; i++)
    {
      std::unordered_set<int> candidate_layer;
      for (auto x : candidate_drop)
      {
        hnswlib::linklistsizeint *ll_cur = alg_hnsw1->get_linklist0(x);
        size_t linklistCount = alg_hnsw1->getListCount(ll_cur);
        hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
        for (size_t i = 0; i < linklistCount; i++)
        {
          candidate_layer.insert(data[i]);
        }
      }
      candidate_drop.insert(candidate_layer.begin(), candidate_layer.end());
    }

    for (auto x : candidate_drop)
    {
      hnswlib::linklistsizeint *ll_cur = alg_hnsw1->get_linklist0(x);
      size_t linklistCount = alg_hnsw1->getListCount(ll_cur);
      hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
      for (size_t i = 0; i < linklistCount; i++)
      {
        if (candidate_drop.find(data[i]) == candidate_drop.end())
          candidate_knn.insert(data[i]);
      }
    }

    while (!neighbors_in_alg_hnsw2.empty())
    {
      int q = neighbors_in_alg_hnsw2.top().second;
      neighbors_in_alg_hnsw2.pop();

      auto neighbors_of_q = alg_hnsw1->searchKnn(&xb[(500000 + q) * dim], 10);

      bool found = false;
      while (!neighbors_of_q.empty())
      {
        int neighbor_of_q = neighbors_of_q.top().second;
        neighbors_of_q.pop();
        // printf("neighbor_of_q = %d\n", neighbor_of_q);

        if (candidate_knn.find(neighbor_of_q) != candidate_knn.end())
        {
          found = true;
          break;
        }
      }
      if (found == true)
      {
        cnt++;
      }
    }
    // break;
  }
  printf("Intersection-merged index R@10 = %d/%d\n", cnt, sum);
}

void workload2() // test node connections
{
  int dim = 128;              // Dimension of the elements
  int max_elements = 1000000; // Maximum number of elements, should be known beforehand
  int nb = 1000000;
  int M = 32;               // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
  int ef_construction = 40; // Controls index search speed/build speed tradeoff
  int k = 100;

  hnswlib::L2Space space(dim);
  hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
  alg_hnsw0->loadIndex("./indexes/sift1m.hnsw", &space);
  alg_hnsw0->setEf(200);

  hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
  alg_hnsw1->loadIndex("./indexes/index1.hnsw", &space);
  alg_hnsw1->setEf(200);
  hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
  alg_hnsw2->loadIndex("./indexes/index2.hnsw", &space);
  alg_hnsw2->setEf(200);

  hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
  alg_hnsw3->loadIndex("./indexes/merged-index.hnsw", &space);
  alg_hnsw3->setEf(200);

  // auto write_neighbors = [&](hnswlib::HierarchicalNSW<float> *alg, string filename)
  // {
  //   FILE *file = fopen(filename.c_str(), "w");
  //   for (int i = 0; i < alg->cur_element_count; i++)
  //   {
  //     hnswlib::linklistsizeint *ll_cur = alg->get_linklist0(i);
  //     size_t linklistCount = alg->getListCount(ll_cur);
  //     hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
  //     hnswlib:float *dist = (float *)alg->get_dist0(i);
  //     fprintf(file, "%ld: ", alg->getExternalLabel(i));
  //     for (size_t i = 0; i < linklistCount; i++)
  //     {
  //       fprintf(file, "(%ld, %.0f); ", alg->getExternalLabel(data[i]), dist[i]);
  //     }
  //     fprintf(file, "\n");
  //   }
  //   fclose(file);
  // };

  // write_neighbors(alg_hnsw0, "indexes/neighbors0.txt");
  // printf("write neighbors0\n");
  // write_neighbors(alg_hnsw1, "indexes/neighbors1.txt");
  // printf("write neighbors1\n");
  // write_neighbors(alg_hnsw2, "indexes/neighbors2.txt");
  // printf("write neighbors2\n");
  // write_neighbors(alg_hnsw3, "indexes/neighbors3.txt");
  // printf("write neighbors3\n");

  // auto print_edges = [&](hnswlib::HierarchicalNSW<float> *alg, int id, int level)
  // {
  //   printf("level %d: %d: ", level, id);
  //   if (alg->element_levels_[id] < level)
  //   {
  //     printf("no such level\n");
  //   }
  //   else
  //   {
  //     hnswlib::linklistsizeint *ll_cur = alg->get_linklist_at_level(id, level);
  //     size_t linklistCount = alg->getListCount(ll_cur);
  //     hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
  //     float *dist = (float *)alg->get_dist_at_level(id, level);
  //     for (size_t i = 0; i < linklistCount; i++)
  //     {
  //       printf("(%d, %.0f); ", data[i], dist[i]);
  //     }
  //     printf("\n");
  //   }
  // };

  // for (int i = 0; i < 10; i++)
  // {
  //   int rand_int = rand() % 1000000;
  //   printf("alg_hnsw0\n");
  //   print_edges(alg_hnsw0, rand_int, 0);
  //   print_edges(alg_hnsw0, rand_int, 1);
  //   if (rand_int < 500000)
  //   {
  //     printf("alg_hnsw1\n");
  //     print_edges(alg_hnsw1, rand_int, 0);
  //     print_edges(alg_hnsw1, rand_int, 1);
  //   }
  //   else
  //   {
  //     printf("alg_hnsw2\n");
  //     print_edges(alg_hnsw2, rand_int - 500000, 0);
  //     print_edges(alg_hnsw2, rand_int - 500000, 1);
  //   }
  //   printf("alg_hnsw3\n");
  //   print_edges(alg_hnsw3, rand_int, 0);
  //   print_edges(alg_hnsw3, rand_int, 1);
  // }

  // auto level_count = [&](hnswlib::HierarchicalNSW<float> *alg)
  // {
  //   vector<int> level_count(alg->maxlevel_+1, 0);
  //   for (int i = 0; i < alg->cur_element_count; i++)
  //   {
  //     level_count[alg->element_levels_[i]]++;
  //   }
  //   for(int i=0;i<=alg->maxlevel_;i++){
  //     printf("level %d: %d\n", i, level_count[i]);
  //   }
  // };
  // printf("alg_hnsw0\n");
  // level_count(alg_hnsw0);
  // printf("alg_hnsw1\n");
  // level_count(alg_hnsw1);
  // printf("alg_hnsw2\n");
  // level_count(alg_hnsw2);
  // printf("alg_hnsw3\n");
  // level_count(alg_hnsw3);

  // std::uniform_real_distribution<double> distribution(0.0, 1.0);
  // int cnt = 0;
  //   for(int i=0;i<34;i++){
  //     auto x = distribution(alg_hnsw0->level_generator_);
  //     printf("%d: %f & %f\n", i, x, 1.0 / M);
  //     if(x < 1.0 / M){
  //       cnt++;
  //     }
  //   }
  //   printf("cnt = %d\n", cnt);

  // auto connections = [&](hnswlib::HierarchicalNSW<float> *alg, int id = -1, int level = 0)
  // {
  //   if (id != -1)
  //   {
  //     hnswlib::linklistsizeint *ll_cur = alg->get_linklist_at_level(id, level);
  //     size_t linklistCount = alg->getListCount(ll_cur);
  //     hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
  //     float *dist = (float *)alg->get_dist_at_level(id, level);
  //     printf("%d: ", id);
  //     for (size_t i = 0; i < linklistCount; i++)
  //     {
  //       printf("(%d, %.0f); ", data[i], dist[i]);
  //     }
  //     printf("\n");
  //   }
  //   else{
  //     int cnt = 0;
  //     for(int i=0;i<alg->cur_element_count;i++){
  //       hnswlib::linklistsizeint *ll_cur = alg->get_linklist_at_level(i, level);
  //       size_t linklistCount = alg->getListCount(ll_cur);
  //       cnt += linklistCount;
  //     }
  //     printf("avg = %f\n", cnt / (float)alg->cur_element_count);
  //   }
  // };
  // printf("alg_hnsw0\n");
  // connections(alg_hnsw0);
  // printf("alg_hnsw3\n");
  // connections(alg_hnsw3);

  // auto edgeLengthVariation = [&](hnswlib::HierarchicalNSW<float> *alg, string filename)
  // {
  //   FILE* file = fopen(filename.c_str(), "w");
  //   for (int i = 0; i < alg->cur_element_count; i++)
  //   {
  //     hnswlib::linklistsizeint *ll_cur = alg->get_linklist0(i);
  //     size_t linklistCount = alg->getListCount(ll_cur);
  //     hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
  //     float *dist = (float *)alg->get_dist0(i);
  //     float mean = 0.0, variance = 0.0;

  //     // Calculate mean
  //     for (size_t i = 0; i < linklistCount; i++)
  //     {
  //       mean += dist[i];
  //     }
  //     mean /= linklistCount;

  //     // Calculate variance
  //     for (size_t i = 0; i < linklistCount; i++)
  //     {
  //       variance += (dist[i] - mean) * (dist[i] - mean);
  //     }
  //     variance /= linklistCount;
  //     fprintf(file, "%ld, %f, %f\n", linklistCount, mean, variance);
  //   }
  //   fclose(file);
  // };
  // edgeLengthVariation(alg_hnsw0, "indexes/edgeLengthVariance0.txt");
  // edgeLengthVariation(alg_hnsw3, "indexes/edgeLengthVariance3.txt");

  // auto statByLinklistCount = [&](hnswlib::HierarchicalNSW<float> *alg, string filename)
  // {
  //   std::map<size_t, std::vector<float>> dist_map;
  //   for (int i = 0; i < alg->cur_element_count; i++)
  //   {
  //     hnswlib::linklistsizeint *ll_cur = alg->get_linklist0(i);
  //     size_t linklistCount = alg->getListCount(ll_cur);
  //     float *dist = (float *)alg->get_dist0(i);
  //     dist_map[linklistCount].insert(dist_map[linklistCount].end(), dist, dist + linklistCount);
  //   }
  //   FILE *file = fopen(filename.c_str(), "w");
  //   for (const auto &pair : dist_map)
  //   {
  //     size_t linklistCount = pair.first;
  //     const std::vector<float> &dist_list = pair.second;
  //     if (dist_list.empty())
  //       continue;
  //     float mean = 0.0, variance = 0.0, median = 0.0, q1 = 0.0, q3 = 0.0;
  //     for (float d : dist_list)
  //       mean += d;
  //     mean /= dist_list.size();
  //     for (float d : dist_list)
  //       variance += (d - mean) * (d - mean);
  //     variance /= dist_list.size();
  //     std::vector<float> sorted_dist = dist_list;
  //     std::sort(sorted_dist.begin(), sorted_dist.end());
  //     size_t n = sorted_dist.size();
  //     if (n % 2 == 0)
  //     {
  //       median = (sorted_dist[n / 2 - 1] + sorted_dist[n / 2]) / 2.0;
  //     }
  //     else
  //     {
  //       median = sorted_dist[n / 2];
  //     }
  //     size_t q1_index = n / 4;
  //     size_t q3_index = 3 * n / 4;
  //     q1 = sorted_dist[q1_index];
  //     q3 = sorted_dist[q3_index];
  //     fprintf(file, "%ld, %f, %f, %f, %f, %f\n", linklistCount, mean, variance, median, q1, q3);
  //   }
  //   fclose(file);
  // };

  // statByLinklistCount(alg_hnsw0, "indexes/statByLinklistCount0.txt");
  // statByLinklistCount(alg_hnsw3, "indexes/statByLinklistCount3.txt");


  auto bfs_layers = [](hnswlib::HierarchicalNSW<float> *index1, hnswlib::HierarchicalNSW<float> *index2, hnswlib::tableint p)
  {
    // Step 1: Get a random neighbor p' of p in index1
    auto ll_cur_p = index1->get_linklist0(p);
    size_t linklistCount_p = index1->getListCount(ll_cur_p);
    hnswlib::tableint *data_p = (hnswlib::tableint *)(ll_cur_p + 1);
    hnswlib::tableint p_prime = data_p[rand() % linklistCount_p];
    printf("p' = %d\n", p_prime);

    // Step 2: Find p' in index2 and get its nearest neighbor q
    auto search_result = index2->searchKnn((void *)index1->getDataByInternalId(p_prime), 1);
    if (search_result.empty())
    {
      std::cout << "p' has no nearest neighbor in index2." << std::endl;
      return;
    }
    hnswlib::tableint q = index2->label_lookup_[search_result.top().second];
    printf("q = %d\n", q);

    // Step 3: Get q's neighbors in index2
    auto ll_cur_q = index2->get_linklist0(q);
    size_t linklistCount_q = index2->getListCount(ll_cur_q);
    hnswlib::tableint *data_q = (hnswlib::tableint *)(ll_cur_q + 1);
    for (size_t i = 0; i < linklistCount_q; i++)
    {
      hnswlib::tableint q_prime = data_q[i];
      printf("q%ld: %d\n", i, q_prime);
      // Step 4: Find q' in index1 and get its top 2 nearest neighbors
      auto q_prime_neighbors = index1->searchKnn((void *)index1->getDataByInternalId(q_prime), 2);
      if (q_prime_neighbors.size() < 2)
      {
        std::cout << "q' has less than 2 neighbors in index1." << std::endl;
        return;
      }
      // Step 5: BFS to determine how many layers are needed to reach p' from nn1 and nn2 in index1
      auto bfs_layers_to_p = [&](hnswlib::tableint start)
      {
        std::queue<std::pair<hnswlib::tableint, int>> q;
        std::unordered_set<hnswlib::tableint> visited;
        q.push({start, 0});
        visited.insert(start);
        while (!q.empty())
        {
          auto [node, depth] = q.front();
          node =index1->label_lookup_[node];
          q.pop();
          if (node == p)
            return depth;

          std::vector<hnswlib::tableint> neighbors;
          hnswlib::linklistsizeint* linkList = index1->get_linklist_at_level(node, 0);
          if (linkList)
          {
            size_t size = *linkList;
            hnswlib::tableint *data = (hnswlib::tableint *)(linkList + 1);
            for (size_t i = 0; i < size; i++)
            {
              neighbors.push_back(data[i]);
            }
          }

          for (auto &n : neighbors)
          {
            if (visited.find(n) == visited.end())
            {
              q.push({n, depth + 1});
              visited.insert(n);
            }
          }
        }
        return -1; // If not found
      };

      while (!q_prime_neighbors.empty())
      {
        auto q_prime_neighbor = q_prime_neighbors.top();
        q_prime_neighbors.pop();
        int layers = bfs_layers_to_p(q_prime_neighbor.second);
        if (layers != -1)
        {
          printf("p' is %d layers away from q%ld\n", layers, q_prime_neighbor.second);
        }
        else
        {
          printf("p' is unreachable from q%ld\n", q_prime_neighbor.second);
        }
      }
    }
  };

  for(int i=0;i<10;i++){
    hnswlib::tableint rand_int = rand() % 500000;
    printf("rand_int = %d\n", rand_int);
    bfs_layers(alg_hnsw1, alg_hnsw2, rand_int);
  }
}

int main()
{
  workload2();

  return 0;
}