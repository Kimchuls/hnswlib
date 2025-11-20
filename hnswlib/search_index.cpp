#include "hnswalg.h"
#include "hnswlib.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <map>
using namespace std;

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out) {
  FILE *f = fopen(fname, "r");
  printf("fvecs_read: %s\n", fname);
  if (!f) {
    fprintf(stderr, "could not open %s\n", fname);
    perror("");
    abort();
  }
  int d = *d_out;
  size_t n = *n_out;
  float *x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  // cout << "Read " << nr << " vectors of dimension " << d << endl;
  assert(nr == n * (d + 1) || !"could not read whole file");
  for (size_t i = 0; i < n; i++)
    memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

  fclose(f);
  return x;
}
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out) {
  return (int *)fvecs_read(fname, d_out, n_out);
}

int main() {
  size_t d, n;
  size_t d_query, n_query;
  size_t d_ground_truth, n_ground_truth;
  int M = 16;
  int ef_construction = 200;
  int k = 100;

  // read query dataset
  d=128;
  n=1000000;
  string query_path = "/scratch1/jin467/dataset/sift/sift_query.fvecs";
  d_query = 128;
  n_query = 10000;
  float *query_data = fvecs_read(query_path.c_str(), &d_query, &n_query);
  cout << "Read " << n_query << " query vectors of dimension " << d_query
       << endl;
  // read ground truth dataset
  string ground_truth_path =
      "/scratch1/jin467/dataset/sift/sift_groundtruth.ivecs";
  d_ground_truth = 100;
  n_ground_truth = 10000;
  int *ground_truth_data =
      ivecs_read(ground_truth_path.c_str(), &d_ground_truth, &n_ground_truth);
  cout << "Read " << n_ground_truth << " ground truth vectors of dimension "
       << d_ground_truth << endl;

  // // read query dataset
  // d = 960;
  // n = 1000000;
  // string query_path = "/scratch1/jin467/dataset/gist/gist_query.fvecs";
  // d_query = 960;
  // n_query = 1000;
  // float *query_data = fvecs_read(query_path.c_str(), &d_query, &n_query);
  // cout << "Read " << n_query << " query vectors of dimension " << d_query <<
  // endl;
  // // read ground truth dataset
  // string ground_truth_path =
  // "/scratch1/jin467/dataset/gist/gist_groundtruth.ivecs"; d_ground_truth =
  // 100; n_ground_truth = 1000; int *ground_truth_data =
  // ivecs_read(ground_truth_path.c_str(), &d_ground_truth, &n_ground_truth);
  // cout << "Read " << n_ground_truth << " ground truth vectors of dimension "
  // << d_ground_truth << endl;

  hnswlib::L2Space space(d);
  hnswlib::HierarchicalNSW<float> *alg_hnsw0 =
      new hnswlib::HierarchicalNSW<float>(&space, 1000000, M,
                                          ef_construction);
  string index_path = "/scratch1/jin467/indexes/sift_index.hnsw";
  alg_hnsw0->loadIndex(index_path, &space);
  // string index_path = "/scratch1/jin467/indexes/gist_index.hnsw";
  // alg_hnsw0->loadIndex(index_path, &space);
  cout << "Loaded index from " << index_path << endl;

  // print result path for each query
  string result_path = "/homes/jin467/MLIndex/sift_result_path.csv";
  ofstream result_file(result_path, ios::trunc);
  // search and compare with ground truth
  int *I = new int[n_query * 100];
  for (int i = 0; i < n_query; i++) {   
    auto result = alg_hnsw0->searchKnn(query_data + i * d_query, k);
    for (int j = k - 1; j >= 0; j--) {
      I[i * k + j] = static_cast<int>(result.top().second);

      // print result path for each query
      
      result_file << i << ",";
      for (int iter = 0; iter < alg_hnsw0->search_path_nodes_layers.size();
           iter++) {
        result_file << alg_hnsw0->search_path_nodes_layers[iter].node_id << ","
                    << alg_hnsw0->search_path_nodes_layers[iter].layer << ",";
      }
      auto path_layer0 = alg_hnsw0->search_path_nodes_layer0[I[i * k + j]];
      for (int iter = 0; iter < path_layer0.size(); iter++) {
        result_file << path_layer0[iter].node_id << ","
                    << path_layer0[iter].layer << ",";
      }
      result_file << j + 1 << "," << sqrt(result.top().first) << ",";
      result_file << endl;

      result.pop();
    }
    if ((i+1) % 1000 == 0) {
      cout << "Processed " << i+1 << " queries" << endl;
    }
  }
  result_file.close();

  int correct = 0;
  for (int i = 0; i < n_query; i++) {
    std::map<int, int> umap;
    for (int j = 0; j < k; j++) {
      umap[ground_truth_data[i * k + j]] = 1;
    }
    for (int j = 0; j < k; j++) {
      if (umap.find(I[i * k + j]) != umap.end()) {
        correct++;
      }
    }
  }
  float recall = correct / static_cast<float>(n_query * k);
  printf("Intersection-merged index R@100 = %.4f\n\n", recall);

  delete[] I;

  return 0;
}