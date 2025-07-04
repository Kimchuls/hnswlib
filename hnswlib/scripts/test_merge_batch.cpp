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
  // int d;
  // fread(&d, 1, sizeof(int), f);
  // assert((d > 0 && d < 1000000) || !"unreasonable dimension");
  // fseek(f, 0, SEEK_SET);
  // struct stat st;
  // fstat(fileno(f), &st);
  // size_t sz = st.st_size;
  // assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
  // size_t n = sz / ((d + 1) * 4);

  // *d_out = d;
  // *n_out = n;
  int d = *d_out;
  size_t n = *n_out;
  float *x = new float[n * (d + 1)];
  size_t nr = fread(x, sizeof(float), n * (d + 1), f);
  assert(nr == n * (d + 1) || !"could not read whole file");

  // shift array to remove row headers
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
int main()
{
  int dim = 128;             // Dimension of the elements
  int max_elements = 500000; // Maximum number of elements, should be known beforehand
  int nb = 10000000;
  int M = 32;               // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
  int ef_construction = 40; // Controls index search speed/build speed tradeoff
  int k = 100;
  size_t nq = 10000;

  char *filepath1 = "/ssd_root/dataset/ann_sift1b/parquet/bigann_base_1b-0.bvecs"; // 10M
  char *filepath2 = "/ssd_root/dataset/ann_sift1b/parquet/bigann_base_1b-1.bvecs"; // 10M

  hnswlib::L2Space space(dim);

  hnswlib::HierarchicalNSW<float> **alg_hnsw_array = (hnswlib::HierarchicalNSW<float> **)malloc(10 * sizeof(hnswlib::HierarchicalNSW<float> *));
  for (int i = 0; i < 10; i++)
  {
    alg_hnsw_array[i] = new hnswlib::HierarchicalNSW<float>(&space, 1000000, M, ef_construction);
  }
  char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
  float *xq = new float[dim * nq];
  size_t dd = 128;   // dimension
  size_t nt = 10000; // the number of query
  xq = bvecs_read(query_filepath, 10000, &dd, &nt);
  // xq = fvecs_read("/ssd_root/dataset/sift1m/sift_query.fvecs", &dd, &nt);
  printf("loaded dataset of vectors \n");

  float *xb = new float[dim * nb];
  size_t dd2 = 128;      // dimension
  size_t nt2 = 10000000; // the number of query
  xb = bvecs_read(filepath1, 1000000, &dd2, &nt2);
  // xb = fvecs_read("/ssd_root/dataset/sift1m/sift_base.fvecs", &dd2, &nt2);
  printf("loaded dataset of vectors 0-10M\n");

  // char *path = "/home/jin467/github_download/hnswlib/bigann20k/gt.txt";
  // int64_t *gt = new int64_t[nq * k];
  // std::ifstream file(path);
  // if (!file.is_open())
  // {
  //   std::cerr << "Error: Could not open gt.txt" << std::endl;
  //   return 1;
  // }
  // std::string line;
  // int index = 0;
  // while (std::getline(file, line) && index < nq * k)
  // {
  //   std::stringstream ss(line);
  //   int64_t value;
  //   ss >> value;
  //   gt[index++] = value;
  // }
  // file.close();
  char *path = "/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs";
  size_t kk = 1000, nqq = 10000;
  int *gt_int = ivecs_read(path, &kk, &nqq);
  int *gt = new int[nq * k];
  for (int i = 0; i < nq; i++)
  {
    for (int j = 0; j < k; j++)
    {
      gt[i * k + j] = gt_int[i * kk + j];
    }
  }

  // hnswlib::HierarchicalNSW<float> *merge2 = new hnswlib::HierarchicalNSW<float>(&space, 1000000, M, ef_construction);
  // merge2->setEf(200);
  // double t0, t01;
  // t0 = elapsed();
  // for (int i = 0; i < 1000000; i++)
  // {
  //   merge2->addPoint(xb + i * dim, i);
  // }
  // t01 = elapsed() - t0;
  // printf("[%.3f s] build index\n", t01);

  double t0, t01;
  t0 = elapsed();
  int batch_num=10; // max = 10
  int batch_size=100000;
  for (int i = 0; i < batch_num; i++)
  {
    for (int j = 0; j < batch_size; j++)
    {
      alg_hnsw_array[i]->addPoint(xb + (batch_size * i + j) * dim, batch_size * i + j);
    }
  }
  t01 = elapsed() - t0;
  printf("[%.3f s] build index\n", t01);

  // int dim0 = *(int*)(alg_hnsw_array[0]->dist_func_param_);
  // printf("%d\n",dim0);
  // exit(0);

  t0 = elapsed();
  hnswlib::HierarchicalNSW<float> *merge1 = hnswlib::HNSWMergerOnMultiple<float>(alg_hnsw_array, batch_num, &space);
  merge1->setEf(200);
  t01 = elapsed() - t0;
  printf("[%.3f s] build merged index one by one\n", t01);

  {
    double t1, t11 = 0.0;
    int *I = new int[nq * k];
    {
      t0 = elapsed();
      for (int i = 0; i < nq; i++)
      {
        t1 = elapsed();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merge1->searchKnn(xq + i * dim, k);
        t11 += elapsed() - t1;
        for (int j = k - 1; j >= 0; j--)
        {
          I[i * k + j] = (int)(result.top().second);
          result.pop();
        }
      }
      printf("[%.3f & %.3f s] searching on merge1 index\n", t11, elapsed() - t0);
    }

    int n2_100 = 0;
    for (int i = 0; i < nq; i++)
    {
      std::map<float, int> umap;
      for (int j = 0; j < k; j++)
      {
        umap.insert({gt[i * k + j], 0});
      }
      for (int l = 0; l < k; l++)
      {
        if (umap.find(I[i * k + l]) != umap.end())
        {
          n2_100++;
        }
      }
      umap.clear();
    }
    printf("Intersection-merge1-index R@100 = %.4f\n\n", n2_100 / float(nq * k));

    // size_t element_count_for_index = merge1->getCurrentElementCount();
    // std::vector<std::vector<int>> layer_node_for_index(merge1->maxlevel_ + 1);
    // searchNodeOnEachLayer(merge1, layer_node_for_index);
    // printf("maxlevel: %d\n",merge1->maxlevel_);
    // for(int i=0;i<layer_node_for_index[layer_node_for_index.size()-1].size();i++){
    //   printf("%d, ",layer_node_for_index[layer_node_for_index.size()-1][i]);
    // }
    // printf("\n");
  }

  // {
  //   double t1, t11 = 0.0;
  //   int *I = new int[nq * k];
  //   {
  //     t0 = elapsed();
  //     for (int i = 0; i < nq; i++)
  //     {
  //       t1 = elapsed();
  //       std::priority_queue<std::pair<float, hnswlib::labeltype>> result = merge2->searchKnn(xq + i * dim, k);
  //       t11 += elapsed() - t1;
  //       for (int j = k - 1; j >= 0; j--)
  //       {
  //         I[i * k + j] = (int)(result.top().second);
  //         result.pop();
  //       }
  //     }
  //     printf("[%.3f & %.3f s] searching on merge2 index\n", t11, elapsed() - t0);
  //   }

  //   int n2_100 = 0;
  //   for (int i = 0; i < nq; i++)
  //   {
  //     std::map<float, int> umap;
  //     for (int j = 0; j < k; j++)
  //     {
  //       umap.insert({gt[i * k + j], 0});
  //     }
  //     for (int l = 0; l < k; l++)
  //     {
  //       if (umap.find(I[i * k + l]) != umap.end())
  //       {
  //         n2_100++;
  //       }
  //     }
  //     umap.clear();
  //   }
  //   printf("Intersection-merge2-index R@100 = %.4f\n\n", n2_100 / float(nq * k));

  // //   size_t element_count_for_index = merge2->getCurrentElementCount();
  // //   std::vector<std::vector<int>> layer_node_for_index(merge2->maxlevel_ + 1);
  // //   searchNodeOnEachLayer(merge2, layer_node_for_index);
  // //   printf("maxlevel: %d\n",merge2->maxlevel_);
  // //   for(int i=0;i<layer_node_for_index[layer_node_for_index.size()-1].size();i++){
  // //     printf("%d, ",layer_node_for_index[layer_node_for_index.size()-1][i]);
  // //   }
  // //   printf("\n");
  // }

  return 0;
}