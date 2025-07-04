// #include "hnswlib.h"
#include "extension.h"
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

void workload1()
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
  // char *base_filepath = "/ssd_root/dataset/ann_sift1b/parquet/bigann_base_1b-0.bvecs";
  float *xb = new float[dim * nb];
  printf("loading dataset of vectors \n");
  size_t dd2 = dim; // dimension
  size_t nt2 = nb;  // the number of query
  xb = fvecs_read(base_filepath, &dd2, &nt2);
  // xb = bvecs_read(base_filepath, 1000000, &dd2, &nt2);
  printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

  double t0;
  // t0 = elapsed();
  // for (int i = 0; i < 1000000; i++)
  // {
  //   alg_hnsw0->addPoint(xb + i * dim, i);
  //   // if (i == 96)
  //   // {
  //   //   for (int j = 0; j < 128; j++)
  //   //   {
  //   //     printf("%.0f, ", *(xb + i * dim + j));
  //   //   }
  //   //   printf("\n\n");
  //   // }
  // }
  // printf("[%.3f s] build index0\n", elapsed() - t0);

  t0 = elapsed();
  for (int i = 0; i < 500000; i++)
  {
    // printf("%d\n", i);
    alg_hnsw1->addPoint(xb + i * dim, i);
  }
  printf("[%.3f s] build index1\n", elapsed() - t0);

  for (int i = 500000; i < 1000000; i++)
  {
    alg_hnsw2->addPoint(xb + i * dim, i);
  }
  printf("[%.3f s] build index1 and index2\n", elapsed() - t0);

  // hnswlib::linklistsizeint *llcur = alg_hnsw2->get_linklist0(686);
  // printf("%d %p\n", alg_hnsw2->getListCount(llcur),llcur);
  //  return 0;

  t0 = elapsed();
  hnswlib::HierarchicalNSW<float> *alg_hnsw3 = hnswlib::HNSWMerger<float>(alg_hnsw1, alg_hnsw2, &space);
  printf("[%.3f s] build merged index\n", elapsed() - t0);
  printf("[%.3f s] for check part\n", alg_hnsw3->time_counter_);
  alg_hnsw3->setEf(200);

  // char *query_filepath = "/ssd_root/dataset/sift1m/sift_query.fvecs";
  char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
  size_t nq = 10000;
  float *xq = new float[dim * nq];
  printf("loading dataset of vectors \n");
  size_t dd = dim; // dimension
  size_t nt = nq;  // the number of query
  // xq = fvecs_read(query_filepath, &dd, &nt);
  xq = bvecs_read(query_filepath, 10000, &dd, &nt);
  printf("loaded a %ld vectors in %ld dimension \n", nt, dd);

  // int64_t *gt = new int64_t[nq * k];
  // // std::ifstream file("../sift20k/gt.txt");
  // std::ifstream file("../bigann200k/gt.txt");
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
  // size_t kk = k, nqq = nq;
  // int *gt = ivecs_read("/ssd_root/dataset/sift1m/sift_groundtruth.ivecs", &kk, &nqq);
  size_t kk = 1000, nqq = 10000;
  int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", &kk, &nqq);
  int *gt = new int[nq * k];
  for (int i = 0; i < nq; i++)
  {
    for (int j = 0; j < k; j++)
    {
      gt[i * k + j] = gt_int[i * kk + j];
    }
  }
  printf("loading queries and gt\n");

  double t1, t11 = 0.0;
  int *I = new int[nq * k];
  // nq = 1;
  t0 = elapsed();
  for (int i = 0; i < nq; i++)
  {
    // if (i == 0)
    // {
    //   for (int j = 0; j < 128; j++)
    //   {
    //     printf("%.0f, ", *(xq + i * dim + j));
    //   }
    //   printf("\n\n");
    // }
    t1 = elapsed();
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw3->searchKnn(xq + i * dim, k);
    t11 += elapsed() - t1;
    for (int j = k - 1; j >= 0; j--)
    {
      // printf("(%d, %f),", (int)(result.top().second), (result.top().first));
      I[i * k + j] = (int)(result.top().second);
      result.pop();
    }
  }
  // printf("\n\n");
  printf("[%.3f & %.3f s] searching on merged index\n", t11, elapsed() - t0);

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
  printf("\nIntersection-merged index R@100 = %.4f\n", n2_100 / float(nq * k));

  // I = new int[nq * k];
  // t11 = 0.0;
  // t0 = elapsed();
  // for (int i = 0; i < nq; i++)
  // {
  //   t1 = elapsed();
  //   std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw0->searchKnn(xq + i * dim, k);
  //   t11 += elapsed() - t1;
  //   for (int j = k - 1; j >= 0; j--)
  //   {
  //     // printf("(%d, %f),", (int)(result.top().second), (result.top().first));
  //     I[i * k + j] = (int)(result.top().second);
  //     result.pop();
  //   }
  // }
  // // printf("\n\n");
  // printf("[%.3f & %.3f s] searching on index0\n", t11, elapsed() - t0);

  // n2_100 = 0;
  // for (int i = 0; i < nq; i++)
  // {
  //   std::map<float, int> umap;
  //   for (int j = 0; j < k; j++)
  //   {
  //     umap.insert({gt[i * k + j], 0});
  //   }
  //   for (int l = 0; l < k; l++)
  //   {
  //     if (umap.find(I[i * k + l]) != umap.end())
  //     {
  //       n2_100++;
  //     }
  //   }
  //   umap.clear();
  // }
  // printf("\nIntersection-index0 R@100 = %.4f\n", n2_100 / float(nq * k));
}

void workload2()
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
  alg_hnsw0->setEf(200);

  hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
  alg_hnsw1->setEf(200);
  hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
  alg_hnsw2->setEf(200);

  char *base_filepath = "/ssd_root/dataset/sift1m/sift_base.fvecs";
  // char *base_filepath = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";
  float *xb = new float[dim * nb];
  // printf("loading dataset of vectors \n");
  size_t dd2 = dim; // dimension
  size_t nt2 = nb;  // the number of query
  // xb = fvecs_read(base_filepath, &dd2, &nt2);
  // xb = bvecs_read(base_filepath, 1000000, &dd2, &nt2);
  printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

  double t0;
  // t0 = elapsed();
  // int max_elements0 = 500000;
  // for (int i = 0; i < max_elements0; i++)
  // {
  //   alg_hnsw0->addPoint(xb + i * dim, i);
  // }
  // for(int i=0;i< max_elements0;i++){
  //   hnswlib::linklistsizeint *ll_cur = alg_hnsw0->get_linklist0(i);
  //   auto cnt = alg_hnsw0->getListCount(ll_cur);
  //   hnswlib::tableint *data = (hnswlib::tableint *)(ll_cur + 1);
  //   auto *dist = alg_hnsw0->get_dist0(i);
  //   for(int j=0;j<cnt;j++){
  //     if (dist[j] <1e-6){
  //       printf("error %d %d %f\n",i,data[j],dist[j]);
  //     }
  //   }
  //   // printf("%d %p\n", alg_hnsw0->getListCount(ll_cur),ll_cur);
  // }
  // // alg_hnsw0->saveIndex("sift1m.hnsw");
  // printf("[%.3f s] build index0\n", elapsed() - t0);
  // exit(0);

  t0 = elapsed();
  // for (int i = 0; i < 500000; i++)
  // {
  //   alg_hnsw1->addPoint(xb + i * dim, i);
  // }
  // alg_hnsw1->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index1.hnsw");
  // printf("[%.3f s] build index1\n", elapsed() - t0);
  // alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/index1.hnsw", &space);
  alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_500K.hnsw", &space);

  // for (int i = 500000; i < 1000000; i++)
  // {
  //   alg_hnsw2->addPoint(xb + i * dim, i);
  // }
  // alg_hnsw2->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index2.hnsw");
  // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/index2.hnsw", &space);
  alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_500000_1000000.hnsw", &space);
  printf("[%.3f s] build index1 and index2\n", elapsed() - t0);
  // exit(0);

  // t0 = elapsed();
  // auto clusters = alg_hnsw1->cluster_points(alg_hnsw2,50, 2, 20, 128);
  // printf("[%.3f s] cluster points\n", elapsed() - t0);
  // for (int i = 0; i < 50; i++)
  // {
  //   std::cout << "Cluster " << i << ": ";
  //   for (auto point : clusters[i])
  //   {
  //     std::cout << point << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // exit(0);

  // hnswlib::linklistsizeint *llcur = alg_hnsw2->get_linklist0(686);
  // printf("%d %p\n", alg_hnsw2->getListCount(llcur),llcur);
  //  return 0;

  t0 = elapsed();
  hnswlib::HierarchicalNSW<float> *alg_hnsw3 = hnswlib::HNSWMerger<float>(alg_hnsw1, alg_hnsw2, &space, -1, -1);
  // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw");

  // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, alg_hnsw1->max_elements_+alg_hnsw2->max_elements_, alg_hnsw1->M_, alg_hnsw1->ef_construction_);
  // alg_hnsw3->HNSWMerger(alg_hnsw1, alg_hnsw2, 1);
  // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-balanced-index-opt.hnsw");
  
  // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, alg_hnsw1->max_elements_+alg_hnsw2->max_elements_, alg_hnsw1->M_, alg_hnsw1->ef_construction_);
  // alg_hnsw3->HNSWMerger(alg_hnsw1, alg_hnsw2);
  // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw");

  // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
  // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw", &space);
  // alg_hnsw3->loadIndex("/ssd_root/jin467/merger//indexes/merged-index.hnsw", &space);
  // alg_hnsw3->loadIndex("/ssd_root/jin467/merger//indexes/refined-index.hnsw", &space);
  printf("[%.3f s] build merged index\n", elapsed() - t0);
  printf("[%.3f s] for check part\n", alg_hnsw3->time_counter_);
  alg_hnsw3->setEf(200);

  // t0 = elapsed();
  // hnswlib::HierarchicalNSW<float> *alg_hnsw4 = hnswlib::HNSWRefinement<float>(alg_hnsw3, &space);
  // alg_hnsw4->saveIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw");
  // alg_hnsw1->HNSWMerger(alg_hnsw2);
  // alg_hnsw1->saveIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw");
  // printf("[%.3f s] build refined index\n", elapsed() - t0);
  // exit(0);

  // char *query_filepath = "/ssd_root/dataset/sift1m/sift_query.fvecs";
  char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
  size_t nq = 10000;
  float *xq = new float[dim * nq];
  printf("loading dataset of vectors \n");
  size_t dd = dim; // dimension
  size_t nt = nq;  // the number of query
  // xq = fvecs_read(query_filepath, &dd, &nt);
  xq = bvecs_read(query_filepath, 10000, &dd, &nt);
  printf("loaded a %ld vectors in %ld dimension \n", nt, dd);

  // int64_t *gt = new int64_t[nq * k];
  // // std::ifstream file("../sift20k/gt.txt");
  // std::ifstream file("../bigann200k/gt.txt");
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
  // size_t kk = k, nqq = nq;
  // int *gt = ivecs_read("/ssd_root/dataset/sift1m/sift_groundtruth.ivecs", &kk, &nqq);
  size_t kk = 1000, nqq = 10000;
  int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", &kk, &nqq);
  int *gt = new int[nq * k];
  for (int i = 0; i < nq; i++)
  {
    for (int j = 0; j < k; j++)
    {
      gt[i * k + j] = gt_int[i * kk + j];
    }
  }
  printf("loaded queries and gt\n");

  double t1, t11 = 0.0;
  int *I = new int[nq * k];
  t0 = elapsed();
  for (int i = 0; i < nq; i++)
  {
    t1 = elapsed();
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw3->searchKnn(xq + i * dim, k);
    t11 += elapsed() - t1;
    for (int j = k - 1; j >= 0; j--)
    {
      I[i * k + j] = (int)(result.top().second);
      result.pop();
    }
  }
  printf("[%.3f & %.3f s] searching on merged index\n", t11, elapsed() - t0);

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
  printf("\nIntersection-merged index R@100 = %.4f\n", n2_100 / float(nq * k));
}

int main()
{
  workload2();

  return 0;
}