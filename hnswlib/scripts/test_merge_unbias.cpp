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

/*
  For this separative test, we will build the index alg_hnsw20M from 0 to 10,000,000 separatively and collect time cost in each gap.
  Then we want to merge 2 indexes which are build WITH merger, and test the search performance.
*/

void workload_10M()
{
  int dim = 128;             // Dimension of the elements
  int max_elements = 500000; // Maximum number of elements, should be known beforehand
  int nb = 10000000;
  int M = 32;               // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
  int ef_construction = 40; // Controls index search speed/build speed tradeoff
  int k = 100;
  size_t nq = 10000;

  char *filepath1 = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";

  hnswlib::L2Space space(dim);

  hnswlib::HierarchicalNSW<float> *alg_hnsw20M = new hnswlib::HierarchicalNSW<float>(&space, 10000000, M, ef_construction);
  alg_hnsw20M->setEf(200);
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
  xb = bvecs_read(filepath1, 10000000, &dd2, &nt2);
  // xb = fvecs_read("/ssd_root/dataset/sift1m/sift_base.fvecs", &dd2, &nt2);
  printf("loaded dataset of vectors 0-10M\n");
  int listing[] = {9000000, 10000000};
  // char *path[] = {"", "/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_2M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_5M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs"};
  char *path[] = {"", "/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs"};

  double t0, t00 = 0.0, t01;
  t0 = elapsed();
  // for (int i = 0; i < listing[0]; i++)
  // {
  //   alg_hnsw20M->addPoint(xb + i * dim, i);
  // }
  // alg_hnsw20M->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_9M.hnsw");
  alg_hnsw20M->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_9M.hnsw", &space);
  t01 = elapsed() - t0;
  printf("[%.3f s] build index-%d, [%.3f s] in total build\n", t01, listing[0], t00 += t01);

  for (int iter = 1; iter < 2; iter++)
  {
    hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, listing[iter] - listing[iter - 1], M, ef_construction);
    printf("start build index-(%d to %d)\n", listing[iter - 1], listing[iter]);

    {
      t0 = elapsed();
      // for (int i = listing[iter - 1]; i < listing[iter]; i++)
      // {
      //   alg_hnsw1->addPoint(xb + i * dim, i);
      // }
      // alg_hnsw1->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_" + std::to_string(listing[iter-1]) +"_"+ std::to_string(listing[iter])+".hnsw");
      alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_" + std::to_string(listing[iter - 1]) + "_" + std::to_string(listing[iter]) + ".hnsw", &space);
      t01 = elapsed() - t0;
      printf("[%.3f s] build index-(%d to %d), [%.3f s] in total build\n", t01, listing[iter - 1], listing[iter], t00 += t01);
    }

    {
      t0 = elapsed();
      // alg_hnsw20M = hnswlib::HNSWMerger<float>(alg_hnsw20M, alg_hnsw1, &space);
      hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, 10000000, M, ef_construction);
      alg_hnsw2->HNSWMerger(alg_hnsw20M, alg_hnsw1);
      alg_hnsw20M = alg_hnsw2;
      alg_hnsw20M->setEf(200);
      t01 = elapsed() - t0;
      printf("[%.3f s] build merged index-%d, [%.3f s] in total build\n", t01, listing[iter], t00 += t01);
      // alg_hnsw20M->saveIndex("/ssd_root/jin467/merger/bigann-index" + std::to_string(listing[iter]) + ".hnsw");
    }

    delete alg_hnsw1;
    size_t kk = 1000, nqq = 10000;
    int *gt_int = ivecs_read(path[iter], &kk, &nqq);
    int *gt = new int[nq * k];
    for (int i = 0; i < nq; i++)
    {
      for (int j = 0; j < k; j++)
      {
        gt[i * k + j] = gt_int[i * kk + j];
      }
    }
    double t1, t11 = 0.0;
    int *I = new int[nq * k];

    {
      t0 = elapsed();
      for (int i = 0; i < nq; i++)
      {
        // t1 = elapsed();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw20M->searchKnn(xq + i * dim, k);
        // t11 += elapsed() - t1;
        for (int j = k - 1; j >= 0; j--)
        {
          I[i * k + j] = (int)(result.top().second);
          result.pop();
        }
      }
      // printf("[%.3f & %.3f s] searching on %d index\n", t11, elapsed() - t0, listing[iter]);
      printf("[%.3f s] searching on %d index\n", elapsed() - t0, listing[iter]);
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
    printf("Intersection-index-%d R@100 = %.4f\n\n", listing[iter], n2_100 / float(nq * k));
  }
}

void workload_1M()
{
  int dim = 128;             // Dimension of the elements
  int max_elements = 500000; // Maximum number of elements, should be known beforehand
  int nb = 10000000;
  int M = 32;               // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
  int ef_construction = 40; // Controls index search speed/build speed tradeoff
  int k = 100;
  size_t nq = 10000;

  char *filepath1 = "/ssd_root/dataset/sift1m/sift_base.fvecs";

  hnswlib::L2Space space(dim);

  hnswlib::HierarchicalNSW<float> *alg_hnsw20M = new hnswlib::HierarchicalNSW<float>(&space, 1000000, M, ef_construction);
  alg_hnsw20M->setEf(200);
  char *query_filepath = "/ssd_root/dataset/sift1m/sift_query.fvecs";
  float *xq = new float[dim * nq];
  size_t dd = 128;   // dimension
  size_t nt = 10000; // the number of query
  xq = fvecs_read(query_filepath, &dd, &nt);
  // xq = fvecs_read("/ssd_root/dataset/sift1m/sift_query.fvecs", &dd, &nt);
  printf("loaded dataset of vectors \n");

  float *xb = new float[dim * nb];
  size_t dd2 = 128;      // dimension
  size_t nt2 = 1000000; // the number of query
  xb = fvecs_read(filepath1, &dd2, &nt2);
  // xb = fvecs_read("/ssd_root/dataset/sift1m/sift_base.fvecs", &dd2, &nt2);
  printf("loaded dataset of vectors 0-1M\n");
  int listing[] = {900000, 1000000};
  // char *path[] = {"", "/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_2M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_5M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs"};
  char *path[] = {"", "/ssd_root/dataset/sift1m/sift_groundtruth.ivecs"};

  double t0, t00 = 0.0, t01;
  t0 = elapsed();
  // for (int i = 0; i < listing[0]; i++)
  // {
  //   alg_hnsw20M->addPoint(xb + i * dim, i);
  //   if(i % 100000 == 0)
  //   {
  //     printf("add point %d\n", i);
  //   }
  // }
  // alg_hnsw20M->saveIndex("/ssd_root/jin467/merger/indexes/sift1M-index_900K.hnsw");
  alg_hnsw20M->loadIndex("/ssd_root/jin467/merger/indexes/sift1M-index_900K.hnsw", &space);
  t01 = elapsed() - t0;
  printf("[%.3f s] build index-%d, [%.3f s] in total build\n", t01, listing[0], t00 += t01);

  // {
  //   t0 = elapsed();
  //     for (int i = listing[0]; i < listing[1]; i++)
  //     {
  //       alg_hnsw20M->addPoint(xb + i * dim, i);
  //     }
  //     t01 = elapsed() - t0;
  //     printf("[%.3f s] build index-(%d to %d), [%.3f s] in total build\n", t01, listing[0], listing[1], t00 += t01);
  // }
  // exit(0);

  for (int iter = 1; iter < 2; iter++)
  {
    hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, listing[iter] - listing[iter - 1], M, ef_construction);
    printf("start build index-(%d to %d)\n", listing[iter - 1], listing[iter]);

    {
      t0 = elapsed();
      // for (int i = listing[iter - 1]; i < listing[iter]; i++)
      // {
      //   alg_hnsw1->addPoint(xb + i * dim, i);
      // }
      // alg_hnsw1->saveIndex("/ssd_root/jin467/merger/indexes/sift1M-index_" + std::to_string(listing[iter-1]) +"_"+ std::to_string(listing[iter])+".hnsw");
      alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/sift1M-index_" + std::to_string(listing[iter - 1]) + "_" + std::to_string(listing[iter]) + ".hnsw", &space);
      t01 = elapsed() - t0;
      printf("[%.3f s] build index-(%d to %d), [%.3f s] in total build\n", t01, listing[iter - 1], listing[iter], t00 += t01);
    }

    {
      t0 = elapsed();
      // alg_hnsw20M = hnswlib::HNSWMerger<float>(alg_hnsw20M, alg_hnsw1, &space);
      hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, 1000000, M, ef_construction);
      alg_hnsw2->HNSWMerger(alg_hnsw20M, alg_hnsw1);
      alg_hnsw20M = alg_hnsw2;
      alg_hnsw20M->setEf(200);
      t01 = elapsed() - t0;
      printf("[%.3f s] build merged index-%d, [%.3f s] in total build\n", t01, listing[iter], t00 += t01);
      alg_hnsw20M->saveIndex("/ssd_root/jin467/merger/sift1M-index" + std::to_string(listing[iter]) + ".hnsw");
      // alg_hnsw20M->loadIndex("/ssd_root/jin467/merger/sift1M-index" + std::to_string(listing[iter]) + ".hnsw", &space);
      
    }

    delete alg_hnsw1;
    size_t kk = 100, nqq = 10000;
    int *gt = ivecs_read(path[iter], &kk, &nqq);
    // int *gt = new int[nq * k];
    // for (int i = 0; i < nq; i++)
    // {
    //   for (int j = 0; j < k; j++)
    //   {
    //     gt[i * k + j] = gt_int[i * kk + j];
    //   }
    // }
    double t1, t11 = 0.0;
    int *I = new int[nq * k];

    {
      t0 = elapsed();
      for (int i = 0; i < nq; i++)
      {
        // t1 = elapsed();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw20M->searchKnn(xq + i * dim, k);
        // t11 += elapsed() - t1;
        for (int j = k - 1; j >= 0; j--)
        {
          I[i * k + j] = (int)(result.top().second);
          result.pop();
        }
      }
      // printf("[%.3f & %.3f s] searching on %d index\n", t11, elapsed() - t0, listing[iter]);
      printf("[%.3f s] searching on %d index\n", elapsed() - t0, listing[iter]);
    }

    int n2_100 = 0;
    for (int i = 0; i < nq; i++)
    {
      std::map<float, int> umap;
      for (int j = 0; j < k; j++)
      {
        umap.insert({gt[i * k + j], 0});
        // printf("%d ", gt[i * k + j]);
      }
      // printf("\n");
      for (int l = 0; l < k; l++)
      {
        // printf("%d ", I[i * k + l]);
        if (umap.find(I[i * k + l]) != umap.end())
        {
          n2_100++;
        }
      }
      // printf("\n");
      umap.clear();
      // break;
    }
    printf("Intersection-index-%d R@100 = %.4f\n\n", listing[iter], n2_100 / float(nq * k));
  }
}     
int main()
{
  // workload_10M();
  workload_1M();
  return 0;
}