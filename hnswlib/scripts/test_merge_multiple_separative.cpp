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

  // char *filepath1 = "/ssd_root/dataset/ann_sift1b/parquet/bigann_base_1b-0.bvecs"; // 10M

  char *filepath1 = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";
  // char *filepath2 = "/ssd_root/dataset/ann_sift1b/parquet/bigann_base_1b-1.bvecs"; // 10M

  hnswlib::L2Space space(dim);

  hnswlib::HierarchicalNSW<float> *alg_hnsw10M = new hnswlib::HierarchicalNSW<float>(&space, 10000000, M, ef_construction);
  hnswlib::HierarchicalNSW<float> *alg_hnsw_inc;
  alg_hnsw10M->setEf(200);
  // alg_hnsw_inc->setEf(200);

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
  int listing[] = {0, 500000, 1000000, 2000000, 5000000, 10000000};
  char *path[] = {"", "/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_2M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_5M.ivecs", "/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs"};

  double t0, t1, t_gt = 0.0, t_inc = 0.0, t_merge = 0.0;
  t0 = elapsed();
  for (int i = 0; i < listing[1]; i++)
  {
    alg_hnsw10M->addPoint(xb + i * dim, i);
  }
  t_gt = elapsed() - t0;
  printf("[%.3f s] in total build gt index-%d\n", t_gt, listing[1]);

  for (int iter = 1; iter < 5; iter++)
  {
    hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, listing[iter + 1] - listing[iter], M, ef_construction);

    {
      t0 = elapsed();
      for (int i = listing[iter]; i < listing[iter + 1]; i++)
      {
        alg_hnsw1->addPoint(xb + i * dim, i);
      }
      t_inc = elapsed() - t0;
      printf("[%.3f s] build index-(%d to %d)\n", t_inc, listing[iter], listing[iter + 1]);
    }

    {
      t0 = elapsed();
      alg_hnsw_inc = hnswlib::HNSWMerger<float>(alg_hnsw10M, alg_hnsw1, &space);
      alg_hnsw_inc->setEf(200);
      t_merge = elapsed() - t0;
      printf("[%.3f s] build merged index-%d, [%.3f s] in total build\n", t_merge, listing[iter + 1], t_gt + t_inc + t_merge);
      alg_hnsw_inc->saveIndex("bigann-index" + std::to_string(listing[iter]) + ".hnsw");
    }
    delete alg_hnsw1;

    {
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
          std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw_inc->searchKnn(xq + i * dim, k);
          // t11 += elapsed() - t1;
          for (int j = k - 1; j >= 0; j--)
          {
            I[i * k + j] = (int)(result.top().second);
            result.pop();
          }
        }
        // printf("[%.3f & %.3f s] searching on %d index\n", t11, elapsed() - t0, listing[iter + 1]);
        printf("[%.3f s] searching on %d index\n", elapsed() - t0, listing[iter + 1]);
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
      printf("Intersection-index-%d R@100 = %.4f\n\n", listing[iter + 1], n2_100 / float(nq * k));
    }

    {
      t0 = elapsed();
      for (int i = listing[iter]; i < listing[iter + 1]; i++)
      {
        alg_hnsw10M->addPoint(xb + i * dim, i);
        if ((i + 1) % 1000000 == 0)
        {
          printf("[%.3f s] in total build gt index-%d\n", t_gt + (elapsed() - t0), i + 1);
        }
      }
      t_gt += elapsed() - t0;
    }
  }
  return 0;
}