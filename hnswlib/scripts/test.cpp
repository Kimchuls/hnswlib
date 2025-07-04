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
#include <iostream>
#include <vector>
#include <set>
#include <utility>
using namespace std;

#include <iostream>
#include <vector>

int main() {
    std::vector<float> myVector;

    std::cout << "Maximum size of the vector: " << myVector.max_size() << std::endl;

    return 0;
}

// float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out)
// {
//   FILE *f = fopen(fname, "r");
//   if (!f)
//   {
//     fprintf(stderr, "could not open %s\n", fname);
//     perror("");
//     abort();
//   }
//   int d;
//   fread(&d, 1, sizeof(int), f);
//   assert((d > 0 && d < 1000000) || !"unreasonable dimension");
//   fseek(f, 0, SEEK_SET);
//   struct stat st;
//   fstat(fileno(f), &st);
//   size_t sz = st.st_size;
//   assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
//   size_t n = sz / ((d + 1) * 4);

//   *d_out = d;
//   *n_out = n;
//   float *x = new float[n * (d + 1)];
//   size_t nr = fread(x, sizeof(float), n * (d + 1), f);
//   assert(nr == n * (d + 1) || !"could not read whole file");

//   // shift array to remove row headers
//   for (size_t i = 0; i < n; i++)
//     memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

//   fclose(f);
//   return x;
// }
// int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
// {
//   return (int *)fvecs_read(fname, d_out, n_out);
// }

// float *bvecs_read(const char *input_file, int num_vectors_to_read, size_t *d_out, size_t *n_out)
// {
//   float *result = nullptr;
//   unsigned char *tmp = nullptr;

//   // std::ifstream file(input_file, std::ios::binary);
//   // if (!file.is_open()) {
//   //     std::cerr << "Error: Unable to open file " << input_file << std::endl;
//   //     return result;
//   // }

//   FILE *file = fopen(input_file, "r"); // Open the file in binary read mode
//   if (!file)
//   {
//     std::cerr << "Error: Unable to open file " << input_file << std::endl;
//     return result;
//   }

//   // Read the first 4 bytes to get the value of d (dimension)
//   int32_t d;
//   int32_t n;
//   // file.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
//   fread(&d, sizeof(int32_t), 1, file);
//   *d_out = d;

//   // Calculate the number of vectors (n) in the file
//   // file.seekg(0, std::ios::end);
//   // std::streamsize file_size = file.tellg();
//   // std::streamsize total_n = (file_size) / (d + 4);
//   fseek(file, 0, SEEK_END);
//   long file_size = ftell(file);
//   long total_n = file_size / (d + 4);

//   // Determine the actual number of vectors to read
//   num_vectors_to_read = std::min(num_vectors_to_read, (int)total_n);
//   *n_out = total_n;
//   n = num_vectors_to_read;
//   int64_t tmp_elements = (int64_t)n * (d + 4);
//   int64_t num_elements = (int64_t)num_vectors_to_read * d;

//   std::cout << "num elements " << num_elements << std::endl;
//   // Allocate memory for the float array
//   result = new float[num_elements];
//   std::cout << "num elements stop " << num_vectors_to_read << std::endl;

//   std::cout << "n " << n << std::endl;
//   std::cout << "total_n " << total_n << std::endl;
//   std::cout << "file_size " << file_size << std::endl;
//   tmp = new unsigned char[tmp_elements];
//   std::cout << "num elements tmp " << tmp_elements << std::endl;
//   // file.read(tmp, n * (d + 4));

//   fseek(file, 0, SEEK_SET);
//   fread(tmp, n * (d + 4), 1, file);

//   for (int64_t i = 0; i < num_vectors_to_read; ++i)
//   {
//     // Skip the first 4 bytes of each vector (integer)
//     // file.seekg(4, std::ios::cur);

//     // Read d unsigned char values from the file
//     // file.read(reinterpret_cast<char*>(buffer), d);

//     // Convert unsigned char values to float and store them in the result array
//     for (int j = 0; j < d; ++j)
//     {
//       // result.push_back(static_cast<float>(buffer[j]));
//       result[i * d + j] = static_cast<float>(tmp[i * (d + 4) + 4 + j]);
//     }
//   }

//   // file.close();
//   fclose(file);

//   delete[] tmp;

//   return result;
// }
// int main()
// {
//   int dim = 128;               // Dimension of the elements
//   int max_elements = 20000000; // Maximum number of elements, should be known beforehand
//   int nb = 10000000;
//   int M = 32;               // Tightly connected with internal dimensionality of the data
//                             // strongly affects the memory consumption
//   int ef_construction = 40; // Controls index search speed/build speed tradeoff
//   int k = 100;

//   hnswlib::L2Space space(dim);
//   hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
//   alg_hnsw->setEf(200);

//   char *base_filepath = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";
//   float *xb = new float[dim * nb];
//   printf("loading dataset of vectors \n");
//   size_t dd2; // dimension
//   size_t nt2; // the number of query
//   xb = bvecs_read(base_filepath, nb, &dd2, &nt2);
//   printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

//   // double t0 = elapsed();
//   // for (int i = 0; i < nb; i++)
//   // {
//   //   alg_hnsw->addPoint(xb + i * dim, i);
//   //   // if (i % 1000 == 0)
//   //   // {
//   //   //   printf("checkpoint:%d, [%.3f s] \n", i, e lapsed() - t0);
//   //   // }
//   // }
//   // printf("[%.3f s] adding dataset of vectors into index\n", elapsed() - t0);
//   // alg_hnsw->saveIndex("hnsw_index.bin");
//   alg_hnsw->loadIndex("hnsw_index.bin", &space);

//   size_t dout, nout;
//   float *queries = bvecs_read("/ssd_root/dataset/ann_sift1b/bigann_query.bvecs", 10000, &dout, &nout);
//   // cout<<dout<<" "<<nout<<endl;
//   // for(int i=0;i<10;i++)cout<<queries[i]<<endl;

//   size_t kk, nq;
//   int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs", &kk, &nq);
//   int64_t *gt = new int64_t[nq * k];
//   gt = new int64_t[kk * nq];
//   // FILE *fff = fopen("gt.txt", "w");
//   for (int i = 0; i < nq; i++)
//   {
//     for (int j = 0; j < k; j++)
//     {
//       gt[i * k + j] = gt_int[i * kk + j];
//       // fprintf(fff, "%d ", gt[i * k + j]);
//     }
//     // fprintf(fff, "\n");
//   }
//   printf("loading queries and gt\n");
//   int param[8] = {100, 200, 300, 400, 500, 600, 700, 800};
//   for (int mm = 0; mm < 8; mm++)
//   {
//     // mm = 2;
//     printf("***************ef: %d***************\n", param[mm]);
//     alg_hnsw->setEf(param[mm]);
//     int *I = new int[nq * k];
//     double t1 = elapsed();
//     // FILE *ff = fopen("ans.txt", "w");
//     for (int i = 0; i < nq; i++)
//     {
//       // printf("%d\n",i);
//       // exit(0);

//       std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(queries + i * dim, k);
//       for (int j = k - 1; j >= 0; j--)
//       {
//         I[i * k + j] = (int)(result.top().second);
//         result.pop();
//         // fprintf(ff,"%d ",(int)(result.top().second));
//       }
//       // fprintf(ff, "\n");
//     }
//     // exit(0);
//     printf("[%.3f s] adding dataset of vectors into index\n", elapsed() - t1);
//     int n2_100 = 0;
//     for (int i = 0; i < nq; i++)
//     {
//       std::map<float, int> umap;
//       for (int j = 0; j < k; j++)
//       {
//         umap.insert({gt[i * k + j], 0});
//       }
//       for (int l = 0; l < k; l++)
//       {
//         if (umap.find(I[i * k + l]) != umap.end())
//         {
//           n2_100++;
//         }
//       }
//       umap.clear();
//     }
//     delete[] I;
//     printf("Intersection R@100 = %.4f\n", n2_100 / float(nq * k));
//   }

//   // // Query the elements for themselves and measure recall
//   // float correct = 0;
//   // for (int i = 0; i < max_elements; i++) {
//   //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
//   //     hnswlib::labeltype label = result.top().second;
//   //     if (label == i) correct++;
//   // }
//   // float recall = correct / max_elements;
//   // std::cout << "Recall: " << recall << "\n";

//   // // Serialize index
//   // std::string hnsw_path = "hnsw.bin";
//   // alg_hnsw->saveIndex(hnsw_path);
//   // delete alg_hnsw;

//   // // Deserialize index and check recall
//   // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
//   // correct = 0;
//   // for (int i = 0; i < max_elements; i++) {
//   //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
//   //     hnswlib::labeltype label = result.top().second;
//   //     if (label == i) correct++;
//   // }
//   // recall = (float)correct / max_elements;
//   // std::cout << "Recall of deserialized index: " << recall << "\n";

//   // delete[] data;
//   // delete alg_hnsw;
//   return 0;
// }