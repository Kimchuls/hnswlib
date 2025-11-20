#include "hnswalg.h"
#include "hnswlib.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#include <omp.h>
using namespace std;

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
            if (id % 200000 == 0) {
                std::cout << "Processed " << id << " items." << std::endl;
            }
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                        if (id % 200000 == 0) {
                            std::cout << "Processed " << id << " items." << std::endl;
                        }
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
  FILE *f = fopen(fname, "r");
  printf("fvecs_read: %s\n", fname);
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
  // cout << "Read " << nr << " vectors of dimension " << d << endl;
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

int main() {
    size_t d, n;
    size_t d_query, n_query;
    size_t d_ground_truth, n_ground_truth;

    // // read sift dataset
    // string sift_path = "/scratch1/jin467/dataset/sift/sift_base.fvecs";
    // d = 128;
    // n = 1000000;
    // float *data = fvecs_read(sift_path.c_str(), &d, &n);
    // cout << "Read " << n << " vectors of dimension " << d << endl;
    // // read query dataset
    // string query_path = "/scratch1/jin467/dataset/sift/sift_query.fvecs";
    // d_query = 128;
    // n_query = 10000;
    // float *query_data = fvecs_read(query_path.c_str(), &d_query, &n_query);
    // cout << "Read " << n_query << " query vectors of dimension " << d_query << endl;
    // // read ground truth dataset
    // string ground_truth_path = "/scratch1/jin467/dataset/sift/sift_groundtruth.ivecs";
    // d_ground_truth = 100;
    // n_ground_truth = 10000;
    // int *ground_truth_data = ivecs_read(ground_truth_path.c_str(), &d_ground_truth, &n_ground_truth);
    // cout << "Read " << n_ground_truth << " ground truth vectors of dimension " << d_ground_truth << endl;

    // read gist dataset
    string gist_path = "/scratch1/jin467/dataset/gist/gist_base.fvecs";
    d = 960;
    n = 1000000;
    float *data = fvecs_read(gist_path.c_str(), &d, &n);
    cout << "Read " << n << " vectors of dimension " << d << endl;
    // read query dataset
    string query_path = "/scratch1/jin467/dataset/gist/gist_query.fvecs";
    d_query = 960;
    n_query = 1000;
    float *query_data = fvecs_read(query_path.c_str(), &d_query, &n_query);
    cout << "Read " << n_query << " query vectors of dimension " << d_query << endl;
    // read ground truth dataset
    string ground_truth_path = "/scratch1/jin467/dataset/gist/gist_groundtruth.ivecs";
    d_ground_truth = 100;
    n_ground_truth = 1000;
    int *ground_truth_data = ivecs_read(ground_truth_path.c_str(), &d_ground_truth, &n_ground_truth);
    cout << "Read " << n_ground_truth << " ground truth vectors of dimension " << d_ground_truth << endl;

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, n, 16, 200);
    int thread = omp_get_max_threads();
    ParallelFor(0, n, thread, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(data + d * row), row); });
    // for (size_t i = 0; i < n; i++) {
    //     //print 3 digit
    //     for(size_t j = 0; j < 3; j++) {
    //         cout << data[i * d + j] << " ";
    //     }
    //     cout << endl;
    //     alg_hnsw0->addPoint((void *)(data + d * i), i);
    // }
    // alg_hnsw0->saveIndex("/scratch1/jin467/indexes/sift_index.hnsw");
    // cout << "Saved index to sift_index.hnsw" << endl;

    alg_hnsw0->saveIndex("/scratch1/jin467/indexes/gist_index.hnsw");
    cout << "Saved index to gist_index.hnsw" << endl;
    return 0;
}