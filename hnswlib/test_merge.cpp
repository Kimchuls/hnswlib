// #include "hnswlib.h"
#include "extension.h"
// #include "scripts/cluster_based_method.h"
// #include "./scripts/test_config.h"
#include "test_readfile.h"
#include <cstdint> // For int64_t
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream> // For stringstream
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <thread>
using namespace std;

template <class Function> inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
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
void create_index() {
    int dim = 128;
    int max_elements = 10e6;
    int M = 32;
    int ef_construction = 40;
    int k = 100;
    // int workload_type = SIFT10M;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_500K.hnsw", &space);
    // printf("alg_hnsw0->M_: %ld\n", alg_hnsw0->M_);
    // printf("alg_hnsw0->ef_construction_: %ld\n", alg_hnsw0->ef_construction_);
    // exit(0);
    // double t0;
    // t0 = elapsed();
    // printf("size: %ld\n", alg_hnsw0->cur_element_count.load());
    // printf("time consuming: %.3f s\n", elapsed() - t0);
    // exit(0);

    // char *base_filepath = "/ssd_root/dataset/sift1m/sift_base.fvecs";
    char *base_filepath = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";
    float *xb;
    size_t dd2 = dim;          // dimension
    size_t nt2 = max_elements; // the number of query
    // xb = fvecs_read(base_filepath, &dd2, &nt2);
    xb = bvecs_read(base_filepath, max_elements, &dd2, &nt2);

    printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

    double t0;
    {
        //   xb = load_and_convert_to_float(base_filepath, 50000000, &dd2, &nt2);
        // printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);
        //   t0 = elapsed();
        //   for (int64_t i = 0; i < 50000000; i++)
        //   {
        //     if ((i + 1) % 100000 == 0)
        //     {
        //       printf("checkpoint:%d, [%.3f s] \n", i + 1, elapsed() - t0);
        //     }
        //     alg_hnsw0->addPoint(xb + i * dim, i);
        //     if ((i + 1) % 10000000 == 0)
        //     {
        //       alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_50M.hnsw");
        //     }
        //   }
        //   alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_50M.hnsw");
        //   printf("[%.3f s] build index0\n", elapsed() - t0);
    }

    {
        //   xb = load_and_convert_to_float_range(base_filepath, 0, 1e6, &dd2, &nt2);
        //   printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

        // t0 = elapsed();
        // for (int64_t i = 0; i < 1e6; i++)
        // {
        //   if ((i + 1) % 100000 == 0)
        //   {
        //     printf("checkpoint:%d, [%.3f s] \n", i + 1, elapsed() - t0);
        //   }
        //   alg_hnsw0->addPoint(xb + i * dim, i);
        //   // if ((i + 1) % 10000000 == 0)
        //   // {
        //   //   alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_10M.hnsw");
        //   // }
        // }
        // // ParallelFor(0, 1e6, 1, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
        // printf("[%.3f s] build index0\n", elapsed() - t0);

        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw");
        // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_10M.hnsw", &space);
        // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw", &space);
        alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw", &space);
        alg_hnsw0->setEf(200);
    }

    {
        char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
        size_t nq = 10000;
        float *xq = new float[dim * nq];
        printf("loading dataset of vectors \n");
        size_t dd = dim; // dimension
        size_t nt = nq;  // the number of query
        xq = bvecs_read(query_filepath, 10000, &dd, &nt);
        printf("loaded a %ld vectors in %ld dimension \n", nt, dd);
        size_t kk = 1000, nqq = 10000;
        // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_100M.ivecs", &kk, &nqq);
        // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs", &kk, &nqq);
        int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", &kk, &nqq);
        int *gt = new int[nq * k];
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                gt[i * k + j] = gt_int[i * kk + j];
            }
        }
        printf("loaded queries and gt\n");

        double t1, t11 = 0.0;
        int *I = new int[nq * k];
        t0 = elapsed();
        for (int i = 0; i < nq; i++) {
            t1 = elapsed();
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw0->searchKnn(xq + i * dim, k);
            t11 += elapsed() - t1;
            for (int j = k - 1; j >= 0; j--) {
                I[i * k + j] = (int)(result.top().second);
                result.pop();
            }
        }
        printf("[%.3f & %.3f s] searching on merged index\n", t11, elapsed() - t0);

        int n2_100 = 0;

        for (int i = 0; i < nq; i++) {
            std::map<float, int> umap;
            for (int j = 0; j < k; j++) {
                umap.insert({gt[i * k + j], 0});
            }
            for (int l = 0; l < k; l++) {
                if (umap.find(I[i * k + l]) != umap.end()) {
                    n2_100++;
                }
            }
            umap.clear();
        }
        printf("\nIntersection-merged index R@100 = %.4f\n", n2_100 / float(nq * k));
    }
    // exit(0);
}

void workload() {
    // SIFT10M, SIFT50M, SIFT100M, BIGANN10M, BIGANN50M, BIGANN100M
    int dim = 128;           // Dimension of the elements
    int max_elements = 10e6; // Maximum number of elements, should be known beforehand
    int nb = 5e6;
    int M = 32;               // Tightly connected with internal dimensionality of the data
                              // strongly affects the memory consumption
    int ef_construction = 64; // Controls index search speed/build speed tradeoff
    int k = 100;
    // int workload_type = SIFT10M;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    alg_hnsw0->setEf(200);

    hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    alg_hnsw1->setEf(200);
    hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    alg_hnsw2->setEf(200);

    // char *base_filepath = "/ssd_root/dataset/sift1m/sift_base.fvecs";
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
    // alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_900K.hnsw", &space);
    // alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_9M.hnsw", &space);
    // alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_50M.hnsw", &space);
    // alg_hnsw1->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_5M.hnsw", &space);

    // for (int i = 500000; i < 1000000; i++)
    // {
    //   alg_hnsw2->addPoint(xb + i * dim, i);
    // }
    // alg_hnsw2->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index2.hnsw");
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/index2.hnsw", &space);
    alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_500K_1M.hnsw", &space);
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_900K_1M.hnsw", &space);
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_9M_10M.hnsw", &space);
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_50M_100M.hnsw", &space);
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_5M_10M.hnsw", &space);
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
    alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw");

    // std::vector<hnswlib::HierarchicalNSW<float>*> indices = {alg_hnsw1, alg_hnsw2};
    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = hnswlib::MultiIndexMerger<float>(indices, &space, -1, -1);
    // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/multi-merged-index.hnsw");

    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space,
    // alg_hnsw1->max_elements_+alg_hnsw2->max_elements_, alg_hnsw1->M_, alg_hnsw1->ef_construction_);
    // alg_hnsw3->HNSWMerger(alg_hnsw1, alg_hnsw2, 1);
    // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-balanced-index-opt.hnsw");

    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space,
    // alg_hnsw1->max_elements_+alg_hnsw2->max_elements_, alg_hnsw1->M_, alg_hnsw1->ef_construction_);
    // alg_hnsw3->HNSWMerger(alg_hnsw1, alg_hnsw2);
    // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw");

    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger//indexes/refined-index.hnsw", &space);
    printf("[%.3f s] build merged index\n", elapsed() - t0);
    // printf("[%.3f s] for check part\n", alg_hnsw3->time_counter_);
    alg_hnsw3->setEf(200);
    // exit(0);

    // t0 = elapsed();
    // hnswlib::HierarchicalNSW<float> *alg_hnsw4 = hnswlib::HNSWRefinement<float>(alg_hnsw3, &space);
    // alg_hnsw3 = alg_hnsw4;
    // alg_hnsw3->setEf(200);
    // alg_hnsw4->saveIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw");
    // // alg_hnsw1->HNSWMerger(alg_hnsw2);
    // // alg_hnsw1->saveIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw");
    // printf("[%.3f s] build refined index\n", elapsed() - t0);
    // // exit(0);

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
    // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_100M.ivecs", &kk, &nqq);
    // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs", &kk, &nqq);
    int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", &kk, &nqq);
    int *gt = new int[nq * k];
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            gt[i * k + j] = gt_int[i * kk + j];
        }
    }
    printf("loaded queries and gt\n");

    double t1, t11 = 0.0;
    int *I = new int[nq * k];
    t0 = elapsed();
    for (int i = 0; i < nq; i++) {
        t1 = elapsed();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw3->searchKnn(xq + i * dim, k);
        t11 += elapsed() - t1;
        for (int j = k - 1; j >= 0; j--) {
            I[i * k + j] = (int)(result.top().second);
            result.pop();
        }
    }
    printf("[%.3f & %.3f s] searching on merged index\n", t11, elapsed() - t0);

    int n2_100 = 0;

    for (int i = 0; i < nq; i++) {
        std::map<float, int> umap;
        for (int j = 0; j < k; j++) {
            umap.insert({gt[i * k + j], 0});
        }
        for (int l = 0; l < k; l++) {
            if (umap.find(I[i * k + l]) != umap.end()) {
                n2_100++;
            }
        }
        umap.clear();
    }
    printf("\nIntersection-merged index R@100 = %.4f\n", n2_100 / float(nq * k));
}

int main(int argc, char *argv[]) {
    // const char *value = "1";
    // if (argc >= 2) {
    //   value = argv[1];
    // }

    // setenv("OPENBLAS_NUM_THREADS", value, 1);
    // setenv("GOTO_NUM_THREADS", value, 1);
    // setenv("OMP_DYNAMIC", "false", 1);
    // setenv("OMP_NUM_THREADS", value, 1);

    create_index();
    // workload();

    return 0;
}