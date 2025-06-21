// #include "hnswlib.h"
#include "extension.h"
// #include "scripts/cluster_based_method.h"
#include "test_config.h"
#include "test_readfile.h"
#include "baseline.h"
#include "baseline2.h"
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

// 修改后的 workload 函数：从配置文件读取所有参数
void workload(const std::string &config_path) {
    // 读取配置
    Config cfg = loadConfig(config_path);
    size_t dim = cfg.dim;
    size_t max_elements = cfg.max_elements;
    size_t nb = cfg.nb;
    int M = cfg.M;
    int ef_construction = cfg.ef_construction;
    int iterations = cfg.iterations;
    int kk = cfg.kk;
    int k = cfg.k;
    int nq = cfg.nq;
    int lrange = cfg.lrange;
    int rrange = cfg.rrange;
    int thread = cfg.thread;
    int cnt = cfg.cnt;
    float alpha = cfg.alpha;
    bool save_index = cfg.save_index;

    printf("Configuration:\n");
    printf("  Workload Type: %s\n", workloadTypeToString(cfg.workload_type).c_str());
    printf("  Merge Method: %s\n", mergeMethodToString(cfg.merge_method).c_str());
    printf("  Dimension: %ld\n", dim);
    printf("  Max Elements: %ld\n", max_elements);
    printf("  Base Vectors: %ld\n", nb);
    printf("  M: %d\n", M);
    printf("  ef_construction: %d\n", ef_construction);
    printf("  k: %d\n", k);
    printf("  Query Vectors: %d\n", nq);
    printf("  Iterations: %d\n", iterations);
    printf("  Rerun: %s\n", cfg.rerun ? "true" : "false");
    printf("  Thread Count: %d\n", thread);
    printf("  cnt: %d\n", cnt);
    printf("  Alpha: %.2f\n", alpha);
    printf("  Save Index: %s\n", save_index ? "true" : "false");

    enum WorkloadType workload_type = cfg.workload_type;
    enum MergeMethod merge_method = cfg.merge_method;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = nullptr;
    hnswlib::HierarchicalNSW<float> *alg_hnsw1 = nullptr;
    hnswlib::HierarchicalNSW<float> *alg_hnsw2 = nullptr;

    float *xb = new float[dim * nb];
    size_t dd2 = dim;
    size_t nt2 = nb;
    // xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
    // printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);

    // 读取 query 数据集
    float *xq = new float[dim * nq];
    size_t dd = dim;
    size_t nt = nq;
    xq = read_vectors(const_cast<char *>(cfg.query_filepath.c_str()), nq, &dd, &nt);
    printf("loaded query vectors: %zu vectors of dimension %zu\n", nt, dd);

    // 读取 ground-truth
    size_t k_file = kk;
    size_t nqq = nq;
    int *gt_int = ivecs_read(const_cast<char *>(cfg.groundtruth_filepath.c_str()), &k_file, &nqq);
    int *gt = new int[nq * k];
    if (k_file > k) {
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                gt[i * k + j] = gt_int[i * kk + j];
            }
        }
    } else {
        gt = gt_int;
    }
    printf("loaded ground-truth for %d queries, top %d\n", nq, k);

    printf("Merge method: %s\n", mergeMethodToString(merge_method).c_str());
    if (merge_method == REBUILD) {
        if (cfg.rerun) {
            xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
            printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                delete alg_hnsw0;
                alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
                double t0 = elapsed();
                ParallelFor(lrange, rrange, thread, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string index_path = cfg.index_path[0];
                    alg_hnsw0->saveIndex(index_path);
                    printf("Saved index to: %s\n", index_path.c_str());
                } else {
                    printf("Index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
        } else {
            std::string index_path = cfg.index_path[0];
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path, &space);
            printf("Loaded merged index from: %s\n", index_path.c_str());
        }
    } else if (merge_method == INSERT) {
        if (cfg.rerun) {
            xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
            printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                std::string index_path = (cfg.index_path)[0];
                printf("Loading index from: %s\n", index_path.c_str());
                alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
                alg_hnsw0->loadIndex(index_path, &space);
                std::cout << "Loaded index from: " << index_path << std::endl;

                double t0 = elapsed();
                ParallelFor(lrange, rrange, thread, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
            }
        }
        printf("Insert task do not re-test the performance.\n");
        return;
    } else if (merge_method == TWO_MERGE) {
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw0, alg_hnsw1, &space);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = "/ssd_root/jin467/merger/indexes/merged-index_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/merged-index_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == ABLATION_C) {
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw0, alg_hnsw1, &space, cnt, alpha);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = "/ssd_root/jin467/merger/indexes/ablation_c_" + std::to_string(cnt) + "_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/ablation_c_" + std::to_string(cnt) + "_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == ES) {
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);

                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger_ES<float>(alg_hnsw0, alg_hnsw1, &space);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);

                if (save_index) {
                    std::string merged_index_path = "/ssd_root/jin467/merger/indexes/es_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/es_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == NGM) {
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger_Naive<float>(alg_hnsw0, alg_hnsw1, &space);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = "/ssd_root/jin467/merger/indexes/ngm_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saved merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Merged index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/ngm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == IGTM) {
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger_IGTM<float>(alg_hnsw0, alg_hnsw1, &space);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = "/ssd_root/jin467/merger/indexes/igtm_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saving merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/igtm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else if (merge_method == CGTM) {
        if (cfg.rerun == true) {
            std::vector<std::string> index_path = cfg.index_path;
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(index_path[0], &space);
            alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw1->loadIndex(index_path[1], &space);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                double t0 = elapsed();
                alg_hnsw2 = hnswlib::HNSWMerger_CGTM<float>(alg_hnsw0, alg_hnsw1, &space);
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
                if (save_index) {
                    std::string merged_index_path = "/ssd_root/jin467/merger/indexes/cgtm_" + workloadTypeToString(workload_type) + ".hnsw";
                    alg_hnsw2->saveIndex(merged_index_path);
                    printf("Saving merged index to: %s\n", merged_index_path.c_str());
                } else {
                    printf("Index not saved, rerun with save_index flag in config to save the index.\n");
                }
            }
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/cgtm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    } else {
        std::cerr << "Unknown merge method!" << std::endl;
        exit(1);
    }

    printf("Start searching\n");
    for (int ef_val : cfg.efs_array) {
        alg_hnsw0->setEf(ef_val);
        printf("set ef = %d\n", ef_val);
        for (int iter = 0; iter < 5; iter++) {
            double t_search = 0.0;
            double t0 = elapsed();
            int *I = new int[nq * k];
            for (int i = 0; i < nq; i++) {
                double t1 = elapsed();
                auto result = alg_hnsw0->searchKnn(xq + i * dim, k);
                t_search += elapsed() - t1;
                for (int j = k - 1; j >= 0; j--) {
                    I[i * k + j] = static_cast<int>(result.top().second);
                    result.pop();
                }
            }
            double t_all = elapsed() - t0;
            printf("[search time: %.3f s, pure query time: %.3f s] ef=%d\n", t_all, t_search, ef_val);

            int correct = 0;
            for (int i = 0; i < nq; i++) {
                std::map<int, int> umap;
                for (int j = 0; j < k; j++) {
                    umap[gt[i * k + j]] = 1;
                }
                for (int j = 0; j < k; j++) {
                    if (umap.find(I[i * k + j]) != umap.end()) {
                        correct++;
                    }
                }
            }
            float recall = correct / static_cast<float>(nq * k);
            printf("Intersection-merged index R@100 = %.4f\n\n", recall);

            delete[] I;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>" << std::endl;
        return 1;
    }
    std::string config_path = argv[1];
    workload(config_path);
    return 0;
}