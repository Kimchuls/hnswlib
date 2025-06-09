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

float *read_vectors(const std::string &filepath, int num, size_t *d_out, size_t *n_out) {
    if (filepath.size() >= 6) {
        std::string suffix2 = filepath.substr(filepath.size() - 6); // ".fvecs" 或 ".bvecs"
        if (suffix2 == ".fvecs") {
            return fvecs_read(filepath.c_str(), d_out, n_out);
        }
        if (suffix2 == ".bvecs") {
            return bvecs_read(filepath.c_str(), num, d_out, n_out);
        }
    }
    std::cerr << "Unsupported vector file format: " << filepath << std::endl;
    std::exit(1);
}
// 修改后的 workload 函数：从配置文件读取所有参数
void workload(const std::string &config_path) {
    // 读取配置
    Config cfg = loadConfig(config_path);
    int dim = cfg.dim;
    long max_elements = cfg.max_elements;
    int nb = cfg.nb;
    int M = cfg.M;
    int ef_construction = cfg.ef_construction;
    int iterations = cfg.iterations;
    int kk = cfg.kk;
    int k = cfg.k;
    int nq = cfg.nq;
    int lrange = cfg.lrange;
    int rrange = cfg.rrange;

    printf("Configuration:\n");
    printf("  Workload Type: %s\n", workloadTypeToString(cfg.workload_type).c_str());
    printf("  Merge Method: %s\n", mergeMethodToString(cfg.merge_method).c_str());
    printf("  Dimension: %d\n", dim);
    printf("  Max Elements: %ld\n", max_elements);
    printf("  Base Vectors: %d\n", nb);
    printf("  M: %d\n", M);
    printf("  ef_construction: %d\n", ef_construction);
    printf("  k: %d\n", k);
    printf("  Query Vectors: %d\n", nq);
    printf("  Iterations: %d\n", iterations);
    printf("  Rerun: %s\n", cfg.rerun ? "true" : "false");

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
        // alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        if (cfg.rerun) {
            xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
            printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);
            for (int i = 0; i < iterations; i++) {
                printf("Iteration %d/%d\n", i + 1, iterations);
                delete alg_hnsw0;
                alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
                double t0 = elapsed();
                for (int i = 0; i < max_elements; i++) {
                    alg_hnsw0->addPoint(xb + i * dim, i);
                    // if ((i + 1) % 200000 == 0) {
                    //     printf("Checkpoint: %d, [%.3f s]\n", i + 1, elapsed() - t0);
                    // }
                }
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
            }
            std::string index_path = cfg.index_path[0];
            alg_hnsw0->saveIndex(index_path);
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
                for (size_t j = lrange; j < rrange; j++) {
                    alg_hnsw0->addPoint(xb + j * dim, j);
                    // for (size_t xx = 0; xx < 5; xx++) {
                    //     printf("[%f, %p]\n", *xb, xb);
                    // printf("[%f, %p]\n", *(xb + j * dim) , xb + j * dim );
                    //     printf("[%lld, %lld, %lld]\n", j, dim, xx);
                    //     printf("[%f]%p, ", xb + j * dim + xx, xb + j * dim + xx);
                    //     printf("[%f]%p, ", xb + 0 * dim + xx, xb + 0 * dim + xx);
                    // }
                    // printf("\n");
                    if ((j + 1) % 100000 == 0) {
                        printf("Checkpoint: %d, [%.3f s]\n", j + 1, elapsed() - t0);
                    }
                }
                printf("Total time for insertion: %.3f s\n", elapsed() - t0);
            }
        }
        printf("Insert task don't re-test the performance.\n");
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

                // delete alg_hnsw0;
                // delete alg_hnsw1;
            }
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/merged-index_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw2->saveIndex(merged_index_path);
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/merged-index_" + workloadTypeToString(workload_type) + ".hnsw";
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

                // delete alg_hnsw0;
                // delete alg_hnsw1;
            }
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/es_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw2->saveIndex(merged_index_path);
            alg_hnsw0 = alg_hnsw2;
        } else {
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/es_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
            alg_hnsw0->loadIndex(merged_index_path, &space);
            printf("Loaded merged index from: %s\n", merged_index_path.c_str());
        }
    // } else if (merge_method == MULTI_TWO_MERGE) {
    //     for (int i = 0; i < iterations; i++) {
    //         printf("Iteration %d/%d\n", i + 1, iterations);
    //         std::vector<std::string> index_path = cfg.index_path;
    //         std::vector<hnswlib::HierarchicalNSW<float> *> indices;
    //         for (const auto &path : index_path) {
    //             hnswlib::HierarchicalNSW<float> *index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    //             index->loadIndex(path, &space);
    //             indices.push_back(index);
    //         }

    //         double t0 = elapsed();
    //         alg_hnsw2 = hnswlib::HNSWMerger<float>(indices[0], indices[1], &space, -1, -1);
    //         for (size_t j = 2; j < indices.size(); j++) {
    //             alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw2, indices[j], &space, -1, -1);
    //         }
    //         printf("Total time for insertion: %.3f s\n", elapsed() - t0);
    //     }
    //     std::string merged_index_path = "/ssd_root/jin467/merger/indexes/multi-2way-merged_" + workloadTypeToString(workload_type) + ".hnsw";
    //     alg_hnsw2->saveIndex(merged_index_path);
    //     alg_hnsw0 = alg_hnsw2;
        // } else if (merge_method == MULTI_MERGE) {
        //     for (int i = 0; i < iterations; i++) {
        //         printf("Iteration %d/%d\n", i + 1, iterations);
        //         std::vector<std::string> index_path = cfg.index_path;
        //         std::vector<hnswlib::HierarchicalNSW<float> *> indices;
        //         for (const auto &path : index_path) {
        //             hnswlib::HierarchicalNSW<float> *index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        //             index->loadIndex(path, &space);
        //             indices.push_back(index);
        //         }

        //         double t0 = elapsed();
        //         alg_hnsw2 = hnswlib::MultiIndexMerger<float>(indices, &space, -1, -1);
        //         printf("Total time for insertion: %.3f s\n", elapsed() - t0);
        //     }
        //     std::string merged_index_path = "/ssd_root/jin467/merger/indexes/multi-merged_" + workloadTypeToString(workload_type) + ".hnsw";
        //     alg_hnsw2->saveIndex(merged_index_path);
        //     alg_hnsw0 = alg_hnsw2;
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
            }
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/ngm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw2->saveIndex(merged_index_path);
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
            }
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/igtm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw2->saveIndex(merged_index_path);
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
            }
            std::string merged_index_path = "/ssd_root/jin467/merger/indexes/cgtm_" + workloadTypeToString(workload_type) + ".hnsw";
            alg_hnsw2->saveIndex(merged_index_path);
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

    printf("Start searching");
    for (int ef_val : cfg.efs_array) {
        alg_hnsw0->setEf(ef_val);
        printf("set ef = %d\n", ef_val);
        for (int iter = 0; iter < iterations; iter++) {
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

            // 计算 Recall@100
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
    const char *value = "1";
    // if (argc >= 2) {
    //   value = argv[1];
    // }

    setenv("OPENBLAS_NUM_THREADS", value, 1);
    setenv("GOTO_NUM_THREADS", value, 1);
    setenv("OMP_DYNAMIC", "false", 1);
    setenv("OMP_NUM_THREADS", value, 1);

    // create_index();
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>" << std::endl;
        return 1;
    }
    std::string config_path = argv[1];
    workload(config_path);
    // workload();

    return 0;
}