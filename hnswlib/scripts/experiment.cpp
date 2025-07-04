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

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
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
    // char *base_filepath = "/ssd_root/dataset/turing10m/msturing-10M.fvecs";
    float *xb;
    size_t dd2 = dim;          // dimension
    size_t nt2 = max_elements; // the number of query
    // xb = fvecs_read(base_filepath, &dd2, &nt2);
    xb = bvecs_read(base_filepath, max_elements, &dd2, &nt2);
    // xb = fvecs_read(base_filepath, &dd2, &nt2);

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

        t0 = elapsed();
        // for (size_t i = 0; i < 0.25e6; i++)
        // for (size_t i = 0.25e6; i < 0.5e6; i++)
        // for (size_t i = 0.5e6; i < 0.75e6; i++)
        for (size_t i = 0.75e6; i < 1e6; i++) {
            if ((i + 1) % 100000 == 0) {
                printf("checkpoint:%d, [%.3f s] \n", i + 1, elapsed() - t0);
            }
            alg_hnsw0->addPoint(xb + i * dim, i);
            // if ((i + 1) % 10000000 == 0)
            // {
            //   alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_10M.hnsw");
            // }
        }
        // ParallelFor(0, 10e6, 80, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
        printf("[%.3f s] build index0\n", elapsed() - t0);
        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_250K.hnsw");
        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_250K_500K.hnsw");
        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_500K_750K.hnsw");
        alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_750K_1M.hnsw");

        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw");
        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_500K_900K.hnsw");
        // alg_hnsw0->saveIndex("/ssd_root/jin467/merger/turing/1M.hnsw");
        // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_10M.hnsw", &space);
        // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw", &space);
        // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw", &space);
        // alg_hnsw0->loadIndex("/ssd_root/jin467/merger/turing/1M.hnsw", &space);
        // alg_hnsw0->setEf(1000);
    }
    return;

    {
        // char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
        char *query_filepath = "/ssd_root/dataset/turing10m/msturing-query.fvecs";
        size_t nq = 10000;
        float *xq = new float[dim * nq];
        printf("loading dataset of vectors \n");
        size_t dd = dim; // dimension
        size_t nt = nq;  // the number of query
        // xq = bvecs_read(query_filepath, 10000, &dd, &nt);
        xq = fvecs_read(query_filepath, &dd, &nt);
        printf("loaded a %ld vectors in %ld dimension \n", nt, dd);
        // size_t kk = 1000, nqq = 10000;
        // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_100M.ivecs", &kk, &nqq);
        // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs", &kk, &nqq);
        // int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", &kk, &nqq);
        size_t kk = 100, nqq = 10000;
        int *gt_int = ivecs_read("/ssd_root/dataset/turing10m/msturing10M_gt100.ivecs", &kk, &nqq);
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

void workload2() {
    // SIFT10M, SIFT50M, SIFT100M, BIGANN10M, BIGANN50M, BIGANN100M
    int dim = 128;           // Dimension of the elements
    int max_elements = 10e6; // Maximum number of elements, should be known beforehand
    int nb = 10e6;
    int M = 32;               // Tightly connected with internal dimensionality of the data
                              // strongly affects the memory consumption
    int ef_construction = 40; // Controls index search speed/build speed tradeoff
    int k = 100;
    // int workload_type = SIFT10M;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    char *base_filepath = "/ssd_root/dataset/ann_sift1b/bigann_10m_base.bvecs";
    float *xb = new float[dim * nb];
    size_t dd2 = dim; // dimension
    size_t nt2 = nb;  // the number of query
    xb = bvecs_read(base_filepath, 10000000, &dd2, &nt2);
    printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

    char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
    size_t nq = 10000;
    float *xq = new float[dim * nq];
    printf("loading dataset of vectors \n");
    size_t dd = dim; // dimension
    size_t nt = nq;  // the number of query
    xq = bvecs_read(query_filepath, 10000, &dd, &nt);
    printf("loaded a %ld vectors in %ld dimension \n", nt, dd);

    size_t kk = 1000, nqq = 10000;
    int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_10M.ivecs", &kk, &nqq);
    int *gt = new int[nq * k];
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            gt[i * k + j] = gt_int[i * kk + j];
        }
    }
    printf("loaded queries and gt\n");

    double t0;
    for (int iter = 0; iter < 5; iter++) {
        printf("Iteration %d\n", iter + 1);
        alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        t0 = elapsed();
        // printf("%p, %f\n",xb + 0* dim, xb + 0 * dim);
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw0->addPoint(xb + i * dim, i);
            // if ((i + 1) % 200000 == 0) {
            //     printf("checkpoint:%d, [%.3f s] \n", i + 1, elapsed() - t0);
            // }
        }
        // ParallelFor(0, 10e6, 80, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
        printf("[%.3f s] build index-sift-10M\n", elapsed() - t0);
    }

    alg_hnsw0->saveIndex("/ssd_root/jin467/merger/indexes/bigann-index_10M.hnsw");

    t0 = elapsed();

    hnswlib::HierarchicalNSW<float> *alg_hnsw3; //= new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_10M.hnsw", &space);
    alg_hnsw3 = alg_hnsw0;

    int efs_array[] = {100, 200, 300, 400, 500, 600, 700, 800};
    int length = 8;

    for (int efs = 0; efs < length; efs++) {
        alg_hnsw3->setEf(efs_array[efs]);
        printf("set ef: %d\n", efs_array[efs]);
        for (int iter = 0; iter < 5; iter++) {
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
            printf("Intersection-merged index R@100 = %.4f\n\n", n2_100 / float(nq * k));
        }
    }
}

struct Config {
    int dim;
    long max_elements;
    int nb;
    int M;
    int ef_construction;
    int k;
    int kk;
    int nq;
    int iterations;
    std::string base_filepath;
    std::string query_filepath;
    std::string groundtruth_filepath;
    std::string index_path;
    std::vector<int> efs_array;
};

// 简单的 key=value 配置文件解析函数
Config loadConfig(const std::string& filename) {
    Config cfg;
    std::map<std::string, std::string> kv;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open config file: " << filename << std::endl;
        std::exit(1);
    }
    std::string line;
    while (std::getline(infile, line)) {
        // 去掉首尾空白
        std::istringstream linestream(line);
        std::string key;
        if (line.empty() || line[0] == '#') continue;
        if (line.find('=') == std::string::npos) continue;
        key = line.substr(0, line.find('='));
        std::string value = line.substr(line.find('=') + 1);
        // 去掉可能的空格
        auto trim = [](std::string &s) {
            size_t start = s.find_first_not_of(" \t\r\n");
            size_t end   = s.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) {
                s = "";
            } else {
                s = s.substr(start, end - start + 1);
            }
        };
        trim(key);
        trim(value);
        kv[key] = value;
    }
    infile.close();

    // 将字符串转换为对应类型
    cfg.dim                  = std::stoi(kv["dim"]);
    cfg.max_elements         = std::stol(kv["max_elements"]);
    cfg.nb                   = std::stoi(kv["nb"]);
    cfg.M                    = std::stoi(kv["M"]);
    cfg.ef_construction      = std::stoi(kv["ef_construction"]);
    cfg.k                    = std::stoi(kv["k"]);
    cfg.kk                    = std::stoi(kv["kk"]);
    cfg.nq                   = std::stoi(kv["nq"]);
    cfg.iterations           = std::stoi(kv["iterations"]);
    cfg.base_filepath        = kv["base_filepath"];
    cfg.query_filepath       = kv["query_filepath"];
    cfg.groundtruth_filepath = kv["groundtruth_filepath"];
    cfg.index_path           = kv["index_path"];

    // 解析 efs_array（用逗号分隔）
    {
        std::string list_str = kv["efs_array"];
        std::istringstream ss(list_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // 去掉空白
            auto trim = [](std::string &s) {
                size_t start = s.find_first_not_of(" \t\r\n");
                size_t end   = s.find_last_not_of(" \t\r\n");
                if (start == std::string::npos) {
                    s = "";
                } else {
                    s = s.substr(start, end - start + 1);
                }
            };
            trim(token);
            if (!token.empty()) {
                cfg.efs_array.push_back(std::stoi(token));
            }
        }
    }

    return cfg;
}


float* read_vectors(const std::string& filepath, int num, size_t *d_out, size_t *n_out) {
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
void workload(const std::string& config_path) {
    // 读取配置
    Config cfg = loadConfig(config_path);
    // 直接使用 cfg 中的字段代替原来硬编码的值
    int dim              = cfg.dim;
    long max_elements    = cfg.max_elements;
    int nb               = cfg.nb;
    int M                = cfg.M;
    int ef_construction  = cfg.ef_construction;
    int k                = cfg.k;
    int kk               = cfg.kk;
    int nq               = cfg.nq;
    int iterations       = cfg.iterations;

    // 创建 L2 空间与 HNSW 索引
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw0 = nullptr;

    // 读取 base 数据集
    float *xb = new float[dim * nb];
    size_t dd2 = dim;
    size_t nt2 = nb;
    xb = read_vectors(const_cast<char*>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
    printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);

    // 读取 query 数据集
    float *xq = new float[dim * nq];
    size_t dd = dim;
    size_t nt = nq;
    xq = read_vectors(const_cast<char*>(cfg.query_filepath.c_str()), nq, &dd, &nt);
    printf("loaded query vectors: %zu vectors of dimension %zu\n", nt, dd);

    // 读取 ground-truth
    size_t k_file = kk;
    size_t nqq = nq;
    int *gt_int = ivecs_read(const_cast<char*>(cfg.groundtruth_filepath.c_str()), &k_file, &nqq);
    int *gt = new int[nq * k];
    if(k_file > k){
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                gt[i * k + j] = gt_int[i * kk + j];
            }
        }
    }
    else{
        gt = gt_int;
    }
    printf("loaded ground-truth for %d queries, top %d\n", nq, k);

    // 多次构建索引并保存
    for (int iter = 0; iter < iterations; iter++) {
        printf("Iteration %d/%d: build index\n", iter + 1, iterations);
        // 重新创建索引
        delete alg_hnsw0;
        alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
        double t0 = elapsed();
        for (int i = 0; i < max_elements; i++) {
            alg_hnsw0->addPoint(xb + i * dim, i);
        }
        printf("[%.3f s] build index (dataset size = %ld)\n", elapsed() - t0, max_elements);
    }

    // 保存索引到文件（路径同样可以在配置文件中指定，示例这里硬编码）
    alg_hnsw0->saveIndex(const_cast<char*>(cfg.index_path.c_str()));

    // 搜索部分：遍历 efs_array
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
    // const char *value = "1";
    // if (argc >= 2) {
    //   value = argv[1];
    // }

    // setenv("OPENBLAS_NUM_THREADS", value, 1);
    // setenv("GOTO_NUM_THREADS", value, 1);
    // setenv("OMP_DYNAMIC", "false", 1);
    // setenv("OMP_NUM_THREADS", value, 1);

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