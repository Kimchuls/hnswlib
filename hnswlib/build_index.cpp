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

struct Config {
    int dim;
    long max_elements;
    int nb;
    int M;
    int ef_construction;
    int lrange;
    int rrange;
    std::string base_filepath;
    std::string query_filepath;
    std::string groundtruth_filepath;
    std::string index_path;
    std::vector<int> efs_array;
};

// 简单的 key=value 配置文件解析函数
Config loadConfig(const std::string &filename) {
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
            size_t end = s.find_last_not_of(" \t\r\n");
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
    cfg.dim = std::stoi(kv["dim"]);
    cfg.max_elements = std::stol(kv["max_elements"]);
    cfg.nb = std::stoi(kv["nb"]);
    cfg.M = std::stoi(kv["M"]);
    cfg.ef_construction = std::stoi(kv["ef_construction"]);
    cfg.lrange = std::stoi(kv["lrange"]);
    cfg.rrange = std::stoi(kv["rrange"]);
    cfg.base_filepath = kv["base_filepath"];
    cfg.index_path = kv["index_path"];

    return cfg;
}

// 修改后的 workload 函数：从配置文件读取所有参数
void workload(const std::string &config_path) {
    // 读取配置
    Config cfg = loadConfig(config_path);
    // 直接使用 cfg 中的字段代替原来硬编码的值
    int dim = cfg.dim;
    size_t max_elements = cfg.max_elements;
    size_t nb = cfg.nb;
    int M = cfg.M;
    int ef_construction = cfg.ef_construction;
    int lrange = cfg.lrange;
    int rrange = cfg.rrange;

    if(lrange < 0 || rrange > nb || lrange >= rrange) {
        std::cerr << "Invalid range specified: lrange = " << lrange << ", rrange = " << rrange << ", nb = " << nb << std::endl;
        std::exit(1);
    }

    // 创建 L2 空间与 HNSW 索引
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = nullptr;

    // 读取 base 数据集
    float *xb = new float[dim * nb];
    size_t dd2 = dim;
    size_t nt2 = nb;
    xb = read_vectors(const_cast<char *>(cfg.base_filepath.c_str()), nb, &dd2, &nt2);
    printf("loaded base vectors: %zu vectors of dimension %zu\n", nt2, dd2);

    alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    double t0 = elapsed();
    // for (int i = lrange; i < rrange; i++) {
    //     alg_hnsw0->addPoint(xb + i * dim, i);
    //     if((i+1)%200000 == 0) {
    //         printf("checkpoint: %d, [%.3f s] \n", i + 1, elapsed() - t0);
    //     }
    // }
    int thread = omp_get_max_threads();
    ParallelFor(lrange, rrange, thread, [&](size_t row, size_t threadId) { alg_hnsw0->addPoint((void *)(xb + dim * row), row); });
    printf("[%.3f s] build index (dataset size = %d - %d)\n", elapsed() - t0, lrange, rrange);

    // 保存索引到文件（路径同样可以在配置文件中指定，示例这里硬编码）
    alg_hnsw0->saveIndex(const_cast<char *>(cfg.index_path.c_str()));
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