#include "../extension.h"

#include "../test_readfile.h"

using namespace std;
using namespace hnswlib;

// compare 2 indexes (rebuild and merge)
void workload1() {
    char *rebuild = "/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw";
    char *merge = "/ssd_root/jin467/merger/indexes/merged-index.hnsw";
    int dim = 128;
    int max_elements = 1e6;
    int M = 32;
    int ef_construction = 40;
    int k = 100;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *rebuild_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float> *merge_index = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    rebuild_index->loadIndex(rebuild, &space);
    merge_index->loadIndex(merge, &space);

    char *rebuild_text = "./analysis/rebuild.txt";
    char *merge_text = "./analysis/merge.txt";

    //   auto dump_rebuild_graph = [&](const char *filepath, hnswlib::HierarchicalNSW<float> *index) {
    //     FILE *fp = fopen(filepath, "w");
    //     if (!fp) {
    //         perror("fopen");
    //         return;
    //     }

    //     for (int i = 0; i < max_elements; i++) {
    //         hnswlib::linklistsizeint *llcur = index->get_linklist0(i);
    //         size_t count = index->getListCount(llcur);
    //         hnswlib::tableint *data = (hnswlib::tableint *)(llcur + 1);
    //         float *distData = (float *)index->get_dist_at_level(i, 0);

    //         fprintf(fp, "[%d] ", i);
    //         for (size_t j = 0; j < count; j++) {
    //             fprintf(fp, "%d:%.1f, ", data[j], distData[j]);
    //         }
    //         fprintf(fp, "\n");
    //     }

    //     fclose(fp);
    // };

    auto dump_rebuild_graph = [&](const char *filepath, hnswlib::HierarchicalNSW<float> *index) {
        FILE *fp = fopen(filepath, "wb");
        if (!fp) {
            perror("fopen");
            return;
        }
        fwrite(&max_elements, sizeof(uint32_t), 1, fp);
        for (uint32_t i = 0; i < max_elements; ++i) {
            hnswlib::linklistsizeint *llcur = index->get_linklist0(i);
            size_t count = index->getListCount(llcur);
            hnswlib::tableint *data = reinterpret_cast<hnswlib::tableint *>(llcur + 1);
            float *distData = reinterpret_cast<float *>(index->get_dist_at_level(i, 0));

            fwrite(&i, sizeof(uint32_t), 1, fp);
            uint32_t neighbor_count = static_cast<uint32_t>(count);
            fwrite(&neighbor_count, sizeof(uint32_t), 1, fp);
            for (uint32_t j = 0; j < neighbor_count; ++j) {
                uint32_t neighbor_id = static_cast<uint32_t>(data[j]);
                float dist = distData[j];
                fwrite(&neighbor_id, sizeof(uint32_t), 1, fp);
                fwrite(&dist, sizeof(float), 1, fp);
            }
        }
        fclose(fp);
    };
    dump_rebuild_graph("./analysis/rebuild.bin", rebuild_index);
    dump_rebuild_graph("./analysis/merge.bin", merge_index);
}

// analyze refine effectness
void workload2() {
    // SIFT10M, SIFT50M, SIFT100M, BIGANN10M, BIGANN50M, BIGANN100M
    int dim = 128;           // Dimension of the elements
    int max_elements = 10e6; // Maximum number of elements, should be known beforehand
    int nb = 5e6;
    int M = 32;               // Tightly connected with internal dimensionality of the data
                              // strongly affects the memory consumption
    int ef_construction = 64; // Controls index search speed/build speed tradeoff
    int k = 100;

    hnswlib::L2Space space(dim);

    float *xb = new float[dim * nb];
    size_t dd2 = dim; // dimension
    size_t nt2 = nb;  // the number of query

    char *query_filepath = "/ssd_root/dataset/ann_sift1b/bigann_query.bvecs";
    size_t nq = 10000;
    float *xq = new float[dim * nq];
    printf("loading dataset of vectors \n");
    size_t dd = dim; // dimension
    size_t nt = nq;  // the number of query
    // xq = fvecs_read(query_filepath, &dd, &nt);
    xq = bvecs_read(query_filepath, 10000, &dd, &nt);
    printf("loaded a %ld vectors in %ld dimension \n", nt, dd);

    size_t kk = 1000, nqq = 10000;
    int *gt_int = ivecs_read("/ssd_root/dataset/ann_sift1b/gnd/idx_1M.ivecs", &kk, &nqq);
    int *gt = new int[nq * k];
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            gt[i * k + j] = gt_int[i * kk + j];
        }
    }
    printf("loaded queries and gt\n");

    double t0;

    t0 = elapsed();

    hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw", &space);
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw", &space);
    // alg_hnsw2->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw", &space);
    printf("[%.3f s] build merged index\n", elapsed() - t0);
    alg_hnsw2->setEf(180);

    int i;
    // for (i = 5; i <= 2 * M; i++) 
    {
        // i=1;
        printf("test i: %d\n", i);
        t0 = elapsed();
        hnswlib::HierarchicalNSW<float> *alg_hnsw4 = hnswlib::HNSWRefinement<float>(alg_hnsw2, &space, i);
        alg_hnsw3 = alg_hnsw4;
        alg_hnsw3->setEf(200);
        // alg_hnsw4->saveIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw");
        printf("[%.3f s] build refined index\n", elapsed() - t0);
        // exit(0);

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

// analyze multi index merge
void workload3() {
    // SIFT10M, SIFT50M, SIFT100M, BIGANN10M, BIGANN50M, BIGANN100M
    int dim = 128;           // Dimension of the elements
    int max_elements = 10e6; // Maximum number of elements, should be known beforehand
    int nb = 5e6;
    int M = 32;               // Tightly connected with internal dimensionality of the data
                              // strongly affects the memory consumption
    int ef_construction = 40; // Controls index search speed/build speed tradeoff
    int k = 100;
    // int workload_type = SIFT10M;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw0 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    alg_hnsw0->setEf(200);

    // hnswlib::HierarchicalNSW<float> *alg_hnsw1 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    // alg_hnsw1->setEf(200);
    // hnswlib::HierarchicalNSW<float> *alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    // alg_hnsw2->setEf(200);
    hnswlib::HierarchicalNSW<float> *alg_hnsw5 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    alg_hnsw5->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_250K.hnsw", &space);
    hnswlib::HierarchicalNSW<float> *alg_hnsw6 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    alg_hnsw6->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_250K_500K.hnsw", &space);
    hnswlib::HierarchicalNSW<float> *alg_hnsw7 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    alg_hnsw7->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_500K_750K.hnsw", &space);
    hnswlib::HierarchicalNSW<float> *alg_hnsw8 = new hnswlib::HierarchicalNSW<float>(&space, nb, M, ef_construction);
    alg_hnsw8->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_750K_1M.hnsw", &space);


    float *xb = new float[dim * nb];
    // printf("loading dataset of vectors \n");
    size_t dd2 = dim; // dimension
    size_t nt2 = nb;  // the number of query
    printf("loaded a %ld vectors in %ld dimension \n", nt2, dd2);

    double t0;
    

    // t0 = elapsed();
    // hnswlib::HierarchicalNSW<float> *alg_hnsw1 = hnswlib::HNSWMerger<float>(alg_hnsw5, alg_hnsw6, &space, -1, -1);
    // hnswlib::HierarchicalNSW<float> *alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw7, alg_hnsw8, &space, -1, -1);
    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = hnswlib::HNSWMerger<float>(alg_hnsw2, alg_hnsw1, &space, -1, -1);

    // hnswlib::HierarchicalNSW<float> *alg_hnsw1 = hnswlib::HNSWMerger<float>(alg_hnsw5, alg_hnsw6, &space, -1, -1);
    // hnswlib::HierarchicalNSW<float> *alg_hnsw2 = hnswlib::HNSWMerger<float>(alg_hnsw1, alg_hnsw7, &space, -1, -1);
    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = hnswlib::HNSWMerger<float>(alg_hnsw2, alg_hnsw8, &space, -1, -1);
    // printf("[%.3f s] build merged index\n", elapsed() - t0);
    // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw");

    // std::vector<hnswlib::HierarchicalNSW<float>*> indices = {alg_hnsw5, alg_hnsw6, alg_hnsw7, alg_hnsw8};
    // t0 = elapsed();
    // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = hnswlib::MultiIndexMerger<float>(indices, &space, -1, -1);
    // // // // hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);;
    // // // // alg_hnsw3->MultiIndexMerger(indices, &space, -1, -1);
    // printf("[%.3f s] build merged index\n", elapsed() - t0);
    // alg_hnsw3->saveIndex("/ssd_root/jin467/merger/indexes/multi-merged-index.hnsw");



    hnswlib::HierarchicalNSW<float> *alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, max_elements * 2, M, ef_construction);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/merged-index.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/bigann-index_1M.hnsw", &space);
    // alg_hnsw3->loadIndex("/ssd_root/jin467/merger//indexes/refined-index.hnsw", &space);
    alg_hnsw3->loadIndex("/ssd_root/jin467/merger/indexes/multi-merged-index.hnsw", &space);
    // printf("[%.3f s] build merged index\n", elapsed() - t0);
    // printf("[%.3f s] for check part\n", alg_hnsw3->time_counter_);
    alg_hnsw3->setEf(200);
    // exit(0);

    // t0 = elapsed();
    // hnswlib::HierarchicalNSW<float> *alg_hnsw4 = hnswlib::HNSWRefinement<float>(alg_hnsw3, &space, true);
    // alg_hnsw3 = alg_hnsw4;
    // alg_hnsw3->setEf(200);
    // // alg_hnsw4->saveIndex("/ssd_root/jin467/merger/indexes/refined-index.hnsw");
    // // alg_hnsw1->HNSWMerger(alg_hnsw2);
    // // alg_hnsw1->saveIndex("/ssd_root/jin467/merger/indexes/merged-cluster-index.hnsw");
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

int main() {
    // workload1();
    // workload2();
    workload3();
    return 0;
}