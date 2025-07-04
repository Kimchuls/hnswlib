    // printf("%d, %d\n",index1->cur_element_count.load() , index2->cur_element_count.load());
    // printf("%d, %d\n",index1->maxlevel_ , index2->maxlevel_);
    // printf("%d, %d\n",index1->M_ , index2->M_);
    // printf("%d, %d\n",index1->getCurrentElementCount() , index2->getCurrentElementCount());
    // printf("%d, %d\n",index1->ef_construction_ , index2->ef_construction_);
    // printf("%d, %d\n",index1->max_elements_ , index2->max_elements_);

for (int i = 0; i < 10; i++) {
        int cur_c = i;
        printf("cur_c: %d\n", cur_c);
        linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(cur_c, 0);
        size_t linklistCount = alg_hnsw->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        dist_t *dist = alg_hnsw->get_dist_at_level(cur_c, 0);
        printf("linksize: %ld, ", linklistCount);
        if (linklistCount > 64) {
            throw std::runtime_error("error");
        }
        for (int j = 0; j < linklistCount; j++) {
            printf("[%d, %f], ", data[j], dist[j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 10; i++) {
        int cur_c = i + 5000000;
        printf("cur_c: %d\n", cur_c);
        linklistsizeint *ll_cur = alg_hnsw->get_linklist_at_level(cur_c, 0);
        size_t linklistCount = alg_hnsw->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        dist_t *dist = alg_hnsw->get_dist_at_level(cur_c, 0);
        printf("linksize: %ld, ", linklistCount);
        if (linklistCount > 64) {
            throw std::runtime_error("error");
        }
        for (int j = 0; j < linklistCount; j++) {
            printf("[%d, %f], ", data[j], dist[j]);
        }
        printf("\n");
    }

    HierarchicalNSW<dist_t> *alg1 = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    alg1->loadIndex(location_index1, space);
    HierarchicalNSW<dist_t> *alg2 = new HierarchicalNSW<dist_t>(space, new_max_elements, M, ef_construction);
    alg2->loadIndex(location_index2, space);
    printf("end loading\n");
    for (int i = 0; i < 10; i++) {
        int cur_c = i;
        printf("cur_c: %d\n", cur_c);
        linklistsizeint *ll_cur = alg1->get_linklist_at_level(cur_c, 0);
        size_t linklistCount = alg1->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        dist_t *dist = alg1->get_dist_at_level(cur_c, 0);
        printf("linksize: %ld, ", linklistCount);
        if (linklistCount > 64) {
            throw std::runtime_error("error");
        }
        for (int j = 0; j < linklistCount; j++) {
            printf("[%d, %f], ", data[j], dist[j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 10; i++) {
        int cur_c = i;
        printf("cur_c: %d\n", cur_c);
        linklistsizeint *ll_cur = alg2->get_linklist_at_level(cur_c, 0);
        size_t linklistCount = alg2->getListCount(ll_cur);
        tableint *data = (tableint *)(ll_cur + 1);
        dist_t *dist = alg2->get_dist_at_level(cur_c, 0);
        printf("linksize: %ld, ", linklistCount);
        if (linklistCount > 64) {
            throw std::runtime_error("error");
        }
        for (int j = 0; j < linklistCount; j++) {
            printf("[%d, %f], ", data[j], dist[j]);
        }
        printf("\n");
    }


    int level=3;
    {
        index1->load_graph(level, true);
        printf("end loading me graph\n");
        for (int i = 0; i < layer_node_for_index1[level].size() && i<10; i++) {
            int cur_c = layer_node_for_index1[level][i];
            printf("cur_c: %d\n", cur_c);
            char *data = (char *)malloc(sizeof(char) * index1->data_size_);
            index1->getDataByInternalId(cur_c, data);
            printf("%f\n",((dist_t*)data)[0]);
        }
    }

    // {
    //     for (int i = 0; i < layer_node_for_index1[level].size() && i<10; i++) {
    //         int cur_c = layer_node_for_index1[level][i];
    //         printf("cur_c: %d\n", cur_c);
    //         char *data = (char *)malloc(sizeof(char) * alg1->data_size_);
    //        data =  alg1->getDataByInternalId(cur_c);
    //         printf("%f\n",((dist_t*)data)[0]);
    //     }
    // }