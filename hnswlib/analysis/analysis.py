import struct
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import numpy as np

def parse_bin_file(filepath):
    graph = {}
    with open(filepath, 'rb') as f:
        num_nodes_bytes = f.read(4)
        if not num_nodes_bytes:
            return graph
        num_nodes = struct.unpack('I', num_nodes_bytes)[0]
        for _ in range(num_nodes):
            node_id = struct.unpack('I', f.read(4))[0]
            count = struct.unpack('I', f.read(4))[0]
            neighbors = {}
            for _ in range(count):
                neighbor_id = struct.unpack('I', f.read(4))[0]
                dist = struct.unpack('f', f.read(4))[0]
                neighbors[neighbor_id] = dist
            graph[node_id] = neighbors
    return graph

def analyze_graphs(g1, g2):
    ids = sorted(set(g1.keys()) & set(g2.keys()))
    avg1, avg2 = [], []
    for node in ids:
        n1, n2 = g1[node], g2[node]
        avg1.append(sum(n1.values()) / len(n1) if n1 else 0)
        avg2.append(sum(n2.values()) / len(n2) if n2 else 0)
    return ids, avg1, avg2
  
def plot_cluster_analysis(rebuild_graph, merge_graph, output_dir):
    """
    按照 merge_graph 中每个节点的邻居数(group_size)分组，
    并分别绘制三张盒装图：
      1) rebuild_count / group_size
      2) common_count  / group_size
      3) avg_rebuild   / avg_merge
    保存到 output_dir 目录下。
    """
    # 收集所有节点的指标
    data = []
    for node, merge_neighbors in merge_graph.items():
        rebuild_neighbors = rebuild_graph.get(node, {})
        rebuild_count = len(rebuild_neighbors)
        group_size    = len(merge_neighbors)
        common_count  = len(set(rebuild_neighbors) & set(merge_neighbors))
        avg_rebuild   = (sum(rebuild_neighbors.values()) / rebuild_count
                         if rebuild_count else 0)
        avg_merge     = (sum(merge_neighbors.values())   / group_size
                         if group_size   else 0)
        data.append((rebuild_count, group_size, common_count, avg_rebuild, avg_merge))

    # 按 merge_graph 的邻居数分组
    groups = defaultdict(list)
    for reb_count, group_size, comm_count, avg_reb, avg_mer in data:
        groups[group_size].append((reb_count, comm_count, avg_reb, avg_mer))

    cluster_keys = sorted(groups.keys())

    # 为每个 group_size 生成三类比率列表
    ratios_reb_mer = [
        [reb_count / group_size if group_size else 0
         for reb_count, _, _, _ in groups[group_size]]
        for group_size in cluster_keys
    ]
    ratios_comm_mer = [
        [comm_count / group_size if group_size else 0
         for _, comm_count, _, _ in groups[group_size]]
        for group_size in cluster_keys
    ]
    ratios_avg_dist = [
        [avg_reb / avg_mer if avg_mer else 0
         for *_, avg_reb, avg_mer in groups[group_size]]
        for group_size in cluster_keys
    ]

    # 内部函数：绘图并保存
    def _save_boxplot(data_list, filename, ylabel, title):
        plt.figure()
        plt.boxplot(data_list, labels=cluster_keys)
        plt.xlabel("Merge Neighbor Count")
        plt.ylabel(ylabel)
        plt.title(title)
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"{title} 保存到 {path}")

    _save_boxplot(ratios_reb_mer,
                  'reb_mer_neighbor_ratio_box.png',
                  'Rebuild/Merge Neighbor Count Ratio',
                  'Rebuild vs Merge Neighbor Count Ratio by Cluster')

    _save_boxplot(ratios_comm_mer,
                  'common_mer_neighbor_ratio_box.png',
                  'Common/Merge Neighbor Count Ratio',
                  'Common Neighbor Ratio by Cluster')

    _save_boxplot(ratios_avg_dist,
                  'avg_dist_ratio_box.png',
                  'Avg Rebuild/Avg Merge Distance Ratio',
                  'Average Edge Length Ratio by Cluster')

def analysis():
    import re
    import pandas as pd
    import matplotlib.pyplot as plt

    log = """
    test i: 5
    matchCount: 2407
    Intersection-merged index R@100 = 0.9735

    test i: 6
    matchCount: 3362
    Intersection-merged index R@100 = 0.9736

    test i: 7
    matchCount: 4607
    Intersection-merged index R@100 = 0.9736

    test i: 8
    matchCount: 5795
    Intersection-merged index R@100 = 0.9736

    test i: 9
    matchCount: 7083
    Intersection-merged index R@100 = 0.9736

    test i: 10
    matchCount: 8408
    Intersection-merged index R@100 = 0.9737

    test i: 11
    matchCount: 9267
    Intersection-merged index R@100 = 0.9737

    test i: 12
    matchCount: 10235
    Intersection-merged index R@100 = 0.9737

    test i: 13
    matchCount: 11022
    Intersection-merged index R@100 = 0.9738

    test i: 14
    matchCount: 12318
    Intersection-merged index R@100 = 0.9737

    test i: 15
    matchCount: 13635
    Intersection-merged index R@100 = 0.9739

    test i: 16
    matchCount: 15363
    Intersection-merged index R@100 = 0.9738

    test i: 17
    matchCount: 17818
    Intersection-merged index R@100 = 0.9739

    test i: 18
    matchCount: 20832
    Intersection-merged index R@100 = 0.9739

    test i: 19
    matchCount: 23811
    Intersection-merged index R@100 = 0.9740

    test i: 20
    matchCount: 27118
    Intersection-merged index R@100 = 0.9740

    test i: 21
    matchCount: 30695
    Intersection-merged index R@100 = 0.9741

    test i: 22
    matchCount: 33721
    Intersection-merged index R@100 = 0.9742

    test i: 23
    matchCount: 35572
    Intersection-merged index R@100 = 0.9743

    test i: 24
    matchCount: 37600
    Intersection-merged index R@100 = 0.9743

    test i: 25
    matchCount: 37773
    Intersection-merged index R@100 = 0.9743

    test i: 26
    matchCount: 37853
    Intersection-merged index R@100 = 0.9745

    test i: 27
    matchCount: 37055
    Intersection-merged index R@100 = 0.9744

    test i: 28
    matchCount: 35862
    Intersection-merged index R@100 = 0.9745

    test i: 29
    matchCount: 34175
    Intersection-merged index R@100 = 0.9743

    test i: 30
    matchCount: 31881
    Intersection-merged index R@100 = 0.9742

    test i: 31
    matchCount: 30372
    Intersection-merged index R@100 = 0.9743

    test i: 32
    matchCount: 28998
    Intersection-merged index R@100 = 0.9742

    test i: 33
    matchCount: 27024
    Intersection-merged index R@100 = 0.9742

    test i: 34
    matchCount: 25348
    Intersection-merged index R@100 = 0.9742

    test i: 35
    matchCount: 23796
    Intersection-merged index R@100 = 0.9741

    test i: 36
    matchCount: 22357
    Intersection-merged index R@100 = 0.9740

    test i: 37
    matchCount: 20795
    Intersection-merged index R@100 = 0.9739

    test i: 38
    matchCount: 19827
    Intersection-merged index R@100 = 0.9739

    test i: 39
    matchCount: 18620
    Intersection-merged index R@100 = 0.9739

    test i: 40
    matchCount: 17467
    Intersection-merged index R@100 = 0.9738

    test i: 41
    matchCount: 16531
    Intersection-merged index R@100 = 0.9739

    test i: 42
    matchCount: 15714
    Intersection-merged index R@100 = 0.9738

    test i: 43
    matchCount: 14695
    Intersection-merged index R@100 = 0.9738

    test i: 44
    matchCount: 13842
    Intersection-merged index R@100 = 0.9737

    test i: 45
    matchCount: 13106
    Intersection-merged index R@100 = 0.9737

    test i: 46
    matchCount: 12106
    Intersection-merged index R@100 = 0.9737

    test i: 47
    matchCount: 11552
    Intersection-merged index R@100 = 0.9736

    test i: 48
    matchCount: 10687
    Intersection-merged index R@100 = 0.9736

    test i: 49
    matchCount: 10261
    Intersection-merged index R@100 = 0.9737

    test i: 50
    matchCount: 9680
    Intersection-merged index R@100 = 0.9736

    test i: 51
    matchCount: 8978
    Intersection-merged index R@100 = 0.9736

    test i: 52
    matchCount: 8421
    Intersection-merged index R@100 = 0.9736

    test i: 53
    matchCount: 8145
    Intersection-merged index R@100 = 0.9736

    test i: 54
    matchCount: 7524
    Intersection-merged index R@100 = 0.9735

    test i: 55
    matchCount: 7082
    Intersection-merged index R@100 = 0.9735

    test i: 56
    matchCount: 6514
    Intersection-merged index R@100 = 0.9735

    test i: 57
    matchCount: 6230
    Intersection-merged index R@100 = 0.9736

    test i: 58
    matchCount: 5869
    Intersection-merged index R@100 = 0.9735

    test i: 59
    matchCount: 5422
    Intersection-merged index R@100 = 0.9735

    test i: 60
    matchCount: 4966
    Intersection-merged index R@100 = 0.9735

    test i: 61
    matchCount: 4854
    Intersection-merged index R@100 = 0.9735

    test i: 62
    matchCount: 4439
    Intersection-merged index R@100 = 0.9735

    test i: 63
    matchCount: 4241
    Intersection-merged index R@100 = 0.9735

    test i: 64
    matchCount: 5970
    Intersection-merged index R@100 = 0.9733
    """

    pattern = r"test i: (\d+)\s+matchCount: (\d+)\s+Intersection-merged index R@100 = ([0-9.]+)"
    matches = re.findall(pattern, log)
    df = pd.DataFrame(matches, columns=["test_i", "matchCount", "R100"])
    df = df.astype({"test_i": int, "matchCount": int, "R100": float})

    df.to_csv("results.csv", index=False)
    plt.scatter(df["matchCount"], df["R100"])
    plt.xlabel("matchCount")
    plt.ylabel("R@100")
    plt.title("matchCount vs R@100")
    plt.savefig("match_vs_R100.png")

def plot_degree_bucket_distributions(merge_bin_path, output_dir, K=10000):
    """
    解析 merge.bin，根据每个节点的邻居数（degree）将节点分组，
    并对每个 degree=1..64 的组，将 node_id 划分到 K 个等宽桶中统计分布，
    最终在 output_dir 下保存 degree_<d>_dist.png。
    """
    # 1) 解析 merge.bin，获得每个节点的 degree
    degree_map = {}
    with open(merge_bin_path, 'rb') as f:
        num_nodes = struct.unpack('I', f.read(4))[0]
        for _ in range(num_nodes):
            node_id = struct.unpack('I', f.read(4))[0]
            count   = struct.unpack('I', f.read(4))[0]
            f.seek(count * 8, os.SEEK_CUR)  # 跳过 id+dist 数据
            degree_map[node_id] = count

    N = num_nodes  # 总节点数

    # 2) 为 degree=1..64 分别构建分布图
    os.makedirs(output_dir, exist_ok=True)
    for deg in range(1, 65):
        # 挑出所有 degree==deg 的节点
        ids = [nid for nid, d in degree_map.items() if d == deg]
        if not ids:
            continue

        # 初始化 K 个桶
        buckets = np.zeros(K, dtype=int)
        # 将 id 均匀映射到 [0, K)
        for nid in ids:
            idx = (nid * K) // N
            if idx >= K:
                idx = K - 1
            buckets[idx] += 1

        # 绘图并保存
        plt.figure()
        plt.bar(np.arange(K), buckets, width=1.0)
        plt.xlabel('Bucket Index')
        plt.ylabel('Node Count')
        plt.title(f'Degree {deg} Distribution Across {K} Buckets')
        out_file = os.path.join(output_dir, f'degree_{deg}_dist.png')
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close()
        print(f'Saved: {out_file}')
        
if __name__ == '__main__':
    # Determine script directory or fallback
    analysis()
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    # # File paths
    rebuild_path = os.path.join(script_dir, './analysis/rebuild.bin')
    merge_path = os.path.join(script_dir, './analysis/merge.bin')
    
    output_dir     = os.path.join(script_dir, 'degree_distributions')
    plot_degree_bucket_distributions(merge_path, output_dir, K=10000)

    # # Parse graphs
    # rebuild_graph = parse_bin_file(rebuild_path)
    # merge_graph = parse_bin_file(merge_path)
    # print("loaded graphs")
    
    # plot_cluster_analysis(rebuild_graph, merge_graph, script_dir)

    # # Analyze
    # ids, avg1, avg2 = analyze_graphs(rebuild_graph, merge_graph)
    # print("analyzed graphs")

    # # Plot: scatter of average neighbor distances
    # plt.figure()
    # plt.scatter(avg1, avg2)
    # min_val = min(min(avg1), min(avg2))
    # max_val = max(max(avg1), max(avg2))
    # plt.plot([min_val, max_val], [min_val, max_val])
    # plt.xlabel("Original Avg Distance")
    # plt.ylabel("Merged Avg Distance")
    # plt.title("Node Avg Neighbor Distance Comparison")
    # scatter_path = os.path.join(script_dir, './analysis/avg_distance_scatter.png')
    # plt.savefig(scatter_path)
    # print(f"Scatter plot saved to {scatter_path}")
    # plt.close()

    # # Plot: histogram of distance differences
    # diffs = [b - a for a, b in zip(avg1, avg2)]
    # plt.figure()
    # plt.hist(diffs, bins=50)
    # plt.xlabel("Distance Difference (Merged - Original)")
    # plt.title("Avg Neighbor Distance Difference Distribution")
    # hist_path = os.path.join(script_dir, './analysis/avg_distance_diff_hist.png')
    # plt.savefig(hist_path)
    # print(f"Histogram saved to {hist_path}")
    # plt.close()
