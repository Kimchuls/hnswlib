import os
import matplotlib.pyplot as plt

from collections import defaultdict
import time

class SortedDict:
    def __init__(self):
        self.data = defaultdict(list)

    def insert(self, key: int, value: tuple[int, int]):
        self.data[key].append(value)
        self.data[key].sort(key=lambda x: x[1])

    def get(self, key: int):
        return self.data.get(key, [])

    def display(self):
        for k, v in self.data.items():
            print(f"{k}: {v}")


def refine(file, sorted_dict):
    lines = file.readlines()
    for line in lines:
        line = line.replace(" ", "").replace("\n", "").split(":")
        id = int(line[0])
        line = line[1].split(";")[:-1]
        for item in line:
            item = item.replace('(','').replace(')','').split(",")
            item[0] = int(item[0])
            item[1] = int(item[1])
            item = tuple(item)
            sorted_dict.insert(id, item)
        # exit(0)

def compare(dict1 = None, dict2 = None):
    def compare_lists(list1, list2):
        set1, set2 = set(list1), set(list2)

        only_in_list1 = set1 - set2 
        only_in_list2 = set2 - set1 
        common_elements = set1 & set2  

        return {
            "only_in_list1": sorted(only_in_list1, key=lambda x: x[1]),
            "only_in_list2": sorted(only_in_list2, key=lambda x: x[1]),
            "common_elements": sorted(common_elements, key=lambda x: x[1]),
        }
    
    def compute_ratios(differences):
        list1_sizes = []
        list2_sizes = []
        ratio_all = []
        ratio_only_in_list1 = []
        ratio_only_in_list2 = []
        ratio_common = []

        for key, diff in differences.items():
            size1 = len(diff['only_in_list1']) + len(diff['common_elements'])
            size2 = len(diff['only_in_list2']) + len(diff['common_elements'])

            if size1 > 0 and size2 > 0:
                ratio_all.append(size2 / size1)
            
            if len(diff['only_in_list1']) > 0 and size1 > 0:
                ratio_only_in_list1.append(len(diff['only_in_list1']) / size1)
            
            if len(diff['only_in_list2']) > 0 and size1 > 0:
                ratio_only_in_list2.append(len(diff['only_in_list2']) / size1)
            
            if len(diff['common_elements']) > 0 and size1 > 0:
                ratio_common.append(len(diff['common_elements']) / size1)

        return ratio_all, ratio_only_in_list1, ratio_only_in_list2, ratio_common

    def plot_ratios(ratio_all, ratio_only_in_list1, ratio_only_in_list2, ratio_common):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(ratio_all, bins=2000, alpha=0.7, color='b')
        axes[0, 0].set_title("Ratio of total elements in list2 vs list1")
        
        axes[0, 1].hist(ratio_only_in_list1, bins=2000, alpha=0.7, color='r')
        axes[0, 1].set_title("Ratio of only_in_list1 elements vs list1")
        
        axes[1, 0].hist(ratio_only_in_list2, bins=2000, alpha=0.7, color='g')
        axes[1, 0].set_title("Ratio of only_in_list2 elements vs list1")
        
        axes[1, 1].hist(ratio_common, bins=2000, alpha=0.7, color='purple')
        axes[1, 1].set_title("Ratio of common elements vs list1")
        
        plt.tight_layout()
        plt.savefig("indexes/ratios.png")

    # f=open("indexes/diff.txt", "w")
    diff_results = {}
    # for key in dict1.keys():
    #     value1, value2 = dict1.get(key), dict2.get(key)
    #     diff_results[key] = compare_lists(value1, value2)
    #     f.write(f"{key}: {diff_results[key]}\n")
    # f.close()
    
    start = time.time()
    with open("indexes/diff.txt", "r") as f:
        for line in f:
            key, value = line.split(":", 1) 
            diff_results[int(key.strip())] = eval(value.strip())  
            
    end = time.time()
    print(f"Time taken to read diff file: {end - start}")
    
    ratios = compute_ratios(diff_results)
    plot_ratios(*ratios)
    # return diff_results
    

if __name__ == "__main__":

    # file0 = open("indexes/neighbors0.txt", "r")
    # file1 = open("indexes/neighbors1.txt", "r")
    # file2 = open("indexes/neighbors2.txt", "r")
    # file3 = open("indexes/neighbors3.txt", "r")

    # sorted_dict0 = SortedDict()
    # sorted_dict1 = SortedDict()
    # sorted_dict2 = SortedDict()
    # sorted_dict3 = SortedDict()

    # refine(file0, sorted_dict0)
    # print("refined")
    # refine(file1, sorted_dict1)
    # refine(file2, sorted_dict2)
    # refine(file3, sorted_dict3)
    # print("refined")
    
    # diff1 = compare(sorted_dict0.data, sorted_dict3.data)
    diff1 = compare()
    
