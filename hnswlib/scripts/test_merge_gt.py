import numpy as np
from scipy.spatial.distance import cdist

def read_fvecs(filename, max_vectors=None):
    with open(filename, 'rb') as f:
        all_data = []
        vector_count = 0
        
        while True:
            # Read the dimensionality of the next vector (4 bytes for an int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            
            # Read the actual vector values (dim * 4 bytes for float32 values)
            vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
            
            all_data.append(vector)
            vector_count += 1
            
            # If a limit on the number of vectors is set, stop when reached
            if max_vectors and vector_count >= max_vectors:
                break
    
    return np.vstack(all_data)  # Stack all vectors into a NumPy array

def read_bvecs(filename, max_vectors=None):
    with open(filename, 'rb') as f:
        all_data = []
        vector_count = 0
        
        while True:
            # Read the dimensionality of the next vector (4 bytes for an int32)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # End of file
            dim = np.frombuffer(dim_bytes, dtype=np.int32)[0]
            
            # Read the actual vector values (dim bytes for uint8 values)
            vector = np.frombuffer(f.read(dim), dtype=np.uint8)
            
            all_data.append(vector)
            vector_count += 1
            
            # If max_vectors is set, stop when reaching the limit
            if max_vectors and vector_count >= max_vectors:
                break
    return np.vstack(all_data)  # Stack all vectors into a NumPy array


def save_top_k_results(indices_filename, distances_filename, indices, distances):
    """Save the top-k indices and distances for each query to respective files."""
    with open(indices_filename, 'w') as idx_file, open(distances_filename, 'w') as dist_file:
        for idx_list, dist_list in zip(indices, distances):
            for idx, dist in zip(idx_list, dist_list):
                idx_file.write(f"{idx}\n")  # Indices start from 1
                dist_file.write(f"{dist}\n")

def find_top_k_neighbors(base_vectors, query_vectors, k=100):
    """For each query vector, find the top k nearest neighbors in base_vectors."""
    distances = cdist(base_vectors, query_vectors)
    top_k_indices = np.argsort(distances, axis=0)[:k]  # Get the indices of the k smallest distances
    top_k_distances = np.sort(distances, axis=0)[:k]  # Get the k smallest distances
    return top_k_indices.T, top_k_distances.T  # Transpose to align with query count

# Step 1: Read base vectors (limit to first 20000 vectors)
# base_vectors = read_fvecs('/ssd_root/dataset/sift1m/sift_base.fvecs', max_vectors=20000)
base_vectors = read_bvecs("/ssd_root/dataset/ann_sift1b/parquet/bigann_base_1b-0.bvecs", max_vectors=200000)

# Step 2: Read query vectors
# query_vectors = read_fvecs('/ssd_root/dataset/sift1m/sift_query.fvecs')
query_vectors = read_bvecs("/ssd_root/dataset/ann_sift1b/bigann_query.bvecs")

# Step 3: Find top 100 nearest neighbors for each query, including distances
top_k_indices, top_k_distances = find_top_k_neighbors(base_vectors, query_vectors, k=100)

# # Step 4: Save results to "gt.txt" (indices) and "dis.txt" (distances)
save_top_k_results('../bigann200k/gt.txt', '../bigann200k/dis.txt', top_k_indices, top_k_distances)

# print("Finished computing and saving top-100 nearest neighbors and distances.")
