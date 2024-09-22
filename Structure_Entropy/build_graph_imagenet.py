from scipy import sparse
from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph

from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
import faiss


def create_index(datas_embedding):
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(datas_embedding.shape[1])  
    # gpu_index_flat = index_flat
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 1, index_flat)
    gpu_index_flat.add(datas_embedding)   
    return gpu_index_flat

def data_recall(faiss_index, query_embedding, top_k):
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Distance, Index

data_dir = '../graphs/'
filename = 'imagenet-train'
feature = np.load(data_dir+filename+'.npz')['feature'].astype('float32')


feature_norm = np.linalg.norm(feature , axis = 1)
feature_norm = feature_norm.reshape(-1,1)

print(feature.shape , feature_norm.shape)
feature = feature / feature_norm

print(feature.shape)
gpu_index_flat = create_index(feature)

n = feature.shape[0]

# row = np.array([0, 0, 1, 2, 2, 2])

for knn in [21]:
    row = []

    indptr = np.array([i * knn for i in range(n + 1)])
    indices = np.zeros(n * knn , dtype = int)
    data = np.zeros(n * knn , dtype = np.float64)
    import time

    st = time.time()
    batch = 5000
    for i in range(n):
        l = i * batch
        r = min(n - 1 , (i + 1) * batch)
        if l >= n:
            break
        if i % 10 == 0:
            print(i , n // batch) 
        # print(feature[i : r].shape)
    
        distances , indexs = data_recall(gpu_index_flat , feature[l : r] , knn)
    # print(distances.shape , indexs.shape)
        distances = distances.reshape(-1)
        indexs = indexs.reshape(-1)
    
        indices[l * knn : r * knn] = indexs
        data[l * knn : r * knn] = 1 - distances

    # if l <= 137180 and 137180 < r:
    #     print(distances[knn * (137180 - l) : knn * (137180 - l + 1)])
    # print(distances,indexs)
    print('time' , time.time() - st)
    undirected_adj = csr_matrix((data, indices, indptr), shape=(n, n))
    sparse.save_npz(f'./graph_datas/imagenet-train-swav-{knn}NN_se', undirected_adj)

# csr_matrix((data, indices, indptr), shape=(3, 3))

# print(feature[585184 , :20] , feature[137180 , :20])
# print(np.dot(feature[585184] , feature[137180]) / (np.linalg.norm(feature[585184]) * np.linalg.norm(feature[137180])) )

