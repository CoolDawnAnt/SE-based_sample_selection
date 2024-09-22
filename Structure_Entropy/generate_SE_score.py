from codingtree import PartitionTree
import numpy as np
import pickle
import torch
import math
import argparse
from copy import deepcopy
from scipy import sparse
import os
import torch.nn.functional as F

def R(A):
    amin , amax = torch.min(A) , torch.max(A)
    A = (A - amin) / (amax - amin)
    return A

def SE(data_importance, graph_path):
    k = None
    print(f'Building coding tree on graph {graph_path}.')

    undirected_adj = sparse.load_npz(graph_path)
    undirected_adj.data = (2 - undirected_adj.data) / 2
    n = undirected_adj.shape[0]
    SE_tree = PartitionTree(adj_matrix=undirected_adj, is_sparse=True)
    SE_tree.build_coding_tree(k, 'v1')
    print('Done building coding tree.')
    SE_tree.dfs(k if k else 3)

    scores = [0 for i in range(n)]

    for u in range(n):
        ccnt = 0
        for (v, _) in SE_tree.adj_table[u]:
            if (v >= n):
                continue
            lca = SE_tree.LCA(u, v)
            scores[u] += _ * math.log2(SE_tree.tree_node[lca].vol + 1)
    data_importance['SE'] = torch.tensor(scores).type(torch.float32)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################


parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--knn', type=int, default=11)
parser.add_argument('--data-score-dir', type=str, default=f'../data-model/cifar10/all-data')
parser.add_argument('--data-score-path', type=str, default='data-score-all-data-base.pickle')

parser.add_argument('--graph-path', type=str, default='../graphs')


args = parser.parse_args()

dataset = args.dataset

args = parser.parse_args()
data_dir = os.path.join(args.data_score_dir, args.data_score_path)
print(data_dir)
with open(data_dir, 'rb') as handle:
    data_score = pickle.load(handle)

for knn in [args.knn]:
    new_data_dir = os.path.join(args.data_score_dir, f'{dataset}-data-score-all-{knn}NN-data.pickle')
    SE(data_score, os.path.join(args.graph_path,f'{dataset}-train-{knn}NN.npz'))
    with open(new_data_dir, 'wb') as handle:
        pickle.dump(data_score, handle)

## python calc_SE_scores.py --knn 6 --data-score-dir ./anli/all-data/ --data-score-path data-score-anli.pickle --graph-path ./graphs/anli-6NN.npz