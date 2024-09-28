import numpy as np

from scipy import sparse
from .codingtree import PartitionTree
import copy
import os
import pickle


def SE_bns(dataset, ratio, mis_ratio, gamma=1, remain_version=1, mis_r=0, scores=None, scores2=None, use_gamma=False,
           label_path=None, graph='imagenet-train-6NN_se', dfn_version=0, cut_balance=0, reverse=True):
    n = len(dataset)
    m = int(n * ratio)
    labels = []
    if use_gamma:
        if label_path is None:
            for i in range(n):
                labels.append(dataset[i][1])
        else:
            with open(os.path.join('./graphs', label_path), 'rb') as f:
                labels = pickle.load(f)
    else:
        labels = [0 for i in range(n)]
    CLS = np.max(labels) + 1
    per_class = [0 for i in range(CLS)]
    per_class_remain = [0 for i in range(CLS)]
    per_class_chosen = [0 for i in range(CLS)]

    for i in labels:
        per_class[i] += 1
    for i in range(CLS):
        per_class_remain[i] = int(per_class[i] * ratio)
        per_class_chosen[i] = int(gamma * per_class[i] * ratio)

    if np.sum(per_class_chosen) < m:
        remaining = m - np.sum(per_class_chosen)
        for i in range(remaining):
            per_class_chosen[i] += 1
    if remain_version == 1:
        label_idx = np.argsort(per_class)
        print(label_idx.shape)
        mm = int(n * ratio * gamma)
        for i in range(CLS):
            need = min(per_class[label_idx[i]], mm // (CLS - i))
            per_class_chosen[label_idx[i]] = need
            mm -= need

    print(
        f'Using SE_bns on Graph {graph} \n with cut_balance = {cut_balance} mis = {mis_ratio} dfn = {dfn_version} gamma = {gamma} bin = {remain_version}')

    data_dir = './graphs/'
    undirected_adj = sparse.load_npz(os.path.join(data_dir, f'{graph}.npz'))
    undirected_adj.data = (2 - undirected_adj.data) / 2
    SE_tree = PartitionTree(adj_matrix=undirected_adj, is_sparse=True)

    def check(mid, dfn, _visited):
        visited = list(_visited)
        idx = []
        sums = [0 for i in range(CLS)]
        for i in range(n):
            u = dfn[i]
            if len(idx) >= m:
                return idx
            label_u = labels[u]
            if (visited[u] == 0) and (sums[label_u] < per_class_chosen[label_u]):
                idx.append(u)
                sums[label_u] += 1
                visited[u] = 1
                for (v, _) in SE_tree.adj_table[u]:
                    if (v < n) and (_ > mid):
                        visited[v] = 1
        return idx

    if reverse:
        scores = -scores
        if scores2 is not None:
            scores2 = -scores2

    visited = [0 for i in range(n)]
    if scores2 is not None:
        dfn = np.argsort(scores2)
    else:
        dfn = np.argsort(scores)

    if cut_balance:
        per_class_ban = [0 for i in range(CLS)]
        for i in range(CLS):
            per_class_ban[i] = int(per_class[i] * mis_ratio)
        M = int(n * mis_ratio)
        for i in dfn:
            if per_class_ban[labels[i]] and M:
                visited[i] = 1
                M -= 1
                per_class_ban[labels[i]] -= 1
    else:
        if mis_ratio < 0:
            for i in range(int(n * -mis_ratio)):
                visited[dfn[-i]] = 1
        else:
            for i in range(int(n * mis_ratio)):
                visited[dfn[i]] = 1

        for i in range(int(n * mis_r)):
            visited[dfn[-i]] = 1

    pre_visited = copy.copy(visited)
    if dfn_version == 0:
        dfn = np.argsort(scores)
    else:
        eps = 1e-4
        dfn_order = scores.numpy()
        if reverse:
            dfn_order = -dfn_order
        dfn_order = dfn_order + eps
        dfn_order = dfn_order / np.sum(dfn_order)

        dfn = np.random.choice(n, n, p=dfn_order, replace=False)

    L, R = np.min(undirected_adj.data), np.max(undirected_adj.data)
    ans = R
    eps = 1e-9
    sums = [0 for i in range(CLS)]
    for i in range(n):
        if not visited[i]:
            sums[labels[i]] += 1
    while abs(R - L) > eps:
        mid = (L + R) / 2
        bianary_ans = len(check(mid, dfn, visited))
        if bianary_ans >= m:
            R = mid
            ans = mid
        else:
            L = mid
    print(f'Threshold = {ans}')
    idx = check(ans, dfn, visited)
    sums = [0 for i in range(CLS)]
    for i in idx:
        sums[labels[i]] += 1
    if (len(idx) < m):
        print(
            f'Not enough samples! Needs {m} samples, only select {len(idx)} samples (This means imbalance factor gamma needs bigger)')
        rkx = torch.randperm(n)
        visited = pre_visited
        chosen = [0 for i in range(n)]
        for i in idx:
            chosen[i] = 1
        for i in rkx:
            if (sums[labels[i]] < per_class_chosen[labels[i]]) and (not chosen[i]) and (len(idx) < m):
                idx.append(i)
                sums[labels[i]] += 1
    print(f'Successfully selected {len(idx)} samples')
    return idx
