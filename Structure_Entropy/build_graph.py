from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph
import numpy as np
from scipy import sparse
import scipy
from tqdm import trange
import math
import os

import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')
parser.add_argument('--knn', type=int, default=10)
parser.add_argument('--graph-path', type=str, default='../graphs')
parser.add_argument('--dataset', type=str, default='cifar10')
args = parser.parse_args()

k = args.knn
data_dir = args.graph_path
filename = f'{args.dataset}-train'

feature = np.load(os.path.join(data_dir,filename+'.npz'))['feature']
A = kneighbors_graph(feature, k + 1, mode='distance', metric='cosine', include_self=True)
sparse.save_npz(os.path.join(data_dir,f'{filename}-{k}NN.npz'), A)

