import copy
import math
import heapq
import random
import numba as nb
import numpy as np

import time
from collections import OrderedDict
data_dir = '../graphs/'

def get_id(reset = False):
    global static_var_cnt
    if (reset == None):
        pass
    elif (reset == True):
        static_var_cnt = -1
    else:
        static_var_cnt += 1
    return static_var_cnt

def graph_parse(adj_matrix):
    g_num_nodes = adj_matrix.shape[0]
    adj_table = [[] for i in range(3 * g_num_nodes + 4)]
    # adj_table_r = [[] for i in range(3 * g_num_nodes + 4)]
    # adj_matrix = adj_matrix + adj_matrix.T

    VOL = 0
    node_vol = [0 for i in range(g_num_nodes)]
    indices = np.argwhere(adj_matrix)

    idx = 0
    for (x , y) in indices:
        # assert(x != y)
        if (adj_matrix[y , x] == 0):
            idx += 1
            VOL += adj_matrix[x , y]
            node_vol[y] += adj_matrix[x , y]
            adj_table[y].append( (x , adj_matrix[x , y]) )
            adj_matrix[y , x] = adj_matrix[x , y]
        
        idx += 1
        VOL += adj_matrix[x , y]
        node_vol[x] += adj_matrix[x , y]
        adj_table[x].append( (y , adj_matrix[x , y]) )
        # adj_table_r[y].append( (x , adj_matrix[x , y]) )
    # print('tot edges',idx)
    # print('fafa',VOL)
    for i in range(g_num_nodes):
        if node_vol[i] == 0:
            node_vol[i] = 1e-5
    return g_num_nodes,VOL,node_vol,adj_table


def graph_parse_sparse(undirected_adj):
    g_num_nodes = undirected_adj.shape[0]
    adj_table = [[] for i in range(3 * g_num_nodes + 4)]
    map_adj_table = [{} for i in range(3 * g_num_nodes + 4)]

    VOL = 0
    node_vol = [0 for i in range(g_num_nodes)]
    
    for u in range(g_num_nodes):
        for j in range(undirected_adj.indptr[u] , undirected_adj.indptr[u+1]):
            v = undirected_adj.indices[j]
            w = undirected_adj.data[j]
            if u == v:
                continue
            map_adj_table[u][v] = w 
            map_adj_table[v][u] = w 
    
    idx = 0
    for u in range(g_num_nodes):
        for v, w in map_adj_table[u].items():
            w += 1e-5
            adj_table[u].append((v , w))
            node_vol[u] += w
            VOL += w
            idx += 1
 
    return g_num_nodes,VOL,node_vol,adj_table


@nb.jit(nopython=True)
def cut_volume(adj_matrix,p1,p2,adj_table=None,exists = None,bst1=None,bst2=None,edge_table=None):

    c12 = 0
    for u in p1:
        for v in p2:
            c12 += adj_matrix[u , v]
    return c12



def LayerFirst(node_dict,start_id):
    stack = [start_id]
    while len(stack) != 0:
        node_id = stack.pop(0)
        yield node_id
        if node_dict[node_id].children:
            for c_id in node_dict[node_id].children:
                stack.append(c_id)


def merge(new_ID, id1, id2, cut_v, node_dict):
    new_partition = node_dict[id1].partition + node_dict[id2].partition
    v = node_dict[id1].vol + node_dict[id2].vol
    g = node_dict[id1].g + node_dict[id2].g - 2 * cut_v
    child_h = max(node_dict[id1].child_h,node_dict[id2].child_h) + 1
    new_node = PartitionTreeNode(ID=new_ID,partition=new_partition,children={id1,id2},
                                 g=g, vol=v,child_h= child_h,child_cut = cut_v)
    node_dict[id1].parent = new_ID
    node_dict[id2].parent = new_ID
    node_dict[new_ID] = new_node


def compressNode(node_dict, node_id, parent_id):
    p_child_h = node_dict[parent_id].child_h
    node_children = node_dict[node_id].children
    node_dict[parent_id].child_cut += node_dict[node_id].child_cut
    node_dict[parent_id].children.remove(node_id)
    node_dict[parent_id].children = node_dict[parent_id].children.union(node_children)
    for c in node_children:
        node_dict[c].parent = parent_id
    com_node_child_h = node_dict[node_id].child_h
    node_dict.pop(node_id)

    if (p_child_h - com_node_child_h) == 1:
        while True:
            max_child_h = max([node_dict[f_c].child_h for f_c in node_dict[parent_id].children])
            if node_dict[parent_id].child_h == (max_child_h + 1):
                break
            node_dict[parent_id].child_h = max_child_h + 1
            parent_id = node_dict[parent_id].parent
            if parent_id is None:
                break



def child_tree_deepth(node_dict,nid):
    node = node_dict[nid]
    deepth = 0
    while node.parent is not None:
        node = node_dict[node.parent]
        deepth+=1
    deepth += node_dict[nid].child_h
    return deepth


def CompressDelta(node1,p_node):
    a = node1.child_cut
    v1 = node1.vol + 1
    v2 = p_node.vol + 1
    return a * math.log2(v2/v1)


def CombineDelta(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2

    # assert(cut_v <= g1)
    # assert(cut_v <= g2)
    # return ((v1 - g1) * math.log2(v12/v1) + (v2 - g2) * math.log2(v12/v2) - 2 * cut_v * math.log2(g_vol/v12)) / g_vol
    return ((g1 - cut_v) * math.log2(v12/v1) + (g2 - cut_v) * math.log2(v12/v2) + 2 * cut_v * math.log2(v12 / g_vol))
    
    return cut_v * math.log2(v12 /g_vol)

def CombineDelta2(node1, node2, cut_v, g_vol):
    v1 = node1.vol
    v2 = node2.vol
    g1 = node1.g
    g2 = node2.g
    v12 = v1 + v2

    # assert(cut_v <= g1)
    # assert(cut_v <= g2)
    return ((v1 - g1) * math.log2(v12/v1) + (v2 - g2) * math.log2(v12/v2) - 2 * cut_v * math.log2(g_vol/v12))
    return ((g1 - cut_v) * math.log2(v12/v1) + (g2 - cut_v) * math.log2(v12/v2) + 2 * cut_v * math.log2(v12 / g_vol))
    
    # return cut_v * math.log2(v12)

class PartitionTreeNode():
    def __init__(self, ID, partition, vol, g, children:set = None,parent = None,child_h = 0, child_cut = 0 , edge_bitset = None , is_parse = True):
        self.ID = ID
    
        self.partition = partition
        self.parent = parent
        self.children = children
        self.vol = vol
        self.g = g
        self.merged = False
        self.child_h = child_h
        self.child_cut = child_cut
        self.edge_bitset = edge_bitset

    def __str__(self):
        return "{" + "{}:{}".format(self.__class__.__name__, self.gatherAttrs()) + "}"

    def gatherAttrs(self):
        return ",".join("{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())
class DSU():
    def __init__(self , n , k):
        self.par = [i for i in range((k + 1) * n + 4)]
        self.n = n
        self.k = k
    
    def reset(self):
        self.par = [i for i in range((self.k + 1) * self.n + 4)]

    def findpar(self , x):
        r = x
        while(r != self.par[r]):
            r = self.par[r]

        y = x
        while(self.par[y] != r):
            z = self.par[y]    
            self.par[y] = r    
            y = z         
        return r
    
    def merge(self , x , y):
        # assert(self.par[x] == x)
        # assert(self.par[y] == y)
        
        x = self.findpar(x)
        y = self.findpar(y)
        if (x == y):
            return
        self.par[x] = y
        


class PartitionTree():

    def __init__(self,adj_matrix,is_sparse=False):
        self.adj_matrix = adj_matrix
        self.tree_node = {}
        if is_sparse:
            self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse_sparse(self.adj_matrix)
        else:
            self.g_num_nodes, self.VOL, self.node_vol, self.adj_table = graph_parse(self.adj_matrix)
        
        get_id(True)
        self.leaves = []
        self.dsu = None
        self.build_leaves()


    def __dfs(self , x):
        self.dfn.append(x)
        node_p = self.tree_node[x]
        
        
        stack = [x]
        while len(stack) != 0:
            node_id = stack.pop(0)
            node_p_vol = self.tree_node[node_id].vol
                
            if self.tree_node[node_id].children:
                for c_id in self.tree_node[node_id].children:
                    node_vol = self.tree_node[c_id].vol
                    node_g = self.tree_node[c_id].g
                    
                    self.dep[c_id] = self.dep[node_id] + 1
        
                    # if (node_vol <= 0):
                    #     print(c_id)
                    #     for (xxx,yyy) in self.adj_table[c_id]:
                    #         print(xxx,yyy)
                    # assert(node_vol  > 0)
                    self.wdep[c_id] = self.wdep[node_id] - (node_g / (self.VOL+1)) * math.log2((node_vol+1) / (node_p_vol+1))
                    self.wdep2[c_id] = self.wdep2[node_id] + node_vol/(node_p_vol+1) * math.exp(-node_g / (self.VOL+1))  
                    
                    self.wsz[c_id] =  - node_g / (self.VOL+1) * math.log2((node_vol+1)/(node_p_vol+1))
                    stack.append(c_id)
                    self.dfn.append(c_id)
            else:
                self.sz[node_id] = 1

                    
        
        for i in range(len(self.dfn) - 1 , 0 , -1):
            u = self.dfn[i]
            self.sz[self.tree_node[u].parent] += self.sz[u]
            self.wsz[self.tree_node[u].parent] += self.wsz[u]
        
        
    
    def dfs(self , k):
        self.dep = [0 for i in range((k + 1) * self.g_num_nodes + 4)] 
        self.wdep = [0 for i in range((k + 1) * self.g_num_nodes + 4)]
        self.wdep2 = [0 for i in range((k + 1) * self.g_num_nodes + 4)]
        self.sz = [0 for i in range((k + 1) * self.g_num_nodes + 4)]
        self.wsz = [0 for i in range((k + 1) * self.g_num_nodes + 4)]

        self.dfn = []
        self.__dfs(self.root_id)

    def LCA(self , u , v):
        while u != v:
            if (self.dep[u] > self.dep[v]):
                u = self.tree_node[u].parent
            else:
                v = self.tree_node[v].parent
        return u 
    def build_leaves(self):
        for vertex in range(self.g_num_nodes):
            ID = get_id()
            v = self.node_vol[vertex]
            leaf_node = PartitionTreeNode(ID=ID, partition=[vertex], g = v, vol=v )
            self.tree_node[ID] = leaf_node
            self.leaves.append(ID)


    def build_sub_leaves(self,node_list,p_vol):
        subgraph_node_dict = {}
        ori_ent = 0
        for vertex in node_list:
            ori_ent += -(self.tree_node[vertex].g / self.VOL)\
                       * math.log2((self.tree_node[vertex].vol+1)/(p_vol+1))
            sub_n = []
            vol = 0
            for vertex_n in node_list:
                c = self.adj_matrix[vertex,vertex_n]
                if c != 0:
                    vol += c
                    sub_n.append((vertex_n , c))
            sub_leaf = PartitionTreeNode(ID=vertex,partition=[vertex],g=vol,vol=vol)
            subgraph_node_dict[vertex] = sub_leaf
            self.adj_table[vertex] = sub_n

        return subgraph_node_dict,ori_ent

    def build_root_down(self):
        root_child = self.tree_node[self.root_id].children
        subgraph_node_dict = {}
        ori_en = 0
        g_vol = self.tree_node[self.root_id].vol
        for node_id in root_child:
            node = self.tree_node[node_id]
            ori_en += -(node.g / g_vol) * math.log2((node.vol+1)/g_vol)
            new_n = []
            for (nei , w) in self.adj_table[node_id]:
                if nei in root_child:
                    new_n.append((nei , w))
            self.adj_table[node_id] = new_n

            new_node = PartitionTreeNode(ID=node_id,partition=node.partition,vol=node.vol,g = node.g,children=node.children)
            subgraph_node_dict[node_id] = new_node

        return subgraph_node_dict, ori_en


    def entropy(self,node_dict = None):
        if node_dict is None:
            node_dict = self.tree_node
        ent = 0
        for node_id,node in node_dict.items():
            # if node.parent is not None:
            #     node_p = node_dict[node.parent]
            #     node_vol = node.vol #+ 1
            #     node_g = node.g
            #     node_p_vol = node_p.vol #+ 1
            #     ent += - (node_g / self.VOL) * math.log2(node_vol/node_p_vol)

                
            if node.children:
                ent += 2 * node.child_cut * math.log2(node.vol) / self.VOL
                sums = 0
                for c in node.children:
                    sums += node_dict[c].vol
                assert(sums == node.vol)
            else:
                pass
                # assert(len(node.partition) == 1)
                # ent -= node.g * math.log2(node.vol) / self.VOL
        return ent


    def __build_k_tree(self,g_vol,nodes_dict:dict,k = None,_k = 2,usestd = False):
        start_time = time.time()
        
        min_heap = []
        cmp_heap = []
        nodes_ids = list(nodes_dict.keys())
        new_id = None
        dsu = DSU(self.g_num_nodes , k = _k)
        # _adj_table = self.adj_table

        cnt_bucket = [0.0 for i in range(self.g_num_nodes * (_k + 1) + 4)]
           

        for i in nodes_ids:
            for (j , w) in self.adj_table[i]:
                if j > i:
                    n1 = nodes_dict[i]
                    n2 = nodes_dict[j]
                    cut_v = w
                    diff = CombineDelta(nodes_dict[i], nodes_dict[j], cut_v, g_vol) if not usestd else CombineDelta2(nodes_dict[i], nodes_dict[j], cut_v, g_vol)
                    # print('input',i,j,diff)
                    heapq.heappush(min_heap, (diff, i, j, cut_v))
        unmerged_count = len(nodes_ids)
        # print('------- unmerged count -----',len(nodes_ids))
        # end_time = time.time()
        # print('Merge first',end_time-start_time)
        # start_time = time.time()
    
        while unmerged_count > 1:
            # print(len(nodes_ids))
            # assert(len(nodes_ids) == 50000)
            if len(min_heap) == 0:
                break
            diff, id1, id2, cut_v = heapq.heappop(min_heap)
            if nodes_dict[id1].merged or nodes_dict[id2].merged:
                continue
            # print('merging' , id1 , id2 , diff)
            nodes_dict[id1].merged = True
            nodes_dict[id2].merged = True
            new_id = get_id()
            merge(new_id, id1, id2, cut_v, nodes_dict)  
            # if (len(nodes_dict[id1].partition) + len(nodes_dict[id2].partition) == self.g_num_nodes):
                # print('xiuxiuxiu' ,max(len(nodes_dict[id1].partition) , len(nodes_dict[id2].partition)) / (len(nodes_dict[id1].partition) + len(nodes_dict[id2].partition)) )
            # print('Merging ',len(nodes_dict[id1].partition),len(nodes_dict[id2].partition) , cut_v , diff)          
            dsu.merge(id1 , new_id)
            dsu.merge(id2 , new_id)
            

            temporate_list = self.adj_table[id1] + self.adj_table[id2] 
            # temporate_list_r = self.adj_table_r[id1] + self.adj_table_r[id2] 

            for (u , w) in temporate_list:
                uu = dsu.findpar(u)
                if (uu != u):
                    continue
                if (u != new_id):
                    cnt_bucket[u] += w

            for (u , w) in temporate_list:
                uu = dsu.findpar(u)
                if (uu != u):
                    continue
                if (cnt_bucket[uu] != 0.0):
                    cut_v = cnt_bucket[u]
                    new_diff = CombineDelta(nodes_dict[u], nodes_dict[new_id], cut_v, g_vol) if not usestd else CombineDelta2(nodes_dict[u], nodes_dict[new_id], cut_v, g_vol)
                    heapq.heappush(min_heap, (new_diff, uu, new_id, cut_v))   
                    cnt_bucket[u] = 0.0       
                    self.adj_table[new_id].append( (u , cut_v) )
                    self.adj_table[u].append( (new_id , cut_v) )

                    
            if nodes_dict[id1].child_h > 0:
                heapq.heappush(cmp_heap,[CompressDelta(nodes_dict[id1],nodes_dict[new_id]),id1,new_id])
            if nodes_dict[id2].child_h > 0:
                heapq.heappush(cmp_heap,[CompressDelta(nodes_dict[id2],nodes_dict[new_id]),id2,new_id])
            unmerged_count -= 1
            
        
        root = new_id

        if unmerged_count > 1:
            #combine solitary node
            # print('processing solitary node')
            assert len(min_heap) == 0
            unmerged_nodes = {i for i, j in nodes_dict.items() if not j.merged}
            new_child_h = max([nodes_dict[i].child_h for i in unmerged_nodes]) + 1

            new_id = get_id()
            new_node = PartitionTreeNode(ID=new_id,partition=list(nodes_ids),children=unmerged_nodes,
                                         vol=g_vol,g = 0,child_h=new_child_h)
            nodes_dict[new_id] = new_node
            # print('aka' , len(nodes_ids))

            for i in unmerged_nodes:
                nodes_dict[i].merged = True
                nodes_dict[i].parent = new_id
                if nodes_dict[i].child_h > 0:
                    heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[i], nodes_dict[new_id]), i, new_id])
            root = new_id
        

        if k is not None:
            while nodes_dict[root].child_h > k:
                diff, node_id, p_id = heapq.heappop(cmp_heap)
                if (node_id not in nodes_dict):
                    continue
                if nodes_dict[node_id].parent != p_id:
                    continue
                if CompressDelta(nodes_dict[node_id], nodes_dict[p_id]) != diff:
                    continue
                if child_tree_deepth(nodes_dict, node_id) <= k:
                    continue

                children = nodes_dict[node_id].children
                compressNode(nodes_dict, node_id, p_id)
                if nodes_dict[root].child_h == k:
                    break

                if p_id != root and child_tree_deepth(nodes_dict, p_id) > k:
                    heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[p_id], nodes_dict[nodes_dict[p_id].parent]), p_id, nodes_dict[p_id].parent])
                
                for e in children:
                    if nodes_dict[e].child_h == 0:
                        continue
                    if child_tree_deepth(nodes_dict, e) > k:
                        heapq.heappush(cmp_heap, [CompressDelta(nodes_dict[e], nodes_dict[p_id]), e, p_id])
                
        return root


    def check_balance(self,node_dict,root_id):
        root_c = copy.deepcopy(node_dict[root_id].children)
        for c in root_c:
            if node_dict[c].child_h == 0:
                self.single_up(node_dict,c)
    def k_regular(self):
        for i in self.tree_node.keys():
            node = self.tree_node[i]
            if node.parent is None:
                continue
            if (self.tree_node[node.parent].child_h != node.child_h + 1):
                self.single_up(self.tree_node , i)
            
    def single_up(self,node_dict,node_id):
        new_id = get_id()
        p_id = node_dict[node_id].parent
        grow_node = PartitionTreeNode(ID=new_id, partition=node_dict[node_id].partition, parent=p_id,
                                      children={node_id}, vol=node_dict[node_id].vol, g=node_dict[node_id].g)
        grow_node.merged = node_dict[node_id].merged
        node_dict[node_id].merged = False
        node_dict[node_id].parent = new_id
        node_dict[p_id].children.remove(node_id)
        node_dict[p_id].children.add(new_id)
        node_dict[new_id] = grow_node
        node_dict[new_id].child_h = node_dict[node_id].child_h + 1
        self.adj_table[new_id] = self.adj_table[node_id]
        for (i , w) in self.adj_table[node_id]:
            self.adj_table[i].append((new_id , w))



    def root_down_delta(self , k):
        if len(self.tree_node[self.root_id].children) < 3:
            return 0 , None , None
        subgraph_node_dict, ori_entropy = self.build_root_down()
        g_vol = self.tree_node[self.root_id].vol
        new_root = self.__build_k_tree(g_vol=g_vol,nodes_dict=subgraph_node_dict,k = 2 , _k = k)
        self.check_balance(subgraph_node_dict,new_root)

        new_entropy = self.entropy(subgraph_node_dict)
        delta = (ori_entropy - new_entropy) / len(self.tree_node[self.root_id].children)
        return delta, new_root, subgraph_node_dict

    def leaf_up_entropy(self,sub_node_dict,sub_root_id,node_id):
        ent = 0
        for sub_node_id in LayerFirst(sub_node_dict,sub_root_id):
            if sub_node_id == sub_root_id:
                sub_node_dict[sub_root_id].vol = self.tree_node[node_id].vol
                sub_node_dict[sub_root_id].g = self.tree_node[node_id].g

            elif sub_node_dict[sub_node_id].child_h == 1:
                node = sub_node_dict[sub_node_id]
                inner_vol = node.vol - node.g
                partition = node.partition
                ori_vol = sum(self.tree_node[i].vol for i in partition)
                ori_g = ori_vol - inner_vol
                node.vol = ori_vol
                node.g = ori_g
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2((node.vol+1)/(node_p.vol+1))
            else:
                node = sub_node_dict[sub_node_id]
                node.g = self.tree_node[sub_node_id].g
                node.vol = self.tree_node[sub_node_id].vol
                node_p = sub_node_dict[node.parent]
                ent += -(node.g / self.VOL) * math.log2((node.vol+1)/(node_p.vol+1))
        return ent

    def leaf_up(self , k):
        h1_id = set()
        h1_new_child_tree = {}
        id_mapping = {}
        for l in self.leaves:
            p = self.tree_node[l].parent
            h1_id.add(p)
        delta = 0
        for node_id in h1_id:
            candidate_node = self.tree_node[node_id]
            sub_nodes = candidate_node.partition
            if len(sub_nodes) == 1:
                id_mapping[node_id] = None
            if len(sub_nodes) == 2:
                id_mapping[node_id] = None
            if len(sub_nodes) >= 3:
                sub_g_vol = candidate_node.vol - candidate_node.g
                subgraph_node_dict,ori_ent = self.build_sub_leaves(sub_nodes,candidate_node.vol)
                sub_root = self.__build_k_tree(g_vol=sub_g_vol,nodes_dict=subgraph_node_dict,k = 2 , _k = k)
                self.check_balance(subgraph_node_dict,sub_root)
                new_ent = self.leaf_up_entropy(subgraph_node_dict,sub_root,node_id)
                delta += (ori_ent - new_ent)
                h1_new_child_tree[node_id] = subgraph_node_dict
                id_mapping[node_id] = sub_root
        delta = delta / self.g_num_nodes
        return delta,id_mapping,h1_new_child_tree

    def leaf_up_update(self,id_mapping,leaf_up_dict):
        for node_id,h1_root in id_mapping.items():
            if h1_root is None:
                children = copy.deepcopy(self.tree_node[node_id].children)
                for i in children:
                    self.single_up(self.tree_node,i)
            else:
                h1_dict = leaf_up_dict[node_id]
                self.tree_node[node_id].children = h1_dict[h1_root].children
                for h1_c in h1_dict[h1_root].children:
                    assert h1_c not in self.tree_node
                    h1_dict[h1_c].parent = node_id
                h1_dict.pop(h1_root)
                self.tree_node.update(h1_dict)
        self.tree_node[self.root_id].child_h += 1


    def root_down_update(self, new_id , root_down_dict):
        self.tree_node[self.root_id].children = root_down_dict[new_id].children
        for node_id in root_down_dict[new_id].children:
            assert node_id not in self.tree_node
            root_down_dict[node_id].parent = self.root_id
        root_down_dict.pop(new_id)
        self.tree_node.update(root_down_dict)
        self.tree_node[self.root_id].child_h += 1

    def build_coding_tree(self, k=2, mode='v2',usestd=False):
        if k is not None:
            self.dsu = DSU(n = self.g_num_nodes , k = k)
            self.adj_table = self.adj_table + [[] for i in range(self.g_num_nodes * (k + 1) + 4 - len(self.adj_table))]

        if k == 1:
            return
        if mode == 'v1' or k is None:
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=k , _k = 2 if k is None else k , usestd=usestd)  
            # self.k_regular()

        elif mode == 'v2':
            self.root_id = self.__build_k_tree(self.VOL, self.tree_node, k=2 , _k = k)
            self.check_balance(self.tree_node,self.root_id)


            if self.tree_node[self.root_id].child_h < 2:
                self.tree_node[self.root_id].child_h = 2
            

            flag = 0
            while self.tree_node[self.root_id].child_h < k:
                if flag == 0:
                    leaf_up_delta,id_mapping,leaf_up_dict = self.leaf_up(k = k)
                    root_down_delta, new_id , root_down_dict = self.root_down_delta(k = k)

                elif flag == 1:
                    leaf_up_delta, id_mapping, leaf_up_dict = self.leaf_up(k = k)
                elif flag == 2:
                    root_down_delta, new_id , root_down_dict = self.root_down_delta(k = k)
                else:
                    raise ValueError

                if leaf_up_delta < root_down_delta:
                    # print('root down')
                    # root down update and recompute root down delta
                    flag = 2
                    self.root_down_update(new_id,root_down_dict)

                else:
                    # leaf up update
                    # print('leave up')
                    flag = 1
                    # print(self.tree_node[self.root_id].child_h)
                    self.leaf_up_update(id_mapping,leaf_up_dict)
                    # print(self.tree_node[self.root_id].child_h)


                    # update root down leave nodes' children
                    if root_down_delta != 0:
                        for root_down_id, root_down_node in root_down_dict.items():
                            if root_down_node.child_h == 0:
                                root_down_node.children = self.tree_node[root_down_id].children
        count = 0
        for _ in LayerFirst(self.tree_node, self.root_id):
            count += 1
        # self.dfs(k if k else 2)
        assert len(self.tree_node) == count



from sklearn.neighbors import kneighbors_graph,radius_neighbors_graph



if __name__ == "__main__":
    for test_nodes_num in [200 , 300 , 400 , 500 , 600 , 700]:#[20 , 40 , 60 , 80 , 100 , 200 , 400 , 600 , 800 , 1000 , 1500 , 2000 , 2500 , 3000 , 3500 , 4000]:
            
        for k in [test_nodes_num - 1]:#[25 , 30 , 35 , 40 , 45 , 50]:
            cmean1 = 0
            cmean2 = 0
            if (k >= test_nodes_num):
                continue
            for iii in range(20):
                random.seed((iii + 10) * (iii + 10))
                deg = [0 for i in range(test_nodes_num)]
                undirected_adj = np.zeros([test_nodes_num , test_nodes_num])
                shuffle = [i for i in range(test_nodes_num)]
                # np.random.shuffle(shuffle)
                labels = []

                features = np.random.multivariate_normal([0 , 0] , [[1 , 0] , [0 , 1]] , size = test_nodes_num)
             
                graph = kneighbors_graph(features , k , mode='connectivity', metric= 'minkowski', include_self=False)
                G = graph.todense()
                for i in range(test_nodes_num):
                    for j in range(test_nodes_num):
                        if (G[i , j] == 0) and (G[j , i] == 1):
                            G[i , j] = 1

                # for i in range(0 , test_nodes_num):
                    # for j in range(max(0 , i - k + 1) , min(test_nodes_num , i + k)):
            #        for j in range(i + 1 , test_nodes_num):
                    # for jj in range(k):
                    #     j = np.random.randint(test_nodes_num)
                    #     undirected_adj[shuffle[i] , shuffle[j] ] = undirected_adj[shuffle[j] , shuffle[i]] = 1
                    #     deg[shuffle[i]] += 1
                    #     deg[shuffle[j]] += 1
                for i in range(test_nodes_num):
                    for j in range(i):
                        if (G[i , j]):
                            deg[i] += 1
                            deg[j] += 1
        
                undirected_adj = G
                deg = np.array(deg)
                m = np.sum(deg)
                leaves = np.sum(deg * np.log2(deg)) / m


                y = PartitionTree(adj_matrix=undirected_adj)
                x = y.build_coding_tree(None , 'v1')
                ours = y.entropy()
                y = PartitionTree(adj_matrix=undirected_adj)
                x = y.build_coding_tree(None , 'v1' , True)
                origins = y.entropy()

                y = None
                cmean1 += ours
                cmean2 += origins
        
            print(test_nodes_num ,  k , cmean1 / 20 , cmean2 / 20)

