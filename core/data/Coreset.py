import torch
import numpy as np
from torch.utils.data import Subset
from collections import Counter
import heapq
from scipy import sparse
import pickle
import math
import copy
def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)

    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot


def get_distance(features, labels):

    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))

    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels == i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)

    return distance

class CoresetSelection(object):
    @staticmethod
    def score_monotonic_selection(data_score, key, ratio, descending, class_balanced):
        score = data_score[key]
        score_sorted_index = score.argsort(descending=descending)
        total_num = ratio * data_score['targets'].shape[0]

        if class_balanced:
            print('Class balance mode.')
            all_index = torch.arange(data_score['targets'].shape[0])
            #Permutation
            targets_list = data_score['targets'][score_sorted_index]
            targets_unique = torch.unique(targets_list)
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                targets_num = target_index_mask.sum()

            #Guarantee the class ratio doesn't change
            selected_index = []
            for target in targets_unique:
                target_index_mask = (targets_list == target)
                target_index = all_index[target_index_mask]
                target_coreset_num = targets_num * ratio
                selected_index = selected_index + list(target_index[:int(target_coreset_num)])
            selected_index = torch.tensor(selected_index)
            print(f'High priority {key}: {score[score_sorted_index[selected_index][:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[selected_index][-15:]]}')

            return score_sorted_index[selected_index]

        else:
            print(f'High priority {key}: {score[score_sorted_index[:15]]}')
            print(f'Low priority {key}: {score[score_sorted_index[-15:]]}')
            return score_sorted_index[:int(total_num)]

    @staticmethod
    def mislabel_mask(data_score, mis_key, mis_num, mis_descending, coreset_key):
        mis_score = data_score[mis_key]
        mis_score_sorted_index = mis_score.argsort(descending=mis_descending)
        hard_index = mis_score_sorted_index[:mis_num]
        print(f'Bad data -> High priority {mis_key}: {data_score[mis_key][hard_index][:15]}')
        print(f'Prune {hard_index.shape[0]} samples.')

        easy_index = mis_score_sorted_index[mis_num:]
        data_score[coreset_key] = data_score[coreset_key][easy_index]

        return data_score, easy_index


    @staticmethod
    def stratified_sampling(data_score, coreset_key, coreset_num):
        stratas = 50
        print('Using stratified sampling...')
        score = data_score[coreset_key]
        total_num = coreset_num

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)

        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]

            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num


            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)

            return rst

        budgets = bin_allocate(total_num, strata_num)

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(data_score[coreset_key].shape[0])

        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            rand_index = torch.randperm(pool.shape[0])
            selected_index += [idx.item() for idx in pool[rand_index][:budgets[i]]]

        return selected_index, None

    @staticmethod
    def random_selection(total_num, num):
        print('Random selection.')
        score_random_index = torch.randperm(total_num)

        return score_random_index[:int(num)]
    
    @staticmethod
    def moderate_selection(data_score, ratio, features):

        def get_prune_idx(rate, distance):
            rate = 1-rate
            low = 0.5 - rate / 2
            high = 0.5 + rate / 2

            sorted_idx = distance.argsort()
            low_idx = round(distance.shape[0] * low)
            high_idx = round(distance.shape[0] * high)

            ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))

            return ids

        targets_list = data_score['targets']
        distance = get_distance(features, targets_list)
        ids = get_prune_idx(ratio, distance)

        return ids

    
