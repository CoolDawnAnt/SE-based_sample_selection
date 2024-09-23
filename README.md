# Structure-Entropy-Based Sample Selection for Efficient and Effective Learning





# Setup
install clip



# Image Classification

## Train the model on the entire dataset

```python
python train.py --dataset cifar10 --gpuid 0 --epochs 200 --lr 0.1 --network resnet18 --batch-size 256 --task-name all-data --base-dir ./data-model/cifar10
```

 ## Calculate importance score

```python
python generate_importance_score.py --gpuid 0 --base-dir ./data-model/cifar10 --task-name all-data
```

## Structure Entropy calculation

1. Extract features of the dataset using CLIP model. We present pre-extracted features of CIFAR10/100 in `/extracted_feature`.
```python
cd Structure_Entropy
python extract_feature.py
```
2.  Build a knn graph based on the feature. We recommend $k=\log n$ as an initial setting before searching the best $k$. For CIFAR10/100, $\log n$ is approximately $15$.
```python
python build_graph.py --knn 15
```
3.  Structure entropy calculation and merge it into previous score file.
```python
python generate_SE_score.py --knn 15 
```

## Train the model with Structure-Entropy-Based Sample Selection

```python
python train.py --base-dir ./data-model/cifar10 --dataset cifar10 --gpuid 0 --epochs 200  --coreset --coreset-mode SE_bns --coreset-ratio 0.1 --mis-ratio 0.35 --knn 11 --gamma 1.1 --data-score-path ./data-model/cifar10/all-data/cifar10-data-score-all-11NN-data.pickle
```



### Acknowledgements

Thanks to the authors of [Coverage-centric Coreset Selection for High Pruning Rates](https://github.com/haizhongzheng/Coverage-centric-coreset-selection) and [D2 Pruning: Message Passing for Balancing Diversity & Difficulty in Data Pruning](https://github.com/adymaharana/d2pruning) for releasing their code for evaluating CCS/D2 and training ResNet models on CIFAR10, CIFAR100, ImageNet-1K. Much of this codebase has been adapted from their codes.

