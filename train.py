import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
import random

import torch.nn.functional as F
from datetime import datetime

from torchvision import models

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import CoresetSelection, IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
from core.utils import print_training_info, StdRedirect
from Structure_Entropy.SE_bns import SE_bns
import time
from copy import deepcopy
from functools import partial


def now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")[:-1]


model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N',
                    help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N',
                    help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'cinic10'])

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N',
                    help='The number of iterations for testing model')

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default=None,
                    help='The name of the training task.')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str, choices=['random', 'coreset', 'stratified', 'moderate', 'SE_bns'])

parser.add_argument('--data-score-path', type=str)
parser.add_argument('--feature-path', type=str)
parser.add_argument('--coreset-key', type=str, default='accumulated_margin_log')
parser.add_argument('--data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--class-balanced', type=int, default=0,
                    help='Set 1 to use the same class ratio as to the whole dataset.')
parser.add_argument('--coreset-ratio', type=float)

#### Double-end Pruning Setting ####
parser.add_argument('--mis-key', type=str, default='accumulated_margin_log')
parser.add_argument('--mis-data-score-descending', type=int, default=0,
                    help='Set 1 to use larger score data first.')
parser.add_argument('--mis-ratio', type=float)
parser.add_argument('--gamma', type=float, default=1)
# parser.add_argument('--score', type=str , default = None)
# parser.add_argument('--sample', type=str , default = None)

#### Reversed Sampling Setting ####
parser.add_argument('--reversed-ratio', type=float,
                    help="Ratio for the coreset, not the whole dataset.")

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')
######################### ours #############

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--knn', type=int, default=6)

parser.add_argument('--coreset-only', action='store_true', default=False)

######################### Setting for Future Use #########################
# parser.add_argument('--ckpt-name', type=str, default='model.ckpt',
#                     help='The name of the checkpoint.')
# parser.add_argument('--lr-scheduler', choices=['step', 'cosine'])
# parser.add_argument('--network', choices=model_names, default='resnet18')
# parser.add_argument('--pretrained', action='store_true')
# parser.add_argument('--augment', choices=['cifar10', 'rand'], default='cifar10')

args = parser.parse_args()
start_time = datetime.now()

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"

print(f'Dataset: {args.dataset}')
######################### Set path variable #########################
task_name = f'{args.dataset}-{args.coreset_mode}' if args.task_name is None else args.task_name
if args.coreset_mode is not None:
    task_dir = os.path.join(args.base_dir, f'{task_name}/{args.coreset_ratio}/{now()}')
    while (os.path.exists(task_dir)):
        time.sleep(5 + random.randint(0, 3))
        task_dir = os.path.join(args.base_dir, f'{task_name}/{args.coreset_ratio}/{now()}')
else:
    task_dir = os.path.join(args.base_dir, task_name)

os.makedirs(task_dir, exist_ok=True)
last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')

print(log_path)
######################### Print setting #########################
sys.stdout = StdRedirect(log_path)
print_training_info(args, all=True)
#########################


######################### seed #########################
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#########################


print(f'Last ckpt path: {last_ckpt_path}')

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.join(args.data_dir, args.dataset)
print(f'Data dir: {data_dir}')

if args.dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir)
elif args.dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir)
elif args.dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir)

######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)


def R(A):
    amin, amax = torch.min(A), torch.max(A)
    return (A - amin) / (amax - amin)


if args.coreset:
    if args.coreset_mode != 'random':
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)

    if args.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(trainset),
                                                          num=args.coreset_ratio * len(trainset))

    if args.coreset_mode == 'coreset':
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key,
                                                                   ratio=args.coreset_ratio,
                                                                   descending=(args.data_score_descending == 1),
                                                                   class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'stratified':
        mis_num = int(args.mis_ratio * total_num)
        desdes = False if args.mis_key[0] == 'a' else True
        print(desdes)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key=args.mis_key, mis_num=mis_num,
                                                                 mis_descending=False if args.mis_key[
                                                                                             0] == 'a' else True,
                                                                 coreset_key=args.coreset_key)

        coreset_num = int(args.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key,
                                                                coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]

    if args.coreset_mode == 'moderate':
        features = np.load(args.feature_path)['feature']
        coreset_index = CoresetSelection.moderate_selection(data_score, args.coreset_ratio, features)

    if args.coreset_mode == 'SE_bns':
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)

        se = data_score['SE']
        Aum = data_score['accumulated_margin']
        scores = R(-Aum) * se
        scores2 = -Aum
        sampler = partial(SE_bns, gamma=args.gamma, graph=f'{args.dataset}-train-{args.knn}NN', scores2=scores2,
                          use_gamma=True)
        coreset_index = sampler(dataset=trainset, scores=scores, ratio=args.coreset_ratio, mis_ratio=args.mis_ratio)

    trainset = torch.utils.data.Subset(trainset, coreset_index)
    print(len(trainset))
######################### Coreset Selection end #########################


if args.coreset_only:
    np.save(log_path, np.array(coreset_index))
    sys.exit()
trainset = IndexDataset(trainset)
print(len(trainset))

if args.dataset == 'cifar10':
    testset = CIFARDataset.get_cifar10_test(data_dir)
elif args.dataset == 'cifar100':
    testset = CIFARDataset.get_cifar100_test(data_dir)
elif args.dataset == 'svhn':
    testset = SVHNDataset.get_svhn_test(data_dir)
elif args.dataset == 'cinic10':
    testset = CINIC10Dataset.get_cinic10_test(data_dir)

print(len(testset))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=True, num_workers=8)

iterations_per_epoch = len(trainloader)
if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

if args.dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes = 10
else:
    num_classes = 100

if args.network == 'resnet18':
    print('resnet18')
    model = resnet('resnet18', num_classes=num_classes, device=device)
if args.network == 'resnet50':
    print('resnet50')
    model = resnet('resnet50', num_classes=num_classes, device=device)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)

if args.epochs is None:
    epoch_per_testing = args.iterations_per_testing // iterations_per_epoch
else:
    epoch_per_testing = 1

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

trainer = Trainer()
TD_logger = TrainingDynamicsLogger()

best_acc = 0
best_epoch = -1

current_epoch = 0
while num_of_iterations > 0:
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    trainer.train(current_epoch, -1, model, trainloader, optimizer, criterion, scheduler, device, TD_logger=TD_logger,
                  log_interval=60, printlog=True)

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0 or num_of_iterations == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20, printlog=True)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)

    current_epoch += 1
    # scheduler.step()

# last ckpt testing
test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20, printlog=True)
if test_acc > best_acc:
    best_acc = test_acc
    best_epoch = current_epoch
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': best_epoch
    }
    torch.save(state, best_ckpt_path)
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)
######################### Save #########################
state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)
TD_logger.save_training_dynamics(td_path, data_name=args.dataset)

print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')