import os
import numpy as np
import torch
import argparse
import scipy.sparse as sparse
from utils import sparse_mx_to_torch_sparse_tensor, weight_matrix, create_labeled_index
from preprocess import get_T, get_interface_idx, get_A
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['mnist', 'fashionmnist', 'cifar', 'new_mnist'])
parser.add_argument("--trials", type=int, help='number of test trials', default=100)
parser.add_argument("--label_num", type=int, default=1)
parser.add_argument('--device', type=str, default='0')

parser.add_argument("--ridge", help='lambda weighting factor before regularizer', type=float)
parser.add_argument("--k_hop", help='remove how many k hops', type=int, default=0)
parser.add_argument('--no_subtract_mean', default=False, action='store_true',
                    help='whether to enforce zero mean on each column of u, default is to remove')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # specify which GPU(s) to be used

subtract_mean = not args.no_subtract_mean


def train(A, train_labels):
    m = len(train_labels)
    train_labels = torch.LongTensor(train_labels).cuda()

    I = torch.eye(m, device='cuda')
    Y = F.one_hot(train_labels).float()

    f = A.T @ torch.linalg.solve(A @ A.T + m * args.ridge * I, Y)
    return f


def test(W, f_whole, T):
    D_inv = sparse.spdiags(W.sum(axis=1).A1 ** -1, 0, W.shape)
    DW = D_inv * W
    D_inv = sparse_mx_to_torch_sparse_tensor(D_inv, cuda=True)
    DW = sparse_mx_to_torch_sparse_tensor(DW, cuda=True)
    Df = torch.sparse.mm(D_inv, f_whole)

    u = torch.zeros([len(f_whole), 10], device='cuda')
    for _ in range(T):
        u = Df + torch.sparse.mm(DW, u)
        if subtract_mean:
            u = u - torch.mean(u, dim=0, keepdim=True)

    u = u.detach().cpu().numpy()

    return u


def main():
    if args.dataset == 'mnist':
        M = np.load("data/MNIST_vae_knn.npz", allow_pickle=True)
        J = M['J']
        D = M['D']

        M = np.load("data/MNIST_labels.npz", allow_pickle=True)
        labels = M['labels']
    elif args.dataset == 'fashionmnist':
        M = np.load("data/FashionMNIST_vae_knn.npz", allow_pickle=True)
        J = M['J']
        D = M['D']

        M = np.load("data/FashionMNIST_labels.npz", allow_pickle=True)
        labels = M['labels']
    elif args.dataset == 'cifar':
        M = np.load("data/cifar_aet_knn.npz", allow_pickle=True)
        J = M['J']
        D = M['D']

        M = np.load("data/cifar_labels.npz", allow_pickle=True)
        labels = M['labels']
    elif args.dataset == 'new_mnist':
        M = np.load("data/new_MNIST_vae_knn.npz", allow_pickle=True)
        J = M['J']
        D = M['D']

        M = np.load("data/new_MNIST_labels.npz", allow_pickle=True)
        labels = M['labels']
    else:
        raise NotImplementedError

    all_idx = np.arange(len(labels))
    W = weight_matrix(J, D, 10)

    np.random.seed(0)
    acc_list = []
    for t in range(args.trials):
        train_idx = create_labeled_index(labels, args.label_num)  # Randomly choose training data points
        train_labels = labels[train_idx]

        print('Perm={:02d}'.format(t), end='\t\t')

        # Preprocess
        T = get_T(W, train_idx)
        interface_idx = get_interface_idx(train_idx, all_idx, W, args.k_hop)
        A = get_A(W, train_idx, interface_idx, T, subtract_mean=subtract_mean)

        # Training
        f = train(A, train_labels)
        f_whole = torch.zeros([len(labels), 10], device='cuda')
        f_whole[interface_idx] = f

        # Inference
        u = test(W, f_whole, T)
        test_idx = np.setdiff1d(all_idx, train_idx)
        pred = np.argmax(u, axis=1)
        acc = (pred[test_idx] == labels[test_idx]).mean() * 100

        print('Acc: %.2f%%' % acc)
        acc_list.append(acc)

    acc_list = np.array(acc_list)
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    print('Final Result, Mean={:.2f}%, Std={:.2f}%'.format(acc_mean, acc_std))


if __name__ == '__main__':
    main()
