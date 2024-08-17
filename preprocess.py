import numpy as np
import scipy.sparse as sparse
import torch
from utils import sparse_mx_to_torch_sparse_tensor


def get_T(W, train_idx):
    """
    Calculate the number of iterations. The stopping criterion is max(|p_t-p_inf|) <= 1/n.

    return T (integer)
    """
    D_inv = sparse.spdiags(W.sum(axis=1).A1 ** -1, 0, W.shape)
    W = sparse_mx_to_torch_sparse_tensor(W, cuda=True)
    D_inv = sparse_mx_to_torch_sparse_tensor(D_inv, cuda=True)
    RW = torch.sparse.mm(W, D_inv)

    p = torch.zeros(W.shape[0], 1, device='cuda')
    p[train_idx] = 1 / len(train_idx)

    p_inf = W.sum(dim=1, keepdim=True) / W.sum()

    T = 0
    while (T < 100 or torch.max(torch.abs(p - p_inf)) > 1 / W.shape[0]) and (T < 1000):
        p = torch.sparse.mm(RW, p)
        T = T + 1
    return T


def get_interface_idx(train_idx, all_idx, W, k):
    """
    Remove k hop index from all index and return the remaining indices.

    return interface_idx, type = numpy array
    """
    if k == -1:
        return all_idx
    else:
        khop_idx = train_idx
        for _ in range(k):
            neighbor_idx = W[khop_idx].nonzero()[1]
            khop_idx = np.append(khop_idx, neighbor_idx)
            khop_idx = np.unique(khop_idx)
        interface_idx = np.setdiff1d(all_idx, khop_idx)
    return interface_idx


def get_A(W, train_idx, interface_idx, T, subtract_mean=True):
    """
    Calculate iteration matrix A.
    If subtract_mean = False, A = sum_{i=0}^{T-1} (D^-1W)^i D^-1
    If subtract_mean = True, A = sum_{i=0}^{T-1} J (D^-1WJ)^i D^-1, where J = I - 11^T/n is projection matrix
    For any matrix M, J@M extracts column mean from M, M@J extracts row mean from M

    return A, shape = [m, I], type = torch tensor
    m = len(train_idx), I = len(interface_idx)
    """
    D_inv = sparse.spdiags(W.sum(axis=1).A1 ** -1, 0, W.shape)
    DW = D_inv * W
    DW_n = DW[train_idx].todense()
    # J(D^-1WJ), notice that extracting row sum is not necessary because DW all rows sum 1
    JDW_n = DW[train_idx].todense() - DW.mean(axis=0)

    # Move to GPU for faster computation
    D_inv = sparse_mx_to_torch_sparse_tensor(D_inv, cuda=True)
    DW = sparse_mx_to_torch_sparse_tensor(DW, cuda=True)
    DW_n = torch.from_numpy(DW_n).cuda()
    JDW_n = torch.from_numpy(JDW_n).cuda()

    if subtract_mean:
        # J = I - 11^T/n
        J = torch.ones([len(train_idx), W.shape[1]], device='cuda') * -1 / W.shape[0]
        for i, idx in enumerate(train_idx):
            J[i, idx] += 1

        A = J
        for _ in range(T - 1):
            A = A + JDW_n
            JDW_n = torch.sparse.mm(JDW_n, DW)
            JDW_n = JDW_n - torch.mean(JDW_n, dim=1, keepdim=True)
    else:
        A = torch.zeros([len(train_idx), W.shape[1]], device='cuda')
        for i, idx in enumerate(train_idx):
            A[i, idx] = 1

        for _ in range(T - 1):
            A = A + DW_n
            DW_n = torch.sparse.mm(DW_n, DW)

    A = torch.sparse.mm(A, D_inv)
    A = A[:, interface_idx]
    return A
