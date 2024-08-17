import numpy as np
import scipy.sparse as sparse
import torch


def weight_matrix(J, D, k, symmetrize=True):
    # Self is counted in knn data, so add one
    k += 1

    # Restrict I,J,D to k neighbors
    n = J.shape[0]
    k = np.minimum(J.shape[1], k)

    J = J[:, :k]
    D = D[:, :k]

    D = D * D
    eps = D[:, k - 1] / 4
    D = np.exp(-D / eps[:, None])

    I = np.ones((n, k)) * np.arange(n)[:, None]
    # Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    # Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I, J)), shape=(n, n)).tocsr()

    if symmetrize:
        W = (W + W.transpose()) / 2

    W.setdiag(0)  # Placing zeros on the diagonal, as in Poisson learning
    W = W.astype(np.float32)
    return W


def sparse_mx_to_torch_sparse_tensor(sparse_mx, cuda=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    if cuda:
        return torch.sparse_coo_tensor(indices, values, shape, device='cuda')
    else:
        return torch.sparse_coo_tensor(indices, values, shape)


def create_labeled_index(labels, num_train_per_class):
    unique_labels = np.unique(labels)

    J = np.arange(len(labels))

    L = list()
    for l in unique_labels:
        L = L + np.random.choice(J[labels == l], size=num_train_per_class, replace=False).tolist()
    L = np.array(L)

    return L

