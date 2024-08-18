# Interface Laplace Learning
This repository contains the code for [Interface Laplace Learning: Learnable Interface Term Helps Semi-Supervised Learning](https://arxiv.org/abs/2408.05419)

## Dependencies
NumPy, SciPy and PyTorch. Scipy is for sparse matrix calculation. PyTorch is for acceleration of matrix multiplication on gpu.
The code is tested on the following version:
```
numpy==1.26.4
scipy==1.11.4
torch==2.2.1
```

## Files
[data/](data/) contains pretrained extracted features of each dataset. The files are collected from [GraphLearning package](https://github.com/jwcalder/GraphLearning/tree/master/kNNData) without any change, but renamed for clarity.

[inter_laplace.py](inter_laplace.py) includes the main function to perform training and inference.

[utils.py](utils.py) includes utility functions.

[preprocess.py](preprocess.py) includes the preprocess functions to get T, interface index and A.

## Usage
The optimal parameters are provided in the appendix. Should take less than 1 second for each trial on gpu.
```
python inter_laplace.py --dataset minst --label_num 1 --k_hop 4 --ridge 0.03
```


