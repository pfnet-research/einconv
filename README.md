# Einconv

![image](https://drive.google.com/uc?export=view&id=1A0R5ySqDnHqY9bFPgqcTok4w5AjQlfer)

Einconv is a flexible CNN module in which the structure is described as [tensor network](https://tensornetwork.org/) --- graphical notation for tensor decomposition. Einconv is basically an extension of `numpy.einsum` so that we can compute higher-order convolution and multiplication in a consistent manner. 
- Einconv can mimic many existing CNN modules shown in the above picture.
- You can design your own CNN module by editing a graph structure.
- Applicable to 3D or more higher-order convolution.

The code is based on [Chainer](https://chainer.org/). The basic concept is transferable to other libraries such as [PyTorch](https://pytorch.org/).

### Reference

Kohei Hayashi, Taiki Yamaguchi, Yohei Sugawara, Shin-ichi Maeda.
Exploring Unexplored Tensor Decompositions for Convolutional Neural Networks.
arXiv:xxxx.xxxxx (2019).

## Tested environment

- Python 3.6
- Numpy 1.13.3
- Chainer 5.1.0

## Training with specifying a tensor network

For first demo, you can try `python -u train_fmnist.py --graph 1_2_1_1_1_1_1` to learn with Fashion MNIST data set, where the option `--graph 1_2_1_1_1_1_1` specifies a tensor network that is transformed into a CNN module. 

## Graph format

The `graph` option follows the sequence of digits delimited by the under score `_`, for example `2_3_1_0_1_1_1_1_0_0_1_1_1_1_1_1`. The first digit (`2`) indicates the number of layers involving the convolution module, and the second digit (`3`) indicates the spatial dimension of a data set, e.g., `3` for 3D convolution. The remining digits show an adjacency matrix that describes the tensor network in a flatten manner with row major order. In the example, the adjacency matrix is as follow.
```
i o c c c e r
-------------
1 0 1 1 1 1 0
0 1 1 1 1 1 1
```

In the adjacency matrix, the first two columns associate with the in-channel and out-channel. The next *n* columnsh associate with the nd convolution (inner indices). The *m* columns associate with extra channels (outer indices). The last column indicates whether ReLU is on (`1`) or off (`2`). The number of columns is therefore *2 + n + m + 1*.

## Graph examples
### 2D
- Standard: `1_2_1_1_1_1_0`
- Flattend: `3_2_1_1_0_0_0_0_1_0_1_0_0_1_1_0_0`
- Depthwise separable.: `2_2_1_0_1_1_1_1_1_0_0_1`
- Bottleneck: `3_2_1_0_0_0_3_0_1_0_0_1_1_3_3_1_0_1_0_0_0_3_1`
- Factoring: `2_2_1_0_1_1_1_1_1_1_1_1`
- CP: `4_2_1_0_0_0_3_0_0_0_0_1_3_0_0_0_1_0_3_0_0_1_0_0_3_0`

### 3D
- Standard: `1_3_1_1_1_1_1_0`
- Depthwise separable: `2_3_1_0_1_1_1_0_1_1_0_0_0_0`
- (2+1)D convolution: `2_3_1_0_1_1_0_1_0_0_1_0_0_1_1_0`
- CP: `5_3_1_0_0_0_0_1_0_0_0_0_0_1_1_0_0_0_0_1_0_1_0_0_0_1_0_0_1_0_0_1_0_0_0_1_0`
- TT: `5_3_1_0_0_0_0_0_0_0_1_0_0_0_0_0_1_0_0_1_1_0_0_0_0_1_0_0_1_1_0_0_0_0_1_0_0_1_1_0_0_0_0_1_0_0_0_1_0_0_0_0`
- HT: `7_3_1_0_0_0_0_0_0_0_1_0_0_0_0_0_1_0_0_0_0_0_0_1_0_0_0_0_0_0_0_0_0_0_1_1_1_0_0_0_0_0_1_1_0_0_0_0_0_0_0_0_0_1_0_0_1_0_0_0_0_0_0_0_0_0_0_1_1_1_0_0_0_0_0_1_0_0_0_0_0_1_0_0_1_0`
