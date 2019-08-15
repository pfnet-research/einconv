# Einconv

![image](https://drive.google.com/uc?export=view&id=1A0R5ySqDnHqY9bFPgqcTok4w5AjQlfer)

*Einconv* is a powerful CNN module in which the structure is described in the language of [tensor network](https://tensornetwork.org/). Einconv is basically an extension of Numpy's [einsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) --- we can uniformly compute higher-order convolution and multiplication. In short, Einconv has the following appealing features. 
- Einconv can mimic a lot of CNN modules shown in the above picture.
- You can design your own CNN module by modifying a graph structure.
- Applicable to 3D or more higher-order convolution.

The code is based on [Chainer](https://chainer.org/). The basic concept is transferable to other libraries such as [PyTorch](https://pytorch.org/).

### Reference

Kohei Hayashi, Taiki Yamaguchi, Yohei Sugawara, Shin-ichi Maeda.
[Exploring Unexplored Tensor Decompositions for Convolutional Neural Networks.](https://arxiv.org/abs/1908.04471)
arXiv:1908.04471 (2019).

```
@misc{1908.04471,
Author = {Kohei Hayashi and Taiki Yamaguchi and Yohei Sugawara and Shin-ichi Maeda},
Title = {Einconv: Exploring Unexplored Tensor Decompositions for Convolutional Neural Networks},
Year = {2019},
Eprint = {arXiv:1908.04471},
}
```

## Tested environment

- Python 3.6
- Numpy 1.13.3
- Chainer 5.1.0

## A simple demo

You can try 
```
python -u train_fmnist.py --graph 1_2_1_1_1_1_1
``` 
to learn with Fashion MNIST data set, where the option `--graph 1_2_1_1_1_1_1` specifies a tensor network that is transformed into a CNN module. 

## Graph format

The `graph` option follows the sequence of digits delimited by the underscore, for example `2_3_1_0_1_1_1_1_0_0_1_1_1_1_1_1`. The first digit (`2`) indicates the number of layers involving the convolution module. The second digit (`3`) indicates the spatial dimension of a data set, e.g., `3` for 3D convolution. The remaining digits show an adjacency matrix that describes the tensor network in row-major order. The adjacency matrix of the example is as follow.
```
i o c c c e r
-------------
1 0 1 1 1 1 0
0 1 1 1 1 1 1
```

In the adjacency matrix, the first two columns denoted by `i` and `o` respectively represent the in-channel and out-channel. The next *n* columns `c` represent the *n*d convolution. The next *m* columns `e` represent extra channels (inner indices). The last column `r` indicates whether ReLU is on (`1`) or off (`0`). The number of columns is therefore *2 + n + m + 1*.

## Graph examples
### 2D
- Standard: `1_2_1_1_1_1_0`
- Flattend: `3_2_1_1_0_0_0_0_1_0_1_0_0_1_1_0_0`
- Depthwise separable: `2_2_1_0_1_1_1_1_1_0_0_1`
- Bottleneck: `3_2_1_0_0_0_3_0_1_0_0_1_1_3_3_1_0_1_0_0_0_3_1`
- Factoring: `2_2_1_0_1_1_1_1_1_1_1_1`
- CP: `4_2_1_0_0_0_3_0_0_0_0_1_3_0_0_0_1_0_3_0_0_1_0_0_3_0`

### 3D
- Standard: `1_3_1_1_1_1_1_0`
- Depthwise separable: `2_3_1_0_1_1_1_0_1_1_0_0_0_0`
- (2+1)D convolution: `2_3_1_0_1_1_0_1_0_0_1_0_0_1_1_0`
- CP: `5_3_1_0_0_0_0_1_0_0_0_0_0_1_1_0_0_0_0_1_0_1_0_0_0_1_0_0_1_0_0_1_0_0_0_1_0`
- Tensor Train: `5_3_1_0_0_0_0_0_0_0_1_0_0_0_0_0_1_0_0_1_1_0_0_0_0_1_0_0_1_1_0_0_0_0_1_0_0_1_1_0_0_0_0_1_0_0_0_1_0_0_0_0`
- Hierarchical Tucker: `7_3_1_0_0_0_0_0_0_0_1_0_0_0_0_0_1_0_0_0_0_0_0_1_0_0_0_0_0_0_0_0_0_0_1_1_1_0_0_0_0_0_1_1_0_0_0_0_0_0_0_0_0_1_0_0_1_0_0_0_0_0_0_0_0_0_0_1_1_1_0_0_0_0_0_1_0_0_0_0_0_1_0_0_1_0`

## Enumeration of graphs

```
python enumerate_graph.py n m
```
We can enumerate the possible set of graphs by `enumerate_graph.py`. It takes 2 arguments:
- `n` -- The dimension of convolution.
- `m` -- The number of extra channels.
