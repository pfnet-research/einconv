import chainer
import chainer.functions as F
from chainer import initializers
from chainer.links import BatchNormalization

# for debugging
import logging
# logging.basicConfig(level=logging.DEBUG)
from functools import reduce
from math import sqrt

from graph_manager import GManager


def flatten(x):
    return [z for y in x for z in (
        flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str)
        else (y,))]


def prod(x):
    return 1 if len(x) == 0 else reduce((lambda a, b: a * b), x)


class Einconv(chainer.Chain):
    def __init__(self, graph_str, shapes, batchnorm=False):
        super(Einconv, self).__init__()
        
        self.add_link('gm', GManager(graph_str, shapes))
        logging.debug(self.gm.dims)
        logging.debug(self.gm.graph)

        # define parameters
        bn_axis = [self.gm.BATCH_IND] + \
                  self.gm.get_filter_indices().tolist() + \
                  self.gm.get_image_indices().tolist()
        for i in range(1, self.gm.num_tensors + 1):
            dims = self.gm.get_dims(i, expanded=True)
            henormal_scale = sqrt(2 / self.gm.get_fan_in(i))
            W_initializer = initializers.Normal(scale=henormal_scale)
            self.add_param(self.param_name(i), shape=dims, initializer=W_initializer)
            if batchnorm:
                self.add_link(self.bn_name(i), BatchNormalization(axis=tuple(bn_axis)))

        self.batchnorm = batchnorm

    @staticmethod
    def param_name(tid):
        return 'V%d' % tid

    def get_param(self, tid):
        return getattr(self, self.param_name(tid))

    @staticmethod
    def bn_name(tid):
        return 'BN%d' % tid

    def get_bn(self, tid):
        return getattr(self, self.bn_name(tid))

    def set_param(self, tid, val):
        return setattr(self, self.param_name(tid), val)

    def forward(self, x):
        batchsize = x.shape[0]
        
        Z = F.reshape(x, self.gm.get_dims(tensor_id=0, expanded=True).tolist())
        image_flags = self.gm.indices2flags(self.gm.get_image_indices())
        filter_flags = self.gm.indices2flags(self.gm.get_filter_indices())
        for tensor_id in range(1, self.gm.num_tensors):
            logging.debug('Next processing:')
            logging.debug(tensor_id)

            sum_flags = self.gm.indices2flags(self.gm.get_sum_indices(tensor_id))
            Z = expanded_einconv(Z, self.get_param(tensor_id), sum_flags, image_flags, filter_flags, self.xp)

            if self.gm.is_relu(tensor_id) and tensor_id < (self.gm.num_tensors - 1):
                Z = F.relu(Z)

            if self.batchnorm:
                Z = self.get_bn(tensor_id)(Z)

        Z = F.squeeze(Z)
        for i, d in enumerate(self.gm.dims[image_flags].tolist()):
            if d == 1:
                Z = F.expand_dims(Z, i + 2)

        if batchsize == 1:
            Z = F.expand_dims(Z, 0)
        return Z


def expanded_einconv(X, W, sum_flags, image_flags, filter_flags, xp):
    def sub(x, y):
        return x ^ (x * y)

    spatial_dim = sum(image_flags)
    X_dims = xp.array(X.shape)
    W_dims = xp.array(W.shape)

    X_flags = X_dims > 1
    W_flags = W_dims > 1

    shared_flags = X_flags * W_flags
    parallel_flags = sub(shared_flags, sum_flags)
    other_X_flags = sub(X_flags, shared_flags + image_flags)  # N, O
    other_W_flags = sub(W_flags, shared_flags + filter_flags)  # K

    logging.debug(X.shape)
    image_flagss = decompose_flags(image_flags)
    X = moveaxis_n_reshape(X, [other_X_flags, parallel_flags, sum_flags] + image_flagss)
    X = merge(X, axis=[1, 2])
    # now X should be Ox x [P*C] x H x W

    logging.debug(W.shape)
    filter_flagss = decompose_flags(filter_flags)
    W = moveaxis_n_reshape(W, [parallel_flags, other_W_flags, sum_flags] + filter_flagss)
    W = merge(W, axis=[0, 1])
    # now W should be [P*Ow] x C x h x w

    logging.debug(' X.shape and inds:')
    logging.debug(X.shape)
    logging.debug(' W.shape and inds:')
    logging.debug(W.shape)
    logging.debug(' sum_flags:')
    logging.debug(sum_flags)

    filter_size = W_dims[filter_flags]
    image_size = X_dims[image_flags]
    pad_size = (filter_size // 2).tolist()
    groups = int(X_dims[parallel_flags].prod())  # if sum(parallel_flags) > 0 else 1
    if spatial_dim == 2:
        Z = F.convolution_2d(X, W, pad=pad_size, groups=groups)
    else:
        Z = F.convolution_nd(X, W, pad=pad_size, groups=groups)

    if Z.shape[1] == 1:  ### trick to avoid strange bug in cudnn or chainer
        Z = Z[:, :1]
    logging.debug(' Z.shape and inds after conv:')
    logging.debug(Z.shape)

    # now Z should be Ox x [P*Ow] x H x W

    dims_Ox = X_dims[other_X_flags].tolist()
    dims_P = X_dims[parallel_flags].tolist()
    dims_Ow = W_dims[other_W_flags].tolist()
    Z = split(Z, axis=1, dim=[prod(dims_P), prod(dims_Ow)])

    flagss = [other_X_flags, parallel_flags, other_W_flags] + image_flagss
    dims = dims_Ox + dims_P + dims_Ow + image_size.tolist()
    Z = epahser_n_sixaevom(Z, flagss, dims)

    return Z


def merged_dim(shape, flag):
    return prod([shape[i] if flag[i] else 1 for i in range(len(flag))])


def moveaxis_n_reshape(T, flags):
    permutation = flatten([[flag.nonzero()[0].tolist()] for flag in flags])
    T_T = F.moveaxis(T, permutation, range(len(permutation)))
    new_shape = [merged_dim(T.shape, flag) for flag in flags]
    return F.reshape(T_T, new_shape)


def epahser_n_sixaevom(_T_T, flags, dims):
    permutation = flatten([[flag.nonzero()[0].tolist()] for flag in flags])
    T_T = F.reshape(_T_T, dims + [1] * (max(permutation) - len(dims) + 1))
    return F.moveaxis(T_T, range(len(permutation)), permutation)


def decompose_flags(flags):
    flagss = []
    for i in flags.nonzero()[0]:
        f = flags.copy() * False
        f[i] = True
        flagss.append(f)
    return flagss


def merge(T, axis):
    """merge contiguous axis"""
    shape = list(T.shape)
    return F.reshape(T, shape[:min(axis)] + [-1] + shape[max(axis) + 1:])


def split(T, axis, dim):
    """inversion of merge"""
    shape = list(T.shape)
    return F.reshape(T, shape[:axis] + dim + shape[(axis + 1):])

