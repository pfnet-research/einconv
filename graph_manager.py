import chainer
import numpy as np


def encode_graph(graph, spatial_dim):
    values = [graph.shape[0], spatial_dim] + graph.reshape(-1).tolist()
    return "_".join(map(str, values))


def decode_graph(string):
    vec = [int(i) for i in string.split('_')]
    num_rows, spatial_dim = vec[:2]
    return np.array(vec[2:], dtype=int).reshape([num_rows, -1]), spatial_dim


def decode_graph_and_relu(encoded_graph):
    gr, spatial_dim = decode_graph(encoded_graph)
    return gr[:, :-1], gr[:, -1], spatial_dim


class GManager(chainer.Link):
    INIT_INCH_IND = 0
    INIT_OUTCH_IND = 1

    BATCH_IND = 0

    M_OPEN = 1
    M_PARALLEL = 2
    M_CLOSE = 3
    M_CONV = 4

    def __init__(self, graph_str, shapes):
        super(GManager, self).__init__()
        
        graph, relu_flags, conv_dim = decode_graph_and_relu(graph_str)
        assert self.are_spatial_dims_consistent(conv_dim, shapes),\
            "shapes['image'] and shapes['filter'] should be the same dim as the convolution dim %d" % conv_dim
        self.NUM_VARS = 2 + conv_dim
        self.INIT_CONV_INDS = [i + 2 for i in range(conv_dim)]
        self.INIT_EXCH_INDS = [i + self.NUM_VARS for i in range(graph.shape[1] - self.NUM_VARS)]
        self.INIT_CH_INDS = [self.INIT_INCH_IND, self.INIT_OUTCH_IND] + self.INIT_EXCH_INDS

        self.CONV_INDS = [i + 1 + len(self.INIT_CH_INDS) for i in range(conv_dim)]
        self.HW_INDS = [i + 1 + len(self.INIT_CH_INDS) + conv_dim for i in range(conv_dim)]

        self.CONV_DIM = conv_dim

        self.graph, self.dims, self.relu_flags = self.initialize_graph(graph, shapes, relu_flags)
        self.parsed_graph = self.parse_graph(self.graph)

        self.register_persistent('graph')
        self.register_persistent('dims')
        self.register_persistent('parsed_graph')

        self.num_tensors = self.graph.shape[0] - 1  # number of tensor variables including the input one

    def are_spatial_dims_consistent(self, conv_dim, shapes):
        return conv_dim == len(shapes['image']) == len(shapes['filter'])
    

    def parse_graph(self, graph):
        active_first = self.xp.array([self.xp.min(self.xp.nonzero(graph[:, i]))
                                      for i in range(graph.shape[1])])
        active_last = self.xp.array([self.xp.max(self.xp.nonzero(graph[:, i]))
                                     for i in range(graph.shape[1])])
        parsed_graph = self.xp.zeros(graph.shape, dtype=int)
        for i in range(graph.shape[1]):
            if i in self.HW_INDS:
                continue
            elif i in self.CONV_INDS:
                parsed_graph[graph[:, i] >= 1, i] = self.M_CONV
            else:
                for j in range(graph.shape[0]):
                    if graph[j, i] >= 1:
                        if j == active_first[i]:
                            parsed_graph[j, i] = self.M_OPEN
                        elif j == active_last[i]:
                            parsed_graph[j, i] = self.M_CLOSE
                        else:
                            parsed_graph[j, i] = self.M_PARALLEL
        return parsed_graph

    def get_sum_indices(self, tensor_id):
        return self.xp.where(self.parsed_graph[tensor_id, ] == self.M_CLOSE)[0].tolist()

    def argsort_channels(self, subgraph):
        active_last = self.xp.array([self.xp.max(self.xp.nonzero(subgraph[:, i]))
                                     for i in range(subgraph.shape[1])])
        return self.xp.argsort(-active_last)

    def initialize_graph(self, _graph, shapes, relu_flags=None):
        if relu_flags is None:
            relu_flags = [1] * _graph.shape[0]
        elif hasattr(relu_flags, 'tolist'):
            relu_flags = relu_flags.tolist()
        relu_flags = [0] + relu_flags + [0]

        num_exchannels = len(self.INIT_EXCH_INDS)
        dimgraph = self.join_graph_and_dims(_graph, shapes)
        dimgraph = self.xp.vstack((dimgraph[0, :],
                                   [1, 0] + [0] * self.CONV_DIM + [0] * num_exchannels,
                                   dimgraph[1:, :],
                                   [0, 1] + [0] * self.CONV_DIM + [0] * num_exchannels))

        graph_ch = dimgraph[1:, :][:, self.INIT_CH_INDS]
        order = self.argsort_channels(graph_ch)
        dimgraph_ch = dimgraph[:, [self.INIT_CH_INDS[i] for i in order]]
        dimgraph_filter = dimgraph[:, self.INIT_CONV_INDS]
        dimgraph_batch = self.xp.array([shapes['batch'], 1] + [0] * (dimgraph.shape[0] - 2)).reshape((-1, 1))
        dimgraph_image = self.xp.array([[shapes['image'][i], 1] + [0] * (dimgraph.shape[0] - 2)
                                        for i in range(self.CONV_DIM)]).T

        joint_dimgraph = self.xp.hstack([dimgraph_batch, dimgraph_ch, dimgraph_filter, dimgraph_image])
        dims = joint_dimgraph[0, :]
        graph = joint_dimgraph[1:, :]

        return graph, dims, relu_flags

    def join_graph_and_dims(self, graph, shapes):
        #if len(shapes['filter']) == 1:
        #    shapes['filter'] = shapes['filter'] * self.CONV_DIM
        #assert len(shapes['filter']) == self.CONV_DIM,\
        #    "shapes['filter'] must be scalar or the same dim as convolution dim (%d != %d)" \
        #    % (len(shapes['filter']), self.CONV_DIM)
        dims = self.xp.ones(graph.shape[1], dtype=int)
        dims[self.INIT_INCH_IND] = shapes['inch']
        dims[self.INIT_OUTCH_IND] = shapes['outch']
        dims[self.INIT_CONV_INDS] = shapes['filter']
        dims[self.INIT_EXCH_INDS] = 2 ** (shapes['inner_exp'] + self.xp.max(graph[:, self.INIT_EXCH_INDS], axis=0))
        binarized_graph = self.xp.where(graph >= 1, 1, 0)
        return self.xp.vstack([dims, binarized_graph])

    def get_indices(self, tensor_id):
        """return the list of numeric indices that a tensor specified by tensor_id has"""
        return self.xp.where(self.graph[tensor_id, ] >= 1)[0].tolist()

    def get_dims(self, tensor_id, expanded=False):
        inds = self.get_indices(tensor_id)
        if expanded:
            return self.xp.array([self.dims[i] if i in inds else 1 for i in range(self.graph.shape[1])], dtype=int)
        else:
            return self.dims[inds]

    def get_fan_in(self, tensor_id):
        """return fan_in for initializer"""
        dims = self.get_dims(tensor_id, expanded=True)
        receptive_field_size = self.xp.prod(dims[self.get_filter_indices()])
        incomming_neurons = self.xp.prod(dims[self.get_sum_indices(tensor_id)])
        return incomming_neurons * receptive_field_size

    def get_image_indices(self):
        return self.HW_INDS

    def get_filter_indices(self):
        return self.CONV_INDS

    def indices2flags(self, inds):
        return self.xp.array([i in inds for i in range(self.graph.shape[1])], dtype=bool)

    def is_relu(self, tensor_id):
        return self.relu_flags[tensor_id]

    def get_spatial_dim(self):
        return self.CONV_DIM

    def get_intermediate_dims(self):
        ch = 1
        channels = []
        for i in range(self.parsed_graph.shape[0]):
            for j in self.parsed_graph[i, ].nonzero()[0]:
                if self.parsed_graph[i, j] == self.M_OPEN:
                    ch *= self.dims[j]
                elif self.parsed_graph[i, j] == self.M_CLOSE:
                    ch /= self.dims[j]
            channels.append(ch)
        return channels

    def memory_usage(self, itemsize=8):
        unit = self.dims[self.HW_INDS].prod() * self.dims[self.BATCH_IND] * itemsize
        return [ch * unit for ch in self.get_intermediate_dims()]
