import numpy as np
import logging
# logging.basicConfig(level=logging.DEBUG)
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def same_hyperedge(graph, i, j):
    return np.all((graph[:, i] > 0) == (graph[:, j] > 0))

def proj(val, lower, upper):
    return min(max(val, lower), upper)

def count(arr, axis=None):
    return np.sum(np.where(arr >= 1, 1, 0), axis=axis)


class Mutation:
    def __init__(self, seed=1, num_vars=4, max_node=5, rank_when_splitted=1, max_dilation=3,
                 num_retry=1000,
                 shapes={'batch': 32, 'inch': 64, 'outch': 64, 'image': [32, 32], 'filter': [3, 3], 'inner_exp': 0}):
        np.random.seed(seed)
        self.num_vars = num_vars
        self.max_node = max_node
        self.rank_when_splitted = rank_when_splitted
        self.max_dilation = max_dilation
        self.num_retry = num_retry
        self.shapes = shapes
        self.gprob = {'flip': 1 / 2,
                      'split': 1 / 2 * 1 / 2,
                      'merge': 1 / 2 * 1 / 2}
        self.rprob = {'connect': 1 / 2,
                      'disconnect': 1 / 2}
        self.oprob = {'push': 1 / 2,
                      'swap': 1 / 2}

    def mutate(self, graph, relus):
        for t in range(self.num_retry):
            g = graph.copy()
            r = relus.copy()
            for prob in [self.gprob, self.rprob, self.oprob]:
                orig_g = g.copy()
                orig_r = r.copy()
                g, r = self._mutate(g, r, prob)
                if not self.is_consistent(g):
                    g = orig_g
                    r = orig_r

            if g.shape[0] == 1:
                continue
            return g, r

        return graph, relus

    def _mutate(self, graph, relus, mutate_prob):
        i = np.random.choice(len(mutate_prob), 1, p=list(mutate_prob.values()))
        mutate_type = list(mutate_prob.keys())[int(i)]
        logging.debug('---' + mutate_type)
        mutated_graph, muated_relu = getattr(self, mutate_type)(graph, relus)
        return self.cleanup_inner(mutated_graph), muated_relu

    def swap(self, graph, relus):
        if graph.shape[0] == 1:
            return graph, relus

        gr = np.hstack((graph, relus.reshape(-1, 1)))
        i, j = np.random.choice(gr.shape[0], 2).tolist()
        val = gr[i, :].copy()
        gr[i, :] = gr[j, :]
        gr[j, :] = val
        return gr[:, :-1], gr[:, -1]

    def push(self, graph, relus):
        if graph.shape[0] == 1:
            return graph, relus

        gr = np.hstack((graph, relus.reshape(-1, 1)))
        i, j = np.random.choice(gr.shape[0], 2).tolist()
        gr = self._push(gr, i, j)
        return gr[:, :-1], gr[:, -1]

    def _push(self, gr, i, j):
        val = gr[i, :]
        gr = np.delete(gr, i, axis=0)
        gr = np.insert(gr, j, val, axis=0)
        return gr

    def is_consistent(self, arr):
        return np.all(count(arr[:, :self.num_vars], axis=0) > 0) and \
               np.all(count(arr[:, :self.num_vars], axis=1) > 0)

    def flip(self, graph, relus):
        orig_graph = graph.copy()
        i = np.random.choice(graph.shape[0], 1)
        j = np.random.choice(self.num_vars, 1)
        graph[i, j] = 1 if graph[i, j] == 0 else 0

        k = np.random.choice(graph.shape[0], 1)
        relus[k] = 1 if relus[k] == 0 else 0

        return graph, relus

    def split(self, graph, relus):
        if graph.shape[0] > self.max_node:
            return graph, relus

        splittable_rows = np.where(count(graph[:, :self.num_vars], axis=1) >= 2)[0]
        if len(splittable_rows) == 0:
            return graph, relus

        i = np.random.choice(splittable_rows, 1)
        ind = np.random.permutation(np.where(graph[i, :self.num_vars] > 0)[1])
        n = np.random.choice(range(1, len(ind)))

        # split outer indices
        graph = np.vstack([graph, graph[i, :]])
        graph[i, ind[:n]] = 0
        graph[-1, ind[n:]] = 0
        graph[-1, self.num_vars:] = 0

        # add inner index
        graph = np.hstack([graph, np.zeros((graph.shape[0], 1), dtype=int)])
        graph[i, -1] = self.rank_when_splitted
        graph[-1, -1] = self.rank_when_splitted

        graph = self._push(graph, -1, i)
        relus = np.insert(relus, i, relus[i])
        return graph, relus

    def cleanup_inner(self, graph):
        ### if there exists disconnected inner lines, remove them
        connect_ind = (count(graph[:, self.num_vars:] > 0, axis=0) >= 2).tolist()
        graph = graph[:, [True] * self.num_vars + connect_ind]

        # print(graph)
        ### check if there exists the same hyperedge
        for i in reversed(range(self.num_vars + 1, graph.shape[1])):
            for j in reversed(range(self.num_vars, i)):
                # print(i, j, same_hyperedge(graph, i, j))
                if same_hyperedge(graph, i, j):
                    graph[:, j] += graph[:, i]
                    graph = np.delete(graph, i, axis=1)
                    break

        return graph

    def merge(self, graph, relus):
        if graph.shape[0] <= 2:
            return graph, relus

        orig_graph = graph.copy()
        # i and j are rows to be merged
        i, j = np.random.choice(graph.shape[0], 2, replace=False).tolist()
        graph[i, :] = graph[i, :] | graph[j, :]
        if np.all(graph[i, :self.num_vars] == 1):
            return orig_graph, relus
        graph = np.delete(graph, j, axis=0)
        relus = np.delete(relus, j)

        return graph, relus

    @staticmethod
    def connect(graph, relus):
        if graph.shape[0] == 1:
            return graph, relus

        if graph.shape[0] == 2:
            hyperedge = [0, 1]
        else:
            hyperedges = [i for i in powerset(range(graph.shape[0])) if len(i) >= 2]
            hyperedge = np.random.choice(hyperedges)

        # add new column as a hyperedge
        graph = np.hstack([graph, np.zeros([graph.shape[0], 1], dtype=int)])
        graph[hyperedge, -1] = 1
        return graph, relus

    def disconnect(self, graph, relus):
        if graph.shape[1] == self.num_vars:
            return graph, relus

        i = np.random.choice(range(self.num_vars, graph.shape[1]), 1)
        graph[:, i] -= graph[:, i] > 0

        if count(graph[:, i]) == 0:
            graph = np.delete(graph, i, axis=1)

        return graph, relus

if __name__ == '__main__':
    # graph = np.array([[0, 1, 0, 1, 0, 1],
    #                  [1, 1, 0, 0, 1, 1],
    #                  [0, 0, 1, 0, 1, 1]])
    graph = np.array([[1, 1, 1, 1]])
    relus = np.array([1])
    print('Original graph')
    print(graph)
    shapes = {'batch': 32, 'inch': 64, 'outch': 64, 'image': [32, 32], 'filter': [3, 3], 'inner_exp': 1}

    mut = Mutation(shapes=shapes)
    for i in range(100):
        graph, relus = mut.mutate(graph, relus)
        print(graph)
        print(relus)
