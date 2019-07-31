import sys
import itertools
import numpy as np

def subseteq(a, b):
    ''''a >= b'''
    return b == (a & b)

def length(a):
    return bin(a).count('1')

def splitl(seq, val):
    return [list(g[1]) for g in itertools.groupby(seq, key= lambda x: x != val) if g[0]]

def to_graph(varset, num_vars=4):
    a = []
    for v in varset:
        a.append([int(c) for c in to_bin(v, num_vars)])
    return a

def to_bin(v, num_vars):
    return reversed(format(v, '0%db' % num_vars))

def enumerate_order(num_vars):
    #return [list(order) for order in itertools.permutations(range(num_vars))]
    return list(itertools.permutations(range(1, num_vars + 1)))
    

        
class EnumOuterGraph():
    def __init__(self, num_vars, spatial_dim, dims=None):
        self.num_vars = num_vars
        self.spatial_dim = spatial_dim
        self.dims = np.array(dims)
        self.POWERSET = range(2 ** num_vars)
        self.EMPTY = 0

    def to_bin(self, v):
        return to_bin(v, self.num_vars)

    def num_param(self, varset):
        return np.sum([np.prod(self.dims[[c == '1' for c in self.to_bin(v)]]) for v in varset])
        
    def listup_child(self, ancestry):
        if len(ancestry) > self.num_vars:
            return []
        min_len = length(ancestry[-1])
        return [p for p in self.POWERSET
                if (length(p) <= min_len)
                and (p < ancestry[-1])
                and (not any([subseteq(q, p) for q in ancestry ]))]

    def is_proper(self, v):
        '''
        check 1. the filter indices appear just once
              2. the channel indices appear at least once
        '''

        count = np.sum(np.array(to_graph(v, self.num_vars)), axis=0)
    
        if np.any(count[:self.spatial_dim] != 1):
            return False
        elif np.any(count[self.spatial_dim:self.spatial_dim+2] < 1):
            return False
        else:
            return True

    def _enumerate(self, ancestry):
        children = self.listup_child(ancestry)
        if len(children) == 0:
            return ancestry + [self.EMPTY]
        else:
            return [e for child in children for e in self._enumerate(ancestry + [child])]

    def enumerate(self):
        ans = []
        for r in reversed(self.POWERSET[1:]):
            ans += [l for l in splitl(self._enumerate([r]), self.EMPTY) if self.is_proper(l)]
        #return sorted(ans, key=self.num_param)
        return ans

    
def is_proper(v, num_vars, spatial_dim):
    '''
    check 1. the filter indices appear just once
          2. the inner indices appear at least twice
          3. the indices pattern should not be inclusive among nodes
          4. each inner index should be distributed in a different way
    '''
    count = np.sum(np.array(to_graph(v, num_vars)), axis=0)
    if np.any(count[spatial_dim+2:] < 2):
        return False

    return True

def encode_graph(graph, spatial_dim):
    values = [graph.shape[0], spatial_dim] + graph.reshape(-1).tolist()
    return "_".join(map(str, values))

def relu_patterns(num_rows):
    patterns = itertools.product([0, 1], repeat=num_rows)
    return [np.array(x, dtype=int).reshape(-1, 1) for x in patterns]

if __name__ == '__main__':    
    num_vars = int(sys.argv[1])
    spatial_dim = int(sys.argv[2])
    nonlinear_opt = int(sys.argv[3])
    
    enum_outer = EnumOuterGraph(num_vars, spatial_dim)
    all_outer = enum_outer.enumerate()
    #print(len(all_outer), max([len(a) for a in all_outer]))
    #print([enum_outer.num_param(v) for v in all_outer])

    f = lambda v: is_proper(v, num_vars, spatial_dim)
    all_proper = list(filter(f, all_outer))
    #print(len(all_proper))

    for v in all_proper:
        #count = np.sum(np.array(to_graph(v, num_vars)), axis=0)
        #print(count)
        #if count[2] == 3:
        #    print(np.array(to_graph(v, num_vars)))
        
        graph = np.array(to_graph(v, num_vars))
        if nonlinear_opt == 0: # no nonlinear
            rps  = relu_patterns(graph.shape[0])[:1]
        elif nonlinear_opt == 1: # all nonlinear combinations:
            rps  = relu_patterns(graph.shape[0])[1:]
        else: # always nonlinear
            rps  = [relu_patterns(graph.shape[0])[-1]]

        for rp in rps:
            graph_with_relu = np.hstack((graph, rp))
            print(encode_graph(graph_with_relu, spatial_dim))
