import numpy as np
import itertools

def global2local(local_matrix, EQ_vector, L2G_matrix):
    """Generate the local matrix, and assembly the global matrix iteratively.
    Just a toy example for 2D-finite-element method.
    """
    #get the number of finite elements and the number of
    #nodes in each element.
    nodes_per_elem, elem_n = L2G_matrix.shape

    #get the total number of nodes that generate equations.
    eq_n = EQ_vector[-1]

    #initialize global matrix K.
    K = np.zeros((eq_n, eq_n))

    #for python reasons, the first index of an array is 0.
    #We model the problem starting with 1 everywhere...
    #but we will take advantage of this further up.
    EQ_vector -= 1
    L2G_matrix -= 1

    for elem in range(elem_n):
        #generate all possible combinations among the nodes in a finite element.
        c = itertools.product(range(nodes_per_elem),repeat=2)

        for x in c:
            #get the number of each nodes.
            aux1 = L2G_matrix[x[0]][elem]
            aux2 = L2G_matrix[x[1]][elem]

            #get the equation number of each node, respectively.
            i = EQ_vector[aux1]
            j = EQ_vector[aux2]

            #On this toy example, we are considering every local matrix as a
            #np.ones((nodes_per_elem, nodes_per_elem)) matrix.
            if i != -1 and j != -1:
                K[i][j] += local_matrix[x[0]][x[1]]
    return K

EQ_vector = np.array([1,2,3,0,0,0,4,5,0,6,7,8])

L2G_matrix = np.array(
[
[2,5,6,3,2,8],
[7,1,4,7,12,7],
[3,3,3,8,11,11],
[1,4,9,9,7,10]
]
)
local_matrix = np.ones((4,4))
K = global2local(local_matrix, EQ_vector, L2G_matrix)
print(K)
# print(K == K.T)
