import numpy as np
from k_local import build_local
import itertools

Q = np.array([
        [2,1],
        [1,2]
    ])

def create_mesh(nx, ny, verbose=False):
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    Z = np.array(list(zip(X.ravel(), Y.ravel())))
    nodes = list(enumerate(Z))
    nodes_n = nodes[-1][0] + 1
    elements_n = (nx-1)*(ny-1)

    if verbose:
        print('mesh dimension {0}'.format(X.shape))
        print('number of nodes {0}'.format(nodes_n))
        print('number of elements {0}'.format(elements_n))

    return Z, nodes, nodes_n, elements_n

def nodes_to_element(nx, ny, elements_n, verbose=False):
    G = np.zeros((4, elements_n), dtype=np.int64)

    num = 0
    for d in range(ny-1):
        for i in range((nx-1)):
            G[:,num] = np.array([i+d*nx, i+1+d*nx, i+nx+1+d*nx, i+nx+d*nx])
            num+=1

    if verbose:
        print('global nodes on each element ')
        for x in range(G.shape[1]):
            print(G[:,x])
        print('='*20)

    return G

def map_nodes_to_physical(G, elem, nodes, verbose=False):
    n_ = G[:,elem]
    l = list()

    for idx in n_:
        l.append(nodes[int(idx)][1])

        if verbose:
            print('global node {0} --> point {1} on R2'.format(idx, nodes[int(idx)][1]))

    M = np.array(l).T
    return M

def l2g(nodes, EQ_vector, L2G_matrix, Q_ab, f, prescribed, verbose=False):
    nodes_per_elem, elem_n = L2G_matrix.shape

    if verbose:
        print('shape of element matrix {0}'.format(nodes_per_elem, elem_n))

    eq_n = int(np.max(EQ_vector) + 1)

    if verbose:
        print('number of nodes that generate eqs {0}'.format(eq_n))

    K = np.zeros((eq_n, eq_n))
    F = np.zeros((eq_n, ))

    for elem in range(elem_n):
        global_nodes = L2G_matrix[:,elem]
        f_on_nodes = f[global_nodes]
        pres_on_nodes = prescribed[global_nodes]

        if verbose:
            print('global nodes --> {0}'.format(global_nodes))
            print('f on nodes --> {0}'.format(f_on_nodes))
            print('prescribed on nodes --> {0}'.format(pres_on_nodes))

        c = itertools.product(range(nodes_per_elem),repeat=2)
        M = map_nodes_to_physical(L2G_matrix, elem, nodes, verbose=verbose)
        local_k, local_F = build_local(M, Q, Q_ab, f_on_nodes, pres_on_nodes)

        for x in c:
            aux1 = L2G_matrix[x[0]][elem]
            aux2 = L2G_matrix[x[1]][elem]
            i = EQ_vector[aux1]
            j = EQ_vector[aux2]

            if verbose:
                print('el_node {0} --> glob_node {1} --> eq_vec {2}'.format((x), (aux1, aux2), (i, j)))

            if i != -1 and j != -1:
                K[i][j] += local_k[x[0]][x[1]]

                if verbose:
                    print('\t----> accumulating to global')

        for node in global_nodes:
            i = EQ_vector[node]

            if verbose:
                print('node {0} --> equation {1}'.format(node, i))

            if i != -1:

                if verbose:
                    print('\tnode {0} --> pos {1} on local_f'.format(node, np.where(global_nodes == node)[0]))

                F[i] += local_F[np.where(global_nodes == node)[0]]

        if verbose:
            print('+'*40)

    if verbose:
        print('Global stiffness matrix K')
        print(K)
        print('\nGlobal force vector F')
        print(F)

    return K, F

def generate_Q_ab(verbose=False):
    phi_1 = lambda ksi, eta: 0.25*(1-ksi)*(1-eta)
    phi_2 = lambda ksi, eta: 0.25*(1+ksi)*(1-eta)
    phi_3 = lambda ksi, eta: 0.25*(1+ksi)*(1+eta)
    phi_4 = lambda ksi, eta: 0.25*(1-ksi)*(1+eta)

    gauss_points = [[-np.sqrt(3)/3,-np.sqrt(3)/3],
                    [np.sqrt(3)/3, -np.sqrt(3)/3],
                    [np.sqrt(3)/3, np.sqrt(3)/3],
                    [-np.sqrt(3)/3, np.sqrt(3)/3]]

    interp = np.array([phi_1, phi_2, phi_3, phi_4])

    Q_ab = np.zeros((4,4))

    for i in range(4):
        for j in range(4):
            for pg in gauss_points:
                Q_ab[i, j] += interp[i](pg[0], pg[1])*interp[j](pg[0], pg[1])

                if verbose:
                    print('nodes ({0}, {1}) --> pg = ({2}, {3})'.format(i, j, pg[0], pg[1]))

            if verbose:
                print('+'*10)
    if verbose:
        print('\nQ_ab matrix')
        print(Q_ab)
        print('='*20)

    return Q_ab


def generate_f(f, nodes_n, nodes, verbose=False):
    f = np.zeros((nodes_n,))
    for pts in nodes:
        f[pts[0]] = f_(pts[1])

    if verbose:
        print('f vector')
        for x in f:
            print(x)

    return f

def map_prescribed_nodes(nodes_n, Z, verbose=False):
    prescribed = np.zeros((nodes_n, ))

    gamma_1 = np.where(Z[:,1] == 0)[0]
    prescribed[gamma_1] = np.sin(np.pi*Z[gamma_1,0])

    gamma_2 = np.where(Z[:,0] == 1)[0]

    gamma_3 = np.where(Z[:,1] == 1)[0]
    prescribed[gamma_3] = -np.sin(np.pi*Z[gamma_3,0])
    gamma_4 = np.where(Z[:,0] == 0)[0]

    known_sols = np.concatenate((gamma_1, gamma_2, gamma_3, gamma_4),axis=0)
    d_sols = np.setdiff1d(np.arange(nodes_n), known_sols)

    if verbose:
        print('gamma_1 {0}'.format(gamma_1))
        print('gamma_2 {0}'.format(gamma_2))
        print('gamma_3 {0}'.format(gamma_3))
        print('gamma_4 {0}'.format(gamma_4))
        print('prescribed {0}'.format(prescribed))
        print('nodes with unknown solution {0}'.format(d_sols))

    return d_sols, prescribed

def generate_EQ_vector(nodes_n, d_sols, verbose=False):
    EQ_vector = np.zeros((nodes_n,), dtype=np.int64) - 1
    count = 0

    for idx in range(EQ_vector.shape[0]):
        if idx in d_sols:
            EQ_vector[idx] = count
            count += 1

    if verbose:
        print('EQ_vector {0}'.format(EQ_vector))

    return EQ_vector

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nx", help='granularity on x-axis', type=int)
    parser.add_argument("ny", help='granularity on y-axis', type=int)
    parser.add_argument("--plot", help='whether a surface with the final solution must be plotted', type=bool, default=False)

    args = parser.parse_args()
    kwargs = args.__dict__
    nx = kwargs['nx']
    ny = kwargs['ny']
    plot = kwargs['plot']
    Z, nodes, nnodes, el = create_mesh(nx, ny)
    G = nodes_to_element(nx, ny, el)
    Q_ab = generate_Q_ab()
    f_ = lambda x: (2*np.pi**2)*(2*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])\
        + np.cos(np.pi*x[0])*np.sin(np.pi*x[1]))
    f = generate_f(f_, nnodes, nodes)
    d_sols, prescribed = map_prescribed_nodes(nnodes, Z)
    EQ_vector = generate_EQ_vector(nnodes, d_sols)
    K, F = l2g(nodes, EQ_vector, G, Q_ab, f, prescribed)
    print('solving linear system...')
    d = np.linalg.solve(K, F)
    print('generating the final solution vector...')
    sols = prescribed
    sols[d_sols] = d
    if plot:
        print('plotting...')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = Z[:,0], Z[:,1]

        ax.plot_trisurf(X, Y, sols)
        ax.view_init(20,35)
        plt.show()
