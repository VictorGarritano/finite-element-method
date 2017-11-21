import numpy as np

def gradient_interpolator(ksi_bold):
    """B Matrix"""
    ksi, eta = ksi_bold[0], ksi_bold[1]
    grad_phi1 = [-0.25*(1-eta), -0.25*(1-ksi)]
    grad_phi2 = [0.25*(1-eta), -0.25*(1+ksi)]
    grad_phi3 = [0.25*(1+eta), 0.25*(1+ksi)]
    grad_phi4 = [-0.25*(1+eta), 0.25*(1-ksi)]

    return np.array([
        grad_phi1,
        grad_phi2,
        grad_phi3,
        grad_phi4
    ])

jacobian = lambda x, y: x.dot(y)
gamma = lambda j: np.linalg.inv(j)
det_J = lambda j: np.linalg.det(j)

gauss_points = [[-np.sqrt(3)/3,-np.sqrt(3)/3],
                [np.sqrt(3)/3, -np.sqrt(3)/3],
                [np.sqrt(3)/3, np.sqrt(3)/3],
                [-np.sqrt(3)/3, np.sqrt(3)/3]]

phi_1 = lambda ksi, eta: 0.25*(1-ksi)*(1-eta)
phi_2 = lambda ksi, eta: 0.25*(1+ksi)*(1-eta)
phi_3 = lambda ksi, eta: 0.25*(1+ksi)*(1+eta)
phi_4 = lambda ksi, eta: 0.25*(1-ksi)*(1+eta)

phis = np.array([
    phi_1,
    phi_2,
    phi_3,
    phi_4
])

def build_local(M, Q, Q_ab, f_on_nodes, prescribed_vals, flow_vals):
    # from itertools import product
    # tipo, flow = flow_vals
    #
    # flow_matrix = np.zeros((4,4))
    # if tipo == 1:
    #     eta = -1
    #     pg = np.sqrt(3)/3
    #     for x in product([0,1], repeat=2):
    #         i, j = x
    #         flow_matrix[i, j] = phis[i](-pg,eta)*phis[j](-pg,eta) +\
    #          phis[i](pg, eta)*phis[j](pg, eta)
    # elif tipo == 2:
    #     pass
    # elif tipo == 3:
    #     pass
    # else:
    #     pass

    K_e = np.zeros((4,4))
    for p in gauss_points:
        B = gradient_interpolator(p)
        J = jacobian(M, B)
        G = gamma(J)
        D = det_J(J)
        K_e += (B @ G.T @ Q @ G @ B.T) * D
    F_e = (Q_ab @ f_on_nodes) * D - (K_e @ prescribed_vals)
    return K_e, F_e

if __name__ == '__main__':
    M = np.array([
        [0,0],
        [1,0],
        [1,1],
        [0,1]
    ]).T

    Q = np.array([
        [1,0],
        [0,1]
    ])
    print(build_local_k(M,Q))
