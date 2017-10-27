import numpy as np

def gradient_interpolator(ksi_bold):
    """Matrix B"""
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
                [-np.sqrt(3)/3, np.sqrt(3)/3],
                [np.sqrt(3)/3, np.sqrt(3)/3]]

def build_local_k(M, Q):
    K_e = np.zeros((4,4))
    for p in gauss_points:
        B = gradient_interpolator(p)
        J = jacobian(M, B)
        G = gamma(J)
        D = det_J(J)
        K_e += B.dot(G.T).dot(Q).dot(G).dot(B.T)*D
    return K_e

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
