import numpy as np 
from scipy import sparse

# Since it is much, much faster to define the Hamiltonian in terms of precomputed values, here we define a function that returns 
# a general Hamiltonian function which is based on precomputed values. Thus, we don't need to (explicitly) hard-code the Hamiltonian for 
# each set of parameters
def make_H4_sparse(K, J, precompute_pairs = True, precompute_quads = True):

    # K = Number of fermionic modes/creation-annihilation operators
    # J = propto variance of gaussian distribution, sets an overall "energy level"
    Q = 4 # order of coupling
    N = 2*K # number of fermions
    N_DIM = 2**K # Hilbert-space dimension

    ########################### Define fermionic modes #############################
    cr = sparse.csr_array(np.array([[0,1],[0,0]]))
    an = sparse.csr_array(np.array([[0,0],[1,0]]))
    id = sparse.csr_array(np.identity(2))
    id2 = sparse.csr_array(np.array([[-1,0],[0,1]]))

    def c(n):
        factors = [id for i in range(n-1)]+[cr]+[id2 for i in range(K-n)]
        out = factors[0]
        for i in range(1, K):
            out = sparse.kron(out,factors[i])
        return out

    def cd(n):
        factors = [id for i in range(n-1)]+[an]+[id2 for i in range(K-n)]
        out = factors[0]
        for i in range(1, K):
            out = sparse.kron(out,factors[i])
        return out
    
    ########################### Define fermions ################################
    psi = [None for i in range(N)] 
    for i in range(1,K+1):
        psi[2*(i-1)] = sparse.csr_matrix((c(i)+cd(i))/np.sqrt(2))
        psi[2*(i-1)+1] = sparse.csr_matrix((c(i)-cd(i))*(-1j/np.sqrt(2)))

    ########################### Pre-compute pairwise inner products ############
    if precompute_pairs:
        psi_pairs = [None for i in range(N**2)] 
        for i in range(N-1):
            for j in range(i+1, N):
                index = i*N+j
                psi_pairs[index] = psi[i]@psi[j]

    ########################### Pre-compute quadwise inner products ############
    if precompute_quads:
        if not precompute_pairs:
            raise NotImplementedError("precompute_pairs is False, but precompute_quads is True. This is possible but unnecessarily slow. Try again with precompute_pairs=True")

        psi_quads = [None for i in range(N**4)]
        for i in range(N-3):
            for j in range(i+1, N-2):
                for k in range(j+1, N-1):
                    for l in range(k+1, N):
                        index = i*(N**3)+j*(N**2)+k*N+l
                        psi_quads[index] = psi_pairs[i*N+j]@psi_pairs[k*N+l]
                        
    ########################### Define Hamiltonian #############################
    if precompute_pairs: 
        if precompute_quads:
            def H4_func(js):
                H = sparse.csr_matrix(np.zeros((N_DIM, N_DIM), dtype=np.complex128))
                for i in range(N-3):
                    for j in range(i+1, N-2):
                        for k in range(j+1, N-1):
                            for l in range(k+1, N):
                                index = i*(N**3)+j*(N**2)+k*N+l
                                H += (1j**(Q/2))*js[i, j, k, l]*(psi_quads[index])
                return H
            return H4_func, psi, psi_pairs, psi_quads
        else:
            def H4_func(js):
                H = sparse.csr_matrix(np.zeros((N_DIM, N_DIM), dtype=np.complex128))
                for i in range(N-3):
                    for j in range(i+1, N-2):
                        psi_ij = psi_pairs[i*N+j]

                        for k in range(j+1, N-1):
                            for l in range(k+1, N):
                                H += (1j**(Q/2))*js[i, j, k, l]*(psi_ij@psi_pairs[k*N+l])
                return H
            return H4_func, psi, psi_pairs
    else:
        def H4_func(js):
                H = sparse.csr_matrix(np.zeros((N_DIM, N_DIM), dtype=np.complex128))
                for i in range(N-3):
                    for j in range(i+1, N-2):
                        psi_ij = psi[i]@psi[j]

                        for k in range(j+1, N-1):
                            psi_ijk = psi_ij@psi[k]

                            for l in range(k+1, N):
                                H += (1j**(Q/2))*js[i, j, k, l]*(psi_ijk@psi[l])
                return H
        return H4_func, psi

        
def make_H3_sparse(K, J, precompute_pairs = True, precompute_tris = True):

    # K = Number of fermionic modes/creation-annihilation operators
    # J = propto variance of gaussian distribution, sets an overall "energy level"
    Q = 3 # order of coupling
    N = 2*K # number of fermions
    N_DIM = 2**K # Hilbert-space dimension

    ########################### Define fermionic modes #############################
    cr = sparse.csr_array(np.array([[0,1],[0,0]]))
    an = sparse.csr_array(np.array([[0,0],[1,0]]))
    id = sparse.csr_array(np.identity(2))
    id2 = sparse.csr_array(np.array([[-1,0],[0,1]]))

    def c(n):
        factors = [id for i in range(n-1)]+[cr]+[id2 for i in range(K-n)]
        out = factors[0]
        for i in range(1, K):
            out = sparse.kron(out,factors[i])
        return out

    def cd(n):
        factors = [id for i in range(n-1)]+[an]+[id2 for i in range(K-n)]
        out = factors[0]
        for i in range(1, K):
            out = sparse.kron(out,factors[i])
        return out
    
    ########################### Define fermions ################################
    psi = [None for i in range(N)] 
    for i in range(1,K+1):
        psi[2*(i-1)] = sparse.csr_matrix((c(i)+cd(i))/np.sqrt(2))
        psi[2*(i-1)+1] = sparse.csr_matrix((c(i)-cd(i))*(-1j/np.sqrt(2)))

    ########################### Pre-compute pairwise inner products ############
    if precompute_pairs:
        psi_pairs = [None for i in range(N**2)] 
        for i in range(N-1):
            for j in range(i+1, N):
                index = i*N+j
                psi_pairs[index] = psi[i]@psi[j]

    ########################### Pre-compute quadwise inner products ############
    if precompute_tris:
        if not precompute_pairs:
            raise NotImplementedError("precompute_pairs is False, but precompute_tris is True. This is possible but unnecessarily slow. Try again with precompute_pairs=True")

        psi_tris = [None for i in range(N**3)]
        for i in range(N-2):
            for j in range(i+1, N-1):
                psi_ij = psi_pairs[i*N+j]
                for k in range(j+1, N):
                    index = i*(N**2)+j*N+k
                    psi_tris[index] = psi_ij@psi[k]
                        
    ########################### Define Hamiltonian #############################
    if precompute_pairs: 

        if precompute_tris:

            def H3_func(js):
                H = sparse.csr_matrix(np.zeros((N_DIM, N_DIM), dtype=np.complex128))
                for i in range(N-2):
                    for j in range(i+1, N-1):
                        for k in range(j+1, N):
                            index = i*(N**2)+j*N+k
                            H += js[i, j, k]*(psi_tris[index])
                H *= (1j**Q)
                return H
            
            return H3_func, psi, psi_pairs, psi_tris
        
        else:

            def H3_func(js):
                H = sparse.csr_matrix(np.zeros((N_DIM, N_DIM), dtype=np.complex128))
                for i in range(N-2):
                    for j in range(i+1, N-1):
                        psi_ij = psi_pairs[i*N+j]

                        for k in range(j+1, N):
                            H += (1j**Q)*js[i, j, k]*(psi_ij@psi[k])
                return H
            
            return H3_func, psi, psi_pairs
    else:

        def H3_func(js):
                H = sparse.csr_matrix(np.zeros((N_DIM, N_DIM), dtype=np.complex128))
                for i in range(N-2):
                    for j in range(i+1, N-1):
                        psi_ij = psi[i]@psi[j]
                        for k in range(j+1, N):
                            H += (1j**Q)*js[i, j, k]*(psi_ij@psi[k])
                return H
        
        return H3_func, psi
