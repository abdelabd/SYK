import os 
import time
from tqdm import tqdm
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
import itertools

class HamiltonianGenerator():
    def __init__(self, N, J, n_jobs=20, parent_dir=os.path.join("Excel", "N2_SUSY_SYK")):
        # Physical constants 
        self.N = N
        self.J = J
        self.N_DIM = 2**N # Hilbert-space dimension
        self.Q_COUPLING = 3

        # Computer stuff
        self.N_JOBS = n_jobs
        
        # Directories
        self.PARENT_DIR = parent_dir
        self.PSI_DIR = os.path.join(parent_dir, "Psi", f"N{N}")
        self.RESULT_DIR = os.path.join(parent_dir, "Simulated Hamiltonians", f"N{N}")
        os.makedirs(self.PSI_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

        # Fundamental operators
        self.CR = sparse.csr_array(np.array([[0,1],[0,0]]))
        self.AN = sparse.csr_array(np.array([[0,0],[1,0]]))
        self.ID = sparse.csr_array(np.identity(2))
        self.ID2 = sparse.csr_array(np.array([[-1,0],[0,1]]))

        # Initialize fermions and necessary dot-products
        self.make_fermions()

    # Fermion creation functions
    def psi(self, n):
        factors = [id for i in range(n-1)]+[self.CR]+[self.ID2 for i in range(self.N-n)]
        out = factors[0]
        for i in range(1, self.N):
            out = sparse.kron(out,factors[i])
        return out
    
    def psi_dagger(self, n):
        factors = [id for i in range(n-1)]+[self.AN]+[self.ID2 for i in range(self.N-n)]
        out = factors[0]
        for i in range(1, self.N):
            out = sparse.kron(out,factors[i])
        return out
        
    # Memoized fermion dot-products
    def psi_psi(self, i, j):
        return self.psi_all[i]@self.psi_all[j]

    def psi_psi_psi(self, psi_psi_ij, k):
        return psi_psi_ij@self.psi_all[k]

    # Initialize fermions and necessary dot-products
    def make_fermions(self):
        # Try loading the psi's and psi_dagger's if previously computed, otherwise compute and save them
        try:
            self.psi_all = {}
            for i in range(1,self.N+1):
                psi_i = sparse.load_npz(os.path.join(self.PSI_DIR, f"psi_{i}.npz"))
                self.psi_all[i] = psi_i
        except FileNotFoundError:
            print("Computing psi's...")
            self.psi_all = Parallel(n_jobs=self.N_JOBS)(delayed(self.psi)(n) for n in range(1,self.N+1))
            self.psi_all = {k+1:v for k,v in enumerate(self.psi_all)}
            for i, psi_i_sparse in self.psi_all.items():
                sparse.save_npz(os.path.join(self.PSI_DIR, f"psi_{i}.npz"), psi_i_sparse)

        try:
            self.psi_dagger_all = {}
            for i in range(1,self.N+1):
                psi_dagger_i = sparse.load_npz(os.path.join(self.PSI_DIR, f"psi_dagger_{i}.npz"))
                self.psi_dagger_all[i] = psi_dagger_i
        except FileNotFoundError:
            print("Computing psi_dagger's...")
            self.psi_dagger_all = Parallel(n_jobs=self.N_JOBS)(delayed(self.psi_dagger)(n) for n in range(1,self.N+1))
            self.psi_dagger_all = {k+1:v for k,v in enumerate(self.psi_dagger_all)}
            for i, psi_dagger_i_sparse in self.psi_dagger_all.items():
                sparse.save_npz(os.path.join(self.PSI_DIR, f"psi_dagger_{i}.npz"), psi_dagger_i_sparse)

        # Try loading the pairwise inner-products (psi_psi_all) if previously computed, otherwise compute and save them
        try:
            self.psi_psi_all = {}
            for i in range(1, self.N):
                for j in range(i+1, self.N+1):
                    psi_psi_ij = sparse.load_npz(os.path.join(self.PSI_DIR, f"psi_psi_{i}{j}.npz"))
                    self.psi_psi_all[(i,j)] = psi_psi_ij
            pairs_dur = None
        except FileNotFoundError:
            print("Computing psi_psi's...")
            tic = time.time()
            self.psi_psi_all = {}
            for i in range(1, self.N):
                psi_psi_i = Parallel(n_jobs=self.N_JOBS)(delayed(self.psi_psi)(i,j) for j in range(i+1, self.N+1))
                psi_psi_i = {(i, i+j+1):v for j,v in enumerate(psi_psi_i)}
                for (i_label, j_label), psi_psi_ij_sparse in psi_psi_i.items():
                    sparse.save_npz(os.path.join(self.PSI_DIR, f"psi_psi_{i_label}{j_label}.npz"), psi_psi_ij_sparse)
                self.psi_psi_all.update(psi_psi_i)
            pairs_dur = time.time() - tic

        # Try loading triple inner-products (psi_psi_psi_all) if previously computed, otherwise compute and save them
        try:
            self.psi_psi_psi_all = {}
            for i in range(1, self.N-1):
                for j in range(i+1, self.N):
                    for k in range(j+1, self.N+1):
                        psi_psi_psi_ijk = sparse.load_npz(os.path.join(self.PSI_DIR, f"psi_psi_psi_{i}{j}{k}.npz"))
                        self.psi_psi_psi_all[(i,j,k)] = psi_psi_psi_ijk
        except FileNotFoundError:
            print("Computing psi_psi_psi's...")
            exp_tri_dur = pairs_dur*((self.N-3)**3)/((self.N-2)**2)
            print(f"Expected tri's duration: {exp_tri_dur//60} minutes, {exp_tri_dur%60} seconds")
            self.psi_psi_psi_all = {}
            for i in range(1, self.N-1):
                for j in range(i+1, self.N):
                    psi_psi_ij = self.psi_psi_all[(i,j)]
                    psi_psi_psi_ij = Parallel(n_jobs=self.N_JOBS)(delayed(self.psi_psi_psi)(psi_psi_ij, k) for k in range(j+1, self.N+1))
                    psi_psi_psi_ij = {(i, j, j+k+1):v for k,v in enumerate(psi_psi_psi_ij)}
                    for (i_label, j_label, k_label), psi_psi_psi_ijk_sparse in psi_psi_psi_ij.items():
                        sparse.save_npz(os.path.join(self.PSI_DIR, f"psi_psi_psi_{i_label}{j_label}{k_label}.npz"), psi_psi_psi_ijk_sparse)
                    self.psi_psi_psi_all.update(psi_psi_psi_ij)


    # Helper functions for creating the antisymmetric coefficient tensor
    def levi_civita_tensor(self, dim):   
        arr=np.zeros(tuple([dim for i in range(dim)]), dtype=np.int32)
        for x in itertools.permutations(tuple(range(dim))):
            mat = np.zeros((dim, dim), dtype=np.int32)
            for i, j in zip(range(dim), x):
                mat[i, j] = 1
            arr[x]=int(np.linalg.det(mat))
        return arr

    def asym_perm(self, iterable):
        n_elem = len(iterable)
        if len(set(iterable)) < n_elem:
            return 0 # <-- If there are repeated elements, levi-civita value is identically zero 

        order_0 = tuple(list(range(n_elem)))
        all_orders = list(itertools.permutations(order_0))
        all_permutations = [tuple([iterable[i] for i in order]) for order in all_orders]

        lc_tensor = self.levi_civita_tensor(n_elem)
        out = {}
        for i in range(len(all_orders)):
            order_i = all_orders[i]
            multiplier = lc_tensor[order_i]

            permutation_i = all_permutations[i]
            out[permutation_i] = multiplier
        return out
    
    # Function to create the antisymmetric coefficient tensor
    def make_C(self, random_seed):
        np.random.seed(random_seed)

        sigma_C_squared = 2*self.J/(self.N**2)
        sigma_C_prime_squared = sigma_C_squared*(self.N**2)/((self.N-1)*(self.N-2))
        sigma_C = np.sqrt(sigma_C_squared)
        sigma_C_prime = np.sqrt(sigma_C_prime_squared)

        # Generate distribution for X', Y'
        n_upper_elem = self.N*(self.N-1)*(self.N-2)//6
        sigma_XY_prime = sigma_C_prime/np.sqrt(2)
        X_prime = np.random.normal(0, sigma_XY_prime, size=(n_upper_elem))
        Y_prime = np.random.normal(0, sigma_XY_prime, size=(n_upper_elem))

        # Generate distribution for C'
        C_prime = X_prime + 1j*Y_prime

        # Initialize upper-upper-triangle array
        C_upper_upper = np.zeros(shape = [self.N for i in range(self.Q_COUPLING)], dtype=np.complex128)

        index = 0
        for i in range(self.N-2):
            for j in range(i+1, self.N-1):
                for k in range(j+1, self.N):
                    C_upper_upper[i, j, k] = C_prime[index]
                    index += 1

        # Antisymmetrize C_upper_upper to form full tensor C
        C = np.zeros(shape=[self.N for i in range(self.Q_COUPLING)], dtype=np.complex128)
        axes_og = range(len(C.shape))
        axes_asym_perms = self.asym_perm(axes_og)
        for axes, multiplier in axes_asym_perms.items():
            addendum = multiplier*np.transpose(C_upper_upper, axes)
            C += addendum

        return C

    # Function to create the supercharge Q
    def make_Q(self, random_seed):
        C = self.make_C(random_seed)
        Q = sparse.csr_array(np.zeros((self.N_DIM, self.N_DIM), dtype=np.complex128))
        for i_index in range(self.N-2):
            i_label = i_index+1

            for j_index in range(i_index+1, self.N-1):
                j_label = j_index+1

                for k_index in range(j_index+1, self.N):
                    k_label = k_index+1

                    C_ijk = C[i_index, j_index, k_index]
                    psi_psi_psi_ijk = self.psi_psi_psi_all[(i_label, j_label, k_label)]
                    Q += C_ijk*psi_psi_psi_ijk
            
        Q *= 1j
        return Q
    
    # Finally, the function to create the Hamiltonian
    def make_H(self, random_seed):
        Q = self.make_Q(random_seed)
        Q_bar = np.transpose(np.conjugate(Q))
        H = Q@Q_bar + Q_bar@Q
        return H