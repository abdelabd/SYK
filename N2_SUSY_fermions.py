import os 
from tqdm import tqdm
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed



def make_fermions(N, n_jobs=20, check_algebra=False, n_parent_dir=os.path.join("Excel", "N2_SUSY_SYK", "Psi")): # N = Number of fermions
    N_DIM = 2**N # Hilbert-space dimension
    N_DIR = os.path.join(n_parent_dir, f"N{N}")
    os.makedirs(N_DIR, exist_ok=True)
    

    ######################### Define what we need for fermions ######################################
    cr = sparse.csr_array(np.array([[0,1],[0,0]]))
    an = sparse.csr_array(np.array([[0,0],[1,0]]))
    id = sparse.csr_array(np.identity(2))
    id2 = sparse.csr_array(np.array([[-1,0],[0,1]]))

    def psi(n):
        factors = [id for i in range(n-1)]+[cr]+[id2 for i in range(N-n)]
        out = factors[0]
        for i in range(1, N):
            out = sparse.kron(out,factors[i])
        return out

    def psi_dagger(n):
        factors = [id for i in range(n-1)]+[an]+[id2 for i in range(N-n)]
        out = factors[0]
        for i in range(1, N):
            out = sparse.kron(out,factors[i])
        return out



    ####################### Load or precompute fermions ############################################
    # Try loading the psi's and psi_dagger's if previously computed, otherwise compute and save them
    try:
        psi_all = {}
        for i in range(1,N+1):
            psi_i = np.load(os.path.join(N_DIR, f"psi_{i}.npy"))
            psi_all[i] = sparse.csr_matrix(psi_i)
    except FileNotFoundError:
        print("Computing psi's...")
        psi_all = Parallel(n_jobs=n_jobs)(delayed(psi)(n) for n in range(1,N+1))
        psi_all = {k+1:v for k,v in enumerate(psi_all)}
        for i, psi_i_sparse in psi_all.items():
            psi_i = psi_i_sparse.toarray()
            np.save(os.path.join(N_DIR, f"psi_{i}.npy"), psi_i)

    try:
        psi_dagger_all = {}
        for i in range(1,N+1):
            psi_dagger_i = np.load(os.path.join(N_DIR, f"psi_dagger_{i}.npy"))
            psi_dagger_all[i] = sparse.csr_matrix(psi_dagger_i)
    except FileNotFoundError:
        print("Computing psi_dagger's...")
        psi_dagger_all = Parallel(n_jobs=n_jobs)(delayed(psi_dagger)(n) for n in range(1,N+1))
        psi_dagger_all = {k+1:v for k,v in enumerate(psi_dagger_all)}
        for i, psi_dagger_i_sparse in psi_dagger_all.items():
            psi_dagger_i = psi_dagger_i_sparse.toarray()
            np.save(os.path.join(N_DIR, f"psi_dagger_{i}.npy"), psi_dagger_i)

    # Confirm that psi_dagger as defined indeed gives psi_dagger
    for i, psi_i in psi_all.items():
        psi_daggeri = psi_dagger_all[i]
        assert(np.allclose(np.transpose(np.conjugate(psi_i.toarray())), psi_daggeri.toarray()))



    ####################### Load or precompute pairwise dot-products ################################
    def psi_psi(i, j):
        return psi_all[i]@psi_all[j]

    def psi_psi_dagger(i, j):
        return psi_all[i]@psi_dagger_all[j]

    def psi_dagger_psi(i, j):
        return psi_dagger_all[i]@psi_all[j]

    def psi_dagger_psi_dagger(i, j):
        return psi_dagger_all[i]@psi_dagger_all[j]


    # psi_psi
    if check_algebra:
        try:
            psi_psi_all = {}
            for i in range(1, N+1):
                for j in range(1, N+1):
                    psi_psi_ij = np.load(os.path.join(N_DIR, f"psi_psi_{i}{j}.npy"))
                    psi_psi_all[(i,j)] = sparse.csr_matrix(psi_psi_ij)
        except FileNotFoundError:
            print("Computing psi_psi's...")
            psi_psi_all = {}
            for i in range(1, N+1):
                psi_psi_i = Parallel(n_jobs=n_jobs)(delayed(psi_psi)(i,j) for j in range(1, N+1))
                psi_psi_i = {(i, j+1):v for j,v in enumerate(psi_psi_i)}
                for (i_label, j_label), psi_psi_ij_sparse in psi_psi_i.items():
                    psi_psi_ij = psi_psi_ij_sparse.toarray()
                    np.save(os.path.join(N_DIR, f"psi_psi_{i_label}{j_label}.npy"), psi_psi_ij)
                psi_psi_all.update(psi_psi_i)
    else:
        try:
            psi_psi_all = {}
            for i in range(1, N):
                for j in range(i+1, N+1):
                    psi_psi_ij = np.load(os.path.join(N_DIR, f"psi_psi_{i}{j}.npy"))
                    psi_psi_all[(i,j)] = sparse.csr_matrix(psi_psi_ij)
        except FileNotFoundError:
            print("Computing psi_psi's...")
            psi_psi_all = {}
            for i in range(1, N):
                psi_psi_i = Parallel(n_jobs=n_jobs)(delayed(psi_psi)(i,j) for j in range(i+1, N+1))
                psi_psi_i = {(i, i+j+1):v for j,v in enumerate(psi_psi_i)}
                for (i_label, j_label), psi_psi_ij_sparse in psi_psi_i.items():
                    psi_psi_ij = psi_psi_ij_sparse.toarray()
                    np.save(os.path.join(N_DIR, f"psi_psi_{i_label}{j_label}.npy"), psi_psi_ij)
                psi_psi_all.update(psi_psi_i)

    # We only need daggers to check the fermion algebra
    # psi_psi_dagger
    psi_psi_dagger_all = None
    psi_dagger_psi_all = None
    psi_dagger_psi_dagger_all = None
    if check_algebra:
        try:
            psi_psi_dagger_all = {}
            for i in range(1, N+1):
                for j in range(1, N+1):
                    psi_psi_dagger_ij = np.load(os.path.join(N_DIR, f"psi_psi_dagger_{i}{j}.npy"))
                    psi_psi_dagger_all[(i,j)] = sparse.csr_matrix(psi_psi_dagger_ij)
        except FileNotFoundError:
            print("Computing psi_psi_dagger's...")
            psi_psi_dagger_all = {}
            for i in range(1, N+1):
                psi_psi_dagger_i = Parallel(n_jobs=n_jobs)(delayed(psi_psi_dagger)(i,j) for j in range(1, N+1))
                psi_psi_dagger_i = {(i, j+1):v for j,v in enumerate(psi_psi_dagger_i)}
                for (i_label, j_label), psi_psi_dagger_ij_sparse in psi_psi_dagger_i.items():
                    psi_psi_dagger_ij = psi_psi_dagger_ij_sparse.toarray()
                    np.save(os.path.join(N_DIR, f"psi_psi_dagger_{i_label}{j_label}.npy"), psi_psi_dagger_ij)
                psi_psi_dagger_all.update(psi_psi_dagger_i)

        # psi_dagger_psi
        try:
            psi_dagger_psi_all = {}
            for i in range(1, N+1):
                for j in range(1, N+1):
                    psi_dagger_psi_ij = np.load(os.path.join(N_DIR, f"psi_dagger_psi_{i}{j}.npy"))
                    psi_dagger_psi_all[(i,j)] = sparse.csr_matrix(psi_dagger_psi_ij)
        except FileNotFoundError:
            print("Computing psi_dagger_psi's...")
            psi_dagger_psi_all = {}
            for i in range(1, N+1):
                psi_dagger_psi_i = Parallel(n_jobs=n_jobs)(delayed(psi_dagger_psi)(i,j) for j in range(1, N+1))
                psi_dagger_psi_i = {(i, j+1):v for j,v in enumerate(psi_dagger_psi_i)}
                for (i_label, j_label), psi_dagger_psi_ij_sparse in psi_dagger_psi_i.items():
                    psi_dagger_psi_ij = psi_dagger_psi_ij_sparse.toarray()
                    np.save(os.path.join(N_DIR, f"psi_dagger_psi_{i_label}{j_label}.npy"), psi_dagger_psi_ij)
                psi_dagger_psi_all.update(psi_dagger_psi_i)

        # psi_dagger_psi_dagger
        try:
            psi_dagger_psi_dagger_all = {}
            for i in range(1, N+1):
                for j in range(1, N+1):
                    psi_dagger_psi_dagger_ij = np.load(os.path.join(N_DIR, f"psi_dagger_psi_dagger_{i}{j}.npy"))
                    psi_dagger_psi_dagger_all[(i,j)] = sparse.csr_matrix(psi_dagger_psi_dagger_ij)
        except FileNotFoundError:
            print("Computing psi_dagger_psi_dagger's...")
            psi_dagger_psi_dagger_all = {}
            for i in range(1, N+1):
                psi_dagger_psi_dagger_i = Parallel(n_jobs=n_jobs)(delayed(psi_dagger_psi_dagger)(i,j) for j in range(1, N+1))
                psi_dagger_psi_dagger_i = {(i, j+1):v for j,v in enumerate(psi_dagger_psi_dagger_i)}
                for (i_label, j_label), psi_dagger_psi_dagger_ij_sparse in psi_dagger_psi_dagger_i.items():
                    psi_dagger_psi_dagger_ij = psi_dagger_psi_dagger_ij_sparse.toarray()
                    np.save(os.path.join(N_DIR, f"psi_dagger_psi_dagger_{i_label}{j_label}.npy"), psi_dagger_psi_dagger_ij)
                psi_dagger_psi_dagger_all.update(psi_dagger_psi_dagger_i)




    ############################ Check algebra #####################################################
    def process_j(i_index, j_index):
        psi_algebra_satisfied_ij = True

        i_label = i_index+1
        j_label = j_index+1

        # ac_i_j = psi(i)@psi(j)+psi(j)@psi(i)
        ac_i_j = psi_psi_all[(i_label,j_label)]+psi_psi_all[(j_label,i_label)]    
        ac_i_j = ac_i_j.toarray()
        if not np.allclose(ac_i_j, np.zeros((N_DIM, N_DIM))):
            psi_algebra_satisfied_ij = False

        # ac_id_jd = psi_dagger(i)@psi_dagger(j)+psi_dagger(j)@psi_dagger(i)
        ac_idagger_jdagger = psi_dagger_psi_dagger_all[(i_label, j_label)]+psi_dagger_psi_dagger_all[(j_label, i_label)]
        ac_idagger_jdagger = ac_idagger_jdagger.toarray()
        if not np.allclose(ac_idagger_jdagger, np.zeros((N_DIM, N_DIM))):
            psi_algebra_satisfied_ij = False

        ac_i_jdagger = psi_psi_dagger_all[(i_label, j_label)]+psi_dagger_psi_all[(j_label, i_label)]
        ac_i_jdagger = ac_i_jdagger.toarray()
        if i_label==j_label:
            if not np.allclose(ac_i_jdagger, np.identity(N_DIM)):
                psi_algebra_satisfied_ij = False
        else:
            if not np.allclose(ac_i_jdagger, np.zeros((N_DIM, N_DIM))):
                psi_algebra_satisfied_ij = False

        return psi_algebra_satisfied_ij

    if check_algebra:
        print("Checking fermion algebra...")
        psi_algebra_satisfied = True
        for i_index in tqdm(range(N)):

            # anticommutator(A, B) \equiv anticommutator(A, B), for ANY operators A, B
            # Since these are all anticommutation relations, it's therefore only necessary to check for j>=i 
            # Note that we do have to include j==i because that's in the prescribed anticommutation relations
            psi_algebra_satisfied_i = Parallel(n_jobs=n_jobs)(delayed(process_j)(i_index, j_index) for j_index in range(i_index, N))
            if False in psi_algebra_satisfied_i:
                psi_algebra_satisfied = False
                break

        print(f"Fermion algebra satisfied: {psi_algebra_satisfied}")


    ####################### Load or precompute triple dot-products ##################################
    def psi_psi_psi(psi_psi_ij,k):
        psi_ijk = psi_psi_ij@psi_all[k]
        return psi_ijk

    # psi_psi_psi
    try:
        psi_psi_psi_all = {}
        for i in range(1, N-1):
            for j in range(i+1, N):
                for k in range(j+1, N+1):
                    psi_psi_psi_ijk = np.load(os.path.join(N_DIR, f"psi_psi_psi_{i}{j}{k}.npy"))
                    psi_psi_psi_all[(i,j,k)] = sparse.csr_matrix(psi_psi_psi_ijk)
    except FileNotFoundError:
        print("Computing psi_psi_psi's...")
        psi_psi_psi_all = {}
        for i in range(1, N-1):
            for j in range(i+1, N):
                psi_psi_ij = psi_psi_all[(i,j)]
                psi_psi_psi_ij = Parallel(n_jobs=n_jobs)(delayed(psi_psi_psi)(psi_psi_ij, k) for k in range(j+1, N+1))
                psi_psi_psi_ij = {(i, j, j+k+1):v for k,v in enumerate(psi_psi_psi_ij)}
                for (i_label, j_label, k_label), psi_psi_psi_ijk_sparse in psi_psi_psi_ij.items():
                    psi_psi_psi_ijk = psi_psi_psi_ijk_sparse.toarray()
                    np.save(os.path.join(N_DIR, f"psi_psi_psi_{i_label}{j_label}{k_label}.npy"), psi_psi_psi_ijk)
                psi_psi_psi_all.update(psi_psi_psi_ij)

    out = {"psi_all": psi_all,
           "psi_dagger_all": psi_dagger_all,
           "psi_psi_all": psi_psi_all,
           "psi_psi_dagger_all": psi_psi_dagger_all,
           "psi_dagger_psi_all": psi_dagger_psi_all,
           "psi_dagger_psi_dagger_all": psi_dagger_psi_dagger_all,
           "psi_psi_psi_all": psi_psi_psi_all}
    return out





