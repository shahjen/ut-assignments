#%%
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt

#%%
def run_random_walk(N:int, t:int, initial_state:np.ndarray = None, P:np.ndarray = None):
    """Uses the initial state and transition matrix to form the state at time t 

    Args:
        N (int): _description_
        initial_state (np.ndarray): _description_
        P (np.ndarray): _description_
    """
    
    if P is None:
        P = create_random_walk_transition_matrix(N)
    
    if initial_state is None:
        initial_state = np.zeros(N+1)
        initial_state[N] = 1.
    
    return np.dot(initial_state, np.power(P, t).toarray())

def create_random_walk_transition_matrix(N):
    P = np.zeros((N+1, N+1))
    sup = np.square(np.arange(1, N+1)/N)
    sub = np.square((N-np.arange(N))/N)
    diag = 1 - np.square(np.arange(N+1)/N) - np.square((N-np.arange(N+1))/N)
    diagonals = [diag, sub, sup]
    P = ss.diags(diagonals, [0, 1, -1])
    return P

            
            
def expected_value(N:int, t:int, initial_state:np.ndarray = None, P:np.ndarray = None):
    return  np.dot(run_random_walk(N, t, initial_state=initial_state, P=P), np.arange(N+1))

#%%
if __name__=="__main__":

    for N in [3, 5, 8]:
        P = create_random_walk_transition_matrix(N)
        evals = [expected_value(N, t, P=P) for t in [0, 5, 10, 15, 20, 200]]
        plt.plot(evals)
        print(evals[-1])
    plt.legend([3, 5, 8])
    plt.show()
        
    
            

# %%
