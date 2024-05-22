import numpy as np
import torch

def machine_process(file_path):
    """This function is specificaly defined for Jef Heinz & Dakotah Lambert DFA encoding.

    We suppose the transitions are well defined, that is to say for every states and letters we know what to do.
    """
    transitions = []
    finals = []
    alphabet = []
    num_stats = 0
    with open(file_path) as file:
        for line in file.readlines():
            splited_line = line.split()
            if len(splited_line)==4:
                splited_line[0] = int(splited_line[0])
                splited_line[1] = int(splited_line[1])
                transitions.append(splited_line)      ### creating the transition function 1st step
                alphabet.append(splited_line[2])      ### creating the alphabet 1st step
                num_stats = max(num_stats, splited_line[0], splited_line[1])
            else:
                finals.append(int(splited_line[0]))
    num_stats +=1
    alphabet = "".join(sorted(set(alphabet)))
    trans_mat = np.ones((len(alphabet), num_stats)) * (-1)
    decod_mat = np.zeros(num_stats)
    for elem in transitions: ### creating the transition function last step
        letter = elem[2]
        index  = alphabet.index(letter) 
        trans_mat[index,elem[0]] = elem[1]

    if np.any(transitions == -1):
        raise ValueError("It misses transitions in the .att file.")
    
    for elem in finals:
        decod_mat[elem] = 1
            
    return (trans_mat, decod_mat)

DATA_PATH = "./data/Small/"



# (trans_mat, decod_mat, alphabet) = machine_proces(path,name)

# print(trans_mat)
# print(decod_mat)
# print(alphabet)

def dfa2srn(trans_mat, decod_mat, returnJ:bool = False, verbose:bool = False):
    """np.array, np.array -> list(torch.tensors)
    Given an transitions function and a decoder defined by matrices this functions outputs a saturated Simple recurrent network capable of
    simulating the DFA. 
    """
    S,Q = trans_mat.shape
    if verbose:
        print(f"trans_mat.shape = ({S},{Q})")
    J = 128*np.log(2)
    ### Initalizing the parameters

    # The encoder 
    U = torch.zeros((S*Q,S))
    U_b = torch.zeros((S*Q))
    W = 3*torch.ones((S*Q,S*Q))
    W_b = torch.zeros((S*Q))

    # The decoder
    V = torch.zeros(S*Q) #  
    V_c = torch.zeros(1)

    ### Constructing the parameters 

    # The encoder
    for j in range(Q):
        for k in range(S):
            U[k+j*S,k] = 2
    
    for s in range(Q):
        for k in range(S):
            elem = int(trans_mat[k,s])
            
            W[k+elem*S,s*S:(s+1)*S] = 1

    # The decoder
    for k in range(Q):
        if decod_mat[k] == 1:
            V[k*S:(k+1)*S] = decod_mat[k]
        
        else:
            V[k*S:(k+1)*S] = -1
        
        # print(V[k*S:(k+1)*S-1])

    ### Packing all the tensors 
    # target = [J*U, U_b, -J*W, W_b, J*V, V_c]
    target = [U, U_b, -W, W_b, V, V_c]

    if returnJ:
        return target, J
    else:
        return target

# T_m = np.array([[1,0],[0,1]])
# D_m = np.array([0,1])

# T_m, D_m = machine_proces(path,name)

# print(dfa2srn(T_m, D_m))