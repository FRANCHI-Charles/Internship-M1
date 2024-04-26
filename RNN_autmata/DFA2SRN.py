import numpy as np
import torch

path = "/home/volodimir/Bureau/ForLang/data/machines/"
name = "04.04.Zp.2.1.0.att"

def machine_proces(path,name):
    """This function is specificaly defined for Jef Heinz & Dakotah Lambert DFA encoding
    """
    transitions = []
    finals = []
    alphabet = []
    num_stats = 0
    with open(path+name) as file:
        length = len(file.readlines())
    with open(path+name) as file:
        for j in range(length):
            line = file.readline()
            splited_line = line.split()
            if len(splited_line)==4:
                transitions.append(splited_line)      ### creating the transition function 1st step
                alphabet.append(splited_line[2])      ### creating the alphabet 1st step
                if int(splited_line[0])>num_stats:    ### counting the states
                    num_stats = int(splited_line[0])
            else:
                finals.append(int(splited_line[0]))
    num_stats +=1
    alphabet_size = len(transitions)//(num_stats) ## we can do that because we deal with a DFA
    trans_mat = np.zeros((alphabet_size, num_stats)) 
    decod_mat = np.zeros(num_stats)
    alphabet = sorted(list(set(alphabet)))
    for elem in transitions: ### creating the transition function last step
        lettre = elem[2]
        index  = alphabet.index(lettre) 
        trans_mat[index,int(elem[0])] = int(elem[1])
    
    for elem in finals:
        decod_mat[elem] = 1
            
    return (trans_mat, decod_mat)


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
            if verbose:
                print(k+j*Q)
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
        if verbose:
            print(k*S)
            print((k+1)*S-1)

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