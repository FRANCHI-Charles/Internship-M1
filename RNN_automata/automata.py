import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.utils.parametrize as parametrize
from warnings import warn


class DFA:
    """
    Class of deterministic automaton.
    
    Attributes
    ----------
    initstate : int
        State in which the automaton start.

    transition : np.ndarray
        2D matrix of transitions.
        It has as shape (Q,S), with:
                - Q, number of states
                - S, number of symbols of the automaton

    finites : np.ndarray
        A 1D binary array reprenting if the corresponding state
            is terminal (1) or not (0).

    probas : np.ndarray
        Matrix of probabilites for generation of correct
        and incorrect words. 

    letters : str
        List of the order of the symbol 

    Methods
    -------
    check,
    generate,
    dataset
    """
    
    def __init__(self, transition:np.ndarray, finites:np.ndarray, initstate:int=0, probas:np.ndarray|str=None, letters:str=None):
        """
        Parameters
        ----------
        transition : np.ndarray
            A 2D array representing the transitions of the automaton.
            It has as shape (Q,S), with:
                - Q, number of states
                - S, number of symbols of the automaton

        finites : np.ndarray
            A 1D binary array reprenting if the corresponding state
            is terminal (1) or not (0).

        initstate : int
            The initial state where the automaton starts.
            Default is 0.

        probas : np.ndarray or str, optional
            How to generate words.
            If None :
                It supposed the last state (last rows of `transition`)
                is the 'bin' state, a sink which is rejected.
                That way, from every item of the transition matrix,
                -1 means impossible symbol
            If np.ndarray :
                A 2D array with probas to generate accepted and rejected words.
                It has as shape (Q,S), like `transition`.
            If "equal" :
                All characters have as many chance to occur since each states
                (so bigger probability to have a rejected word)

        letters : str, optional
            Alphabet corresponding to the automaton.
            Default is latin alphabet.
            NOTE : if the number of symbols of the automaton is greater than 26,
                it has to be determinate. 
        """
        self.transition = transition
        self.initstate = initstate
        self.probas = probas
        self.finites = finites
        self.letters = letters

        self.data = None # last dataset generated
        
    
    @property
    def transition(self):
        return self._transition
    
    @transition.setter
    def transition(self, matrix):
        if isinstance(matrix, np.ndarray) and len(matrix.shape) == 2:
            if np.any(matrix >= matrix.shape[0]) or np.any(matrix < -1):
                raise ValueError("A value in `tranisition` is not a possible state.")
            self._transition = matrix.copy()
            self._transition[self._transition == (matrix.shape[0]-1)] = -1 # change possible extreme value as -1 for algorithm simplicity
        else:
            raise TypeError("`transisition` must be a 2D numpy array.")
        
    @property
    def finites(self):
        return self._finites
    
    @finites.setter
    def finites(self, array):
        if isinstance(array, np.ndarray) and len(array.shape) == 1:
            if array.shape[0] != self.transition.shape[0]:
                raise ValueError("The number of states of `finites` must be the same as `transition`")
            if array[-1] == 1 and np.all(self.probas[-1] == 0): # check if probas come from "None" option
                raise ValueError("Such `probas` has been set with None, the bins sink (last state) must be rejected, but got 1.")
            self._finites = array.copy()
        else:
            raise TypeError("`finites` must be a 1D numpy array.")
        
    @property
    def initstate(self):
        return self._initstate
    
    @initstate.setter
    def initstate(self, value):
        if isinstance(value, int):
            if value >= self.transition.shape[0] or value < -1:
                raise ValueError("`initstate` is not a valid state number.")
            self._initstate = value
        else:
            raise TypeError("`inistate` must be int.")
        
    @property
    def probas(self):
        return self._probas

    @probas.setter
    def probas(self, value):
        if isinstance(value, np.ndarray):
            if value.shape == self.transition.shape: # if probas is defined, just copy it
                self._probas = value.copy()
            else:
                raise ValueError("`probas` does not have the same shape as `transition`.")
        elif value == "equal": # if equal, do equal proba for the all matrix
            self._probas = np.ones(self.transition.shape) / self.transition.shape[1]
        elif value is None: # else, do equi-probabilities on element which don't lead to the bin
            if np.any(self.transition[-1] != -1):
                warn("The last row of `transition` is not a sink! Switch `probas` to 'equal'...")
                self.probas = "equal"
                return
            nobinselements = (self.transition != -1) # descriminate elements going to the sink
            for array in nobinselements:
                if np.sum(array, axis=0) == 0: # if all the elements are going to the sink, do equiprobability
                    array[:] = True
            nbrelements = np.sum(nobinselements, axis=1).reshape(-1,1)
            self._probas = nobinselements / nbrelements
        else:
            raise TypeError("`probas` has to be a 2D numpy array, 'equal' or None.")

    @property
    def letters(self):
        return self._letters
    
    @letters.setter
    def letters(self, strings):
        if isinstance(strings, str):
            if len(set(strings)) != len(strings):
                raise ValueError("The symbols in `letters` are not unique!")
            if len(strings) < self.transition.shape[1]:
                raise ValueError("Not enough symbols in `letters`.")
            self._letters = strings[:self.transition.shape[1]]
        elif strings is None:
            if len(string.ascii_lowercase) < self.transition.shape[1]:
                raise ValueError("Not enough symbols in `letters`.")
            self._letters = string.ascii_lowercase[:self.transition.shape[1]]
        else:
            raise TypeError("`letters` have to be str or None.")


        
    def _word_to_coord(self, word:str) -> list:
        return [self.letters.rfind(e) for e in word]

    def check(self, word:str) -> int:
        """
        Check if a word is accepted by the automaton or not.
        
        Parameters
        ----------
        word : str
            word to check.

        Returns
        -------
        binary (int)
            0 if rejected, 1 if accepted
        """
        wordlist = self._word_to_coord(word)
        if -1 in wordlist:
            return 0
        
        state = self._initstate
        for i in wordlist:
            state = self.transition[state,i]

        return self.finites[state]
    

    def generate(self, number:int, meanlen:int = 6, random_state:int = None) -> tuple[list[str], list[int]]:
        """
        Generate a list of accepted and rejected word.
        
        Parameters
        ----------
        number : int
            number of words to generate

        meanlen : int
            Mean length of a word. The length of words will follow
            a Poisson law with lambda = meanlen.

        random_state : int, optional
            Fix the random state for reproducibility.
            You can directly give a random Generator object from numpy.
            
        Returns
        -------
        list
            List of words generated
        list
            Corresponding binary list of accepted (1) or rejected (0)
        """
        rng = self._rng(random_state)
        words = list()
        labels = list()
        for _ in range(number):
            word, acc_rej = self._one_generation(meanlen, rng)
            words.append(word)
            labels.append(acc_rej)
        return words, labels

    def _one_generation(self, meanlen, rng):
        """The algorithm to determine the length of the word is the one find with
        the inverse transform sampling method."""
        word = ""
        p = 1
        state = self.initstate
        p *= rng.random()
        threshold = np.exp(-meanlen)
        while p > threshold:
            isymbol = rng.choice(np.arange(self.transition.shape[1]), p=self.probas[state]) # choose the next symbol in terms of the probas
            word += self.letters[isymbol] #add the symbol to the word
            state = self.transition[state, isymbol] # go next state
            p *= rng.random()
        return word, self.finites[state]
    
    def _rng(self, random_state):
        if isinstance(random_state, int):
            return np.random.default_rng(random_state)
        elif isinstance(random_state, np.random._generator.Generator):
            return random_state
        else:
            return np.random
    

    def dataset(self, length:int, prop:float = 0.5, meanlen:int = 6, max_try:int = None, equalneg:bool = True, random_state:int=None) -> tuple[list[str], list[int]]:
        """
        Generate words and labels, with a total of `length` elements
        with `prop`*`length` positive elements.
        First generate all the positive elements, then select or
        generate negative elements.

        Parameters
        ----------
        length : int
            Final size of the dataset

        prop : float between [0,1], default is 0.5
            Proportion of positive elements

        meanlen : int, default is 6
            Mean length of a word. The length of words will follow
            a Poisson law with lambda = meanlen.

        max_try : int, optional
            The maximum number of word generated to create the dataset in the good proportions.
            Default is : int(length / (1-prop)) * 10

        equalneg : bool, default is True
            If True, when generating negative elements,
            used uniform distribution for symbol selection.

        random_state : int, optional
            Fix the random state for reproducibility.
            You can directly give a random Generator object from numpy.

        Returns
        -------
        list
            List of generated words
        list
            List of words class (1 accepted, 0 rejected)
        """
        rng = self._rng(random_state)

        if prop <0 or prop > 1:
            raise ValueError("`prop` has to be a float in [0,1].") 

        if max_try is None:
            max_try = int(length/(1-prop)) * 10
        elif max_try < length:
            raise ValueError("`max_try` has to be greater than the length.")
        
        n_accepted = int(prop * length)
        n_actual_accepted = 0
        words = list()
        labels = list()
        for itry in range(max_try): # while the n_accepted is not the one required of max_try not reach
            word, label = self._one_generation(meanlen, rng)
            n_actual_accepted += label
            words.append(word)
            labels.append(label)
            if n_accepted == n_actual_accepted:
                break

        if itry == max_try-1:
            words, labels = self._zeroselect(length, words, labels, rng) # remove extrazeros
            warn(f"`max_try` limit reach. Final prop = {sum(labels)/len(labels)}")
        else:
            if len(labels) - length >= 0: # if we have generated enough elements
                words, labels = self._zeroselect(length, words, labels, rng) # remove extra zeros
            else: # else keep generating
                if equalneg: # put probas as equal probas
                    temp_probas = self.probas.copy()
                    self.probas = "equal"
                for itry2 in range(itry+1, max_try):
                    word, label = self._one_generation(meanlen, rng)
                    if label == 0: # keep only negative elements (missing ones)
                        words.append(word)
                        labels.append(label)
                    if len(labels) == length:
                        break
                if equalneg:
                    self._probas = temp_probas # reset probas

                if itry2 == max_try-1:
                    lastwords, lastlabels = self.generate(length - len(labels), meanlen, rng)
                    words += lastwords
                    labels += lastlabels
                    warn(f"`max_try` limit reach. Final prop = {sum(labels)/len(labels)}")

        self.data = (words, labels)
        return self.data
    
    def _zeroselect(self, length, words, labels, rng):
        zerolabels = np.nonzero(np.array(labels) == 0)[0] # zero indices
        to_remove = list(rng.permutation(zerolabels)[:len(labels) - length]) # select a random part

        words = list(np.delete(np.array(words), to_remove)) 
        labels = list(np.delete(np.array(labels), to_remove))

        return words, labels
    

    def word_to_matrix(self, word, length:int=None) -> list[list[int]]:
        """
        One-hot encoding the word in the automaton letters.

        Parameters
        ----------
        word : str
            Automaton word to encode.

        length : int or None
            If set, the length of the return list will be this parameters (all adding list are 0 lists).

        Returns
        -------
        list of lists of int
            One-hot encoded word.
        """
        if isinstance(word, str):
            if length is None:
                return [[1 if symbol == letter else 0 for symbol in self.letters] for letter in word]
            else:
                return [[1 if symbol == letter else 0 for symbol in self.letters] for letter in word] + [[0 for _ in self.letters] for _ in range(length - len(word))]
        else:
            raise TypeError("`word` should be a string.")



        

class TorchData(Dataset):
    """
    Transform an automaton dataset in one-hot letters torch Dataset.
    """

    def __init__(self, automaton: DFA) -> None:
        self.words, self.lengths = self._to_tensor(automaton)
        self.labels = automaton.data[1]

    def _to_tensor(self, automaton:DFA):
        datas = list()
        lengths = list()
        maxlength = self._maxlength(automaton.data[0])
        for word in automaton.data[0]:
            coded = [[1 if symbol == letter else 0 for symbol in automaton.letters] for letter in word]
            lengths.append(len(word))
            coded += [[0 for _ in automaton.letters] for _ in range(maxlength - len(coded))]
            datas.append(torch.tensor(coded, dtype=torch.float32))
        return datas, lengths

    def _maxlength(self, wordslist):
        maxlength = 0
        for i in wordslist:
            maxlength = max(maxlength, len(i))
        return maxlength

    
    def __getitem__(self, index) -> tuple[list,int,int]:
        return self.words[index], self.lengths[index], self.labels[index]
    
    def __len__(self) -> int:
        return len(self.labels)
    


class AutomataRNN(nn.Module):
    """
    Model RNN based on a DFA, tanh version.
    """

    def __init__(self, automaton:DFA, device) -> None:
        super().__init__()
        self.automaton = automaton
        self.transshape = automaton.transition.shape
        self.hidden_size = self.transshape[0]*self.transshape[1]
        self.device = device

        self.rnn = nn.RNN(self.transshape[1], self.hidden_size, num_layers=1, batch_first=True, device=device, bias=False)
        self.toclass = nn.Linear(self.hidden_size, 1, device=device, bias=False)
        self.label = nn.Sigmoid()

    def set_parameters(self, target:list[torch.Tensor]):
        """Set the RNN parameters has the target list"""
        newparam = {"rnn.weight_ih_l0" : target[0],
                      "rnn.weight_hh_l0" : target[2],
                      "toclass.weight" :  target[4]}
        self.load_state_dict(newparam)

    def high_init(self, J=40):
        statedict = self.state_dict()
        idmatrix = torch.zeros(self.hidden_size, self.transshape[1])
        for i in range(0, self.hidden_size, self.transshape[1]):
            idmatrix[i:i+self.transshape[1]] = torch.eye(self.transshape[1], requires_grad=False)
        statedict["rnn.weight_ih_l0"] = J * idmatrix
        statedict["rnn.weight_hh_l0"] = -J * torch.ones((self.hidden_size,self.hidden_size))
        self.load_state_dict(statedict)

    def forward(self, x, truelen):
        "`truelen` is a list of the real length of the sequence : that way we can recover the good prediction along the rnn"
        h0 = -torch.ones(1, x.shape[0], self.hidden_size).to(self.device)
        h0[:,:,0] = 1
        out, _ = self.rnn(x.to(self.device), h0)
        out = torch.stack([out[i, truelen[i] -1, :] for i in range(out.shape[0])]) #extract only the require prediction y for each batch
        return self.label(self.toclass(out).reshape(-1))
        
    def predict(self, x, truelen):
        "Add a round step to forward path to ensure binary classification."
        return torch.round(self(x, truelen))
    
    def strpredict(self, word:str):
        "Given a single word, return the prediction by the RNN."
        tensor = torch.tensor(self.automaton.word_to_matrix(word))
        return self.predict(tensor, len(word))
    


class EqualColumns(nn.Module):
    def __init__(self, RNN:AutomataRNN) -> None:
        "function = 'tanh' or 'sigmoid'."
        super().__init__()
        self.transshape = RNN.transshape
        self.hidden_size = RNN.hidden_size

    def forward(self, X):
        "X is a square matrix of the size of the hidden vector."
        cat = [X[:,i].reshape(-1,1).expand(-1, self.transshape[1])  for i in range(0, self.hidden_size, self.transshape[1])]
        return torch.cat(cat, dim=1)


class InputIdentityShape(nn.Module):
    def __init__(self, RNN:AutomataRNN):
        super().__init__()
        self.matrix = torch.zeros(RNN.hidden_size, RNN.transshape[1])
        for i in range(0, RNN.hidden_size, RNN.transshape[1]):
            self.matrix[i:i+RNN.transshape[1]] = torch.eye(RNN.transshape[1], requires_grad=False)

    def forward(self, X):
        "X has to be the weight_ih matrix of a RNN."
        return X[0,0] * self.matrix

class ParametrizeRNN(AutomataRNN):
    """
    RNN with parameters contrained to has a shape similar of what found Volodimir MITARCHUK for automata.
    """
    def __init__(self, automaton: DFA, device) -> None:
        super().__init__(automaton, device)
        statedict = self.state_dict()
        weight_ih = torch.zeros(self.hidden_size, self.transshape[1])
        weight_ih[0,0] = statedict["rnn.weight_ih_l0"][0,0]
        statedict["rnn.weight_ih_l0"] = weight_ih
        weight_hh = torch.zeros(self.hidden_size, self.hidden_size)# *statedict["rnn.weight_hh_l0"].shape)
        weight_oh = torch.zeros(1, self.hidden_size)
        for i in range(0, self.hidden_size, self.transshape[1]):
            weight_hh[:,i] = statedict["rnn.weight_hh_l0"][:,i]
            weight_oh[:,i] = statedict["toclass.weight"][:,i]
        statedict["rnn.weight_hh_l0"] = weight_hh #put the unused part of the hh matrix to 0
        statedict["toclass.weight"] = weight_oh #put the unused part of the oh matrix to 0
        self.load_state_dict(statedict)
        # self.rnn.all_weights[0][0].requires_grad_(False) # turn off the optimization along weight_ih

        parametrize.register_parametrization(self.rnn, "weight_hh_l0", EqualColumns(self))
        parametrize.register_parametrization(self.toclass, "weight", EqualColumns(self))
        parametrize.register_parametrization(self.rnn, "weight_ih_l0", InputIdentityShape(self))

    def high_init(self, J=40):
        statedict = self.state_dict()
        statedict["rnn.parametrizations.weight_ih_l0.original"][0,0] = J
        for i in range(0, self.hidden_size, self.transshape[1]):
            statedict["rnn.parametrizations.weight_hh_l0.original"][:,i] = -J
        self.load_state_dict(statedict)


class Binary_nthRoot_Loss(nn.Module):
    "Loss on binary classes 0,1 using nth root augmentation."
    def __init__(self, nth:int=2) -> None:
        super().__init__()
        self.power = 1/nth
    
    def forward(self, predictions, labels):
        return torch.mean((1 - labels) * (predictions ** self.power) + labels * ((1 - predictions) **self.power), dim=0)


### Saturated calculator

def dfa2srn(trans_mat:np.ndarray, decod_mat:np.ndarray, returnJ:bool = False, verbose:bool = False) -> list[torch.Tensor]:
    """Given a transitions matrix and a decoder matrix, outputs a saturated (for dtype float32) Simple recurrent network capable of
    simulating the DFA. 
    """
    S,Q = trans_mat.shape
    if verbose:
        print(f"trans_mat.shape = ({S},{Q})")
    J = 128*np.log(2) + 20
    ### Initalizing the parameters

    # The encoder 
    U = torch.zeros((S*Q,S))
    U_b = torch.zeros((S*Q))
    W = 3*torch.ones((S*Q,S*Q))
    W_b = torch.zeros((S*Q))

    # The decoder
    V = torch.zeros((1,S*Q)) # MODIFICATION DONE HERE WARNING
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
            V[0, k*S:(k+1)*S] = 1
        else:
            V[0, k*S:(k+1)*S] = -1
        
        # print(V[k*S:(k+1)*S-1])

    ### Packing all the tensors 
    target = [J*U, U_b, -J*W, W_b, J*V, V_c]
    #target = [U, U_b, -W, W_b, V, V_c]

    if returnJ:
        return target, J
    else:
        return target


def sigmoid_to_tanh_change_basis(lenW:int):
    changebase = torch.zeros((lenW, lenW))
    for i in range(lenW):
        changebase[i,i] = 1 - 1/(lenW - 2)
        for j in range(i+1, lenW):
            changebase[i,j] = changebase[j,i] = - 1/(lenW - 2)
    changebase *= 0.5
    return changebase


def sigmoid_to_tanh(target:list[torch.Tensor]):
    target = target.copy()
    lenW = target[2].shape[0]
    if lenW == 0:
        raise ValueError("Inconsistant W in `target`.")
    elif lenW ==2:
        raise ValueError("Can't use tanh for |states|*|\Sigma| =2.")
    elif lenW ==1:
        return target
    else:
        changebase = sigmoid_to_tanh_change_basis(lenW)

        target[2] @= changebase
        target[4] @= changebase
        return target


if __name__ == "__main__":
    transitions1 = np.array([[2,1,0],
                            [3,1,-1],
                            [0,1,-1],
                            [-1,-1,-1]])
    finites1 = np.array([0,1,0,0])

    transitions2 = np.array([[1,0,0,0],
                            [1,2,0,0],
                            [0,0,0,0]])
    finites2 = np.array([0,0,1])

    transitions3 = np.array([[1,2],
                             [2,0],
                             [2,2]])
    finites3 = np.array([1,0,0])
    
    
    automat = DFA(transitions1, finites1)
    print(automat.transition)
    print(automat.initstate)
    print(automat.finites)
    print(automat.probas)
    print(automat.letters)
    print(type(automat.probas[0,0]))

    print(automat.check("abab"), automat.check("aba"))

    rng = np.random.default_rng(42)

    #words, labels = automat.generate(20, random_state=rng)
    words, labels = automat.dataset(500, prop = 0.5, random_state=8)
    print(len(words), len(labels), sum(labels))
    print(type(words[0]), type(labels[0]))

    datatorch = TorchData(automat)
    size = len(datatorch[0][0])
    print(datatorch[1])
    for a,_,_ in datatorch:
        assert len(a) == size